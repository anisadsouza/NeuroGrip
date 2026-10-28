"""Replay bundle export: simulated sEMG the browser can stream.

The simulator is Python and cannot run in the browser. Porting it would mean
maintaining a second implementation and owing it a second conformance gate, for
a component that only ever produces test input. Instead the browser replays a
recording: this module writes a compact bundle of simulated signal, one segment
per gesture, that a `ReplaySource` streams at the sampling rate.

This is honest about what it is. The architecture always had three signal
sources -- simulator, dataset replay, and a future hardware device -- and replay
is the one the browser uses. When real electrodes arrive they replace this
source and nothing downstream changes.

Samples are quantised to int16 with an explicit volts-per-count scale. The
simulator's contractions run 158-354 uV RMS, but sEMG is spiky and the
instantaneous peaks reach about 2.8 mV. At 4e-7 V per count the full-scale
range is +/-13.1 mV, leaving roughly 4.7x headroom over those peaks -- enough
that a louder subject cannot clip. The quantisation step is 0.4 uV against a
15 uV baseline noise floor, so the added quantisation noise is under 1% of the
noise already present.

    python -m neurogrip.replay --out apps/web/public/replay
"""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np

from neurogrip import FEATURE_SPEC_VERSION
from neurogrip.anatomy import muscle_group_weights
from neurogrip.ninapro import DATASET_MOVEMENTS, DEFAULT_MAPPING, load_session
from neurogrip.simulator import (
    GESTURES,
    SimulatorConfig,
    make_subject,
    simulate,
)

#: Volts per int16 count. Sized against measured instantaneous peaks (~2.8 mV),
#: not against RMS -- an RMS-derived scale clips on the spikes.
VOLTS_PER_COUNT = 4e-7
SUBJECT_SEED = 8080
SEGMENT_SECONDS = 1.5


def build_bundle(
    duration_s: float = SEGMENT_SECONDS,
    n_electrodes: int = 8,
    sampling_rate_hz: int = 2000,
    subject_seed: int = SUBJECT_SEED,
) -> tuple[dict, np.ndarray]:
    """Return (manifest, int16 samples) for every gesture.

    Sample layout is gesture-major, then channel-major, then time:
    `samples[gesture][channel][t]` flattened in C order.
    """
    config = SimulatorConfig(
        sampling_rate_hz=sampling_rate_hz, n_electrodes=n_electrodes
    )
    subject = make_subject("replay", seed=subject_seed)
    n_samples = int(round(duration_s * sampling_rate_hz))

    segments = []
    for index, gesture in enumerate(GESTURES):
        signal = simulate(
            gesture,
            duration_s,
            config,
            subject,
            seed=90_000 + index,
            # A little fatigue, so the replay is not implausibly pristine.
            fatigue=0.15,
        )
        segments.append(signal[:, :n_samples])

    stacked = np.stack(segments)  # (gestures, channels, samples)
    counts = np.round(stacked / VOLTS_PER_COUNT)

    peak = float(np.max(np.abs(counts)))
    if peak > np.iinfo(np.int16).max:
        raise ValueError(
            f"replay would clip: peak {peak:.0f} counts exceeds int16. "
            f"Raise VOLTS_PER_COUNT above {VOLTS_PER_COUNT}."
        )

    manifest = {
        "formatVersion": 1,
        "featureSpecVersion": FEATURE_SPEC_VERSION,
        "samplingRateHz": sampling_rate_hz,
        "nChannels": n_electrodes,
        # Per-electrode weighting for each functional muscle group, so the
        # browser can show which muscles are driving the hand without carrying
        # a second copy of the forearm anatomy.
        "muscleGroups": muscle_group_weights(n_electrodes),
        "samplesPerGesture": n_samples,
        "voltsPerCount": VOLTS_PER_COUNT,
        "dtype": "int16",
        "layout": "gesture-major, then channel-major, then time",
        "gestures": list(GESTURES),
        "subjectSeed": subject_seed,
        "peakCounts": int(peak),
        "source": "neurogrip.simulator",
        "caveat": (
            "Simulated signal, not a human recording. Present it as such "
            "wherever it is played back."
        ),
    }
    return manifest, counts.astype(np.int16)


def build_ninapro_bundle(
    root: pathlib.Path,
    subject: int,
    duration_s: float = SEGMENT_SECONDS,
    mapping: dict[str, int] = DEFAULT_MAPPING,
    trained_subjects: tuple[int, ...] = (),
) -> tuple[dict, np.ndarray]:
    """Cut one complete performed segment per gesture from a DB2 subject.

    The selected person must be excluded from decoder training. Unlike the
    simulator bundle, this manifest intentionally omits muscle-group weights:
    simulated anatomy weights would falsely imply an anatomical measurement.
    """
    if subject in trained_subjects:
        raise ValueError("NinaPro replay subject must be excluded from training")
    paths = sorted(pathlib.Path(root).glob(f"S{subject}_E*_A*.mat"))
    if not paths:
        raise FileNotFoundError(f"no DB2 sessions found for subject S{subject}")
    n_samples = int(round(duration_s * 2000))
    raw_labels = {"rest": 0, **mapping}
    segments: dict[str, np.ndarray] = {}
    for path in paths:
        emg, labels, _ = load_session(path)
        for gesture, raw_label in raw_labels.items():
            if gesture in segments:
                continue
            starts = np.flatnonzero(labels == raw_label)
            for start in starts:
                stop = int(start) + n_samples
                if stop <= labels.size and np.all(labels[start:stop] == raw_label):
                    segments[gesture] = emg[:, start:stop]
                    break
    missing = [gesture for gesture in GESTURES if gesture not in segments]
    if missing:
        raise ValueError(
            f"subject S{subject} has no {duration_s:g}s continuous segment for: {missing}"
        )

    stacked = np.stack([segments[gesture] for gesture in GESTURES])
    counts = np.round(stacked / VOLTS_PER_COUNT)
    peak = float(np.max(np.abs(counts)))
    if peak > np.iinfo(np.int16).max:
        raise ValueError("NinaPro replay would clip; increase VOLTS_PER_COUNT")
    rest_rms = np.sqrt(np.mean(segments["rest"].astype(np.float64) ** 2, axis=1))
    if not np.all(np.isfinite(rest_rms)) or np.any(rest_rms <= 0):
        raise ValueError("NinaPro rest segment must provide a finite baseline on every channel")
    min_channel_rms = np.minimum(5e-6, rest_rms * 0.25)
    manifest = {
        "formatVersion": 1,
        "featureSpecVersion": FEATURE_SPEC_VERSION,
        "samplingRateHz": 2000,
        "nChannels": 8,
        "samplesPerGesture": n_samples,
        "voltsPerCount": VOLTS_PER_COUNT,
        "dtype": "int16",
        "layout": "gesture-major, then channel-major, then time",
        "gestures": list(GESTURES),
        "gestureLabels": {"rest": "Rest", **DATASET_MOVEMENTS},
        "signalQuality": {
            "minChannelRmsByChannel": min_channel_rms.tolist(),
            "method": (
                "Per-channel minimum is 25% of RMS measured in this subject's "
                "recorded rest segment, capped at the shared 5 uV minimum. "
                "This replay calibration is not validated for live electrodes."
            ),
        },
        "subject": f"S{subject}",
        "trainedSubjects": [f"S{value}" for value in sorted(trained_subjects)],
        "peakCounts": int(peak),
        "source": "ninapro_db2",
        "caveat": (
            "Recorded NinaPro DB2 signal. This subject is excluded from the named "
            "training subjects; "
            "no simulated muscle-group weights are shown for recorded electrodes."
        ),
    }
    return manifest, counts.astype(np.int16)


def write_bundle(
    out_dir: pathlib.Path, *, source: str = "simulator", root: pathlib.Path | None = None,
    subject: int | None = None, trained_subjects: tuple[int, ...] = (),
) -> dict:
    if source == "ninapro":
        if root is None or subject is None:
            raise ValueError("NinaPro replay needs a data root and subject")
        manifest, samples = build_ninapro_bundle(
            root, subject, trained_subjects=trained_subjects
        )
    else:
        manifest, samples = build_bundle()
    out_dir.mkdir(parents=True, exist_ok=True)

    binary = out_dir / "emg-replay.bin"
    binary.write_bytes(samples.astype("<i2").tobytes())

    manifest["byteLength"] = binary.stat().st_size
    (out_dir / "emg-replay.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Export the browser replay bundle")
    parser.add_argument(
        "--out", type=pathlib.Path, default=pathlib.Path("apps/web/public/replay")
    )
    parser.add_argument("--source", choices=["simulator", "ninapro"], default="simulator")
    parser.add_argument("--data-root", type=pathlib.Path, default=pathlib.Path("data/ninapro/DB2"))
    parser.add_argument("--subject", type=int, default=None)
    args = parser.parse_args()
    if args.source == "ninapro" and args.subject is None:
        parser.error("--source ninapro requires --subject S<number> as a number, e.g. --subject 12")
    manifest = write_bundle(args.out, source=args.source, root=args.data_root, subject=args.subject)
    print(
        f"wrote {len(manifest['gestures'])} gestures x "
        f"{manifest['nChannels']} channels x {manifest['samplesPerGesture']} samples "
        f"({manifest['byteLength'] / 1024:.0f} KB) to {args.out}"
    )


if __name__ == "__main__":
    main()
