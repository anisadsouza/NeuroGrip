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


def write_bundle(out_dir: pathlib.Path) -> dict:
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
    args = parser.parse_args()
    manifest = write_bundle(args.out)
    print(
        f"wrote {len(manifest['gestures'])} gestures x "
        f"{manifest['nChannels']} channels x {manifest['samplesPerGesture']} samples "
        f"({manifest['byteLength'] / 1024:.0f} KB) to {args.out}"
    )


if __name__ == "__main__":
    main()
