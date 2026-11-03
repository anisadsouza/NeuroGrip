"""Probe the recorded replay quality gate with controlled faults.

Run from the repository root after installing the research package and staging
the S10 replay manifest:

    python services/research/scripts/ninapro_quality_fault_probe.py
"""

from __future__ import annotations

import argparse
import json
import pathlib
from collections import Counter
from dataclasses import asdict

import numpy as np

from neurogrip.ninapro import DEFAULT_MAPPING, load_session
from neurogrip.signal_quality import SignalQualityConfig, check_signal_quality

WINDOW_SAMPLES = 400
HOP_SAMPLES = 40
FAULTS = ("clean", "flat_channel", "clipping_50uV", "noise_burst_1mV", "jump_10x")


def _recorded_windows(
    root: pathlib.Path,
    subject: int,
    samples_per_label: int,
    seed: int,
) -> list[tuple[int, np.ndarray]]:
    labels = {0, *DEFAULT_MAPPING.values()}
    candidates: dict[int, list[np.ndarray]] = {label: [] for label in labels}
    for path in sorted(root.glob(f"S{subject}_E*_A1.mat")):
        emg, stimulus, repetition = load_session(path)
        for start in range(0, emg.shape[1] - WINDOW_SAMPLES + 1, HOP_SAMPLES):
            stop = start + WINDOW_SAMPLES
            label = int(stimulus[start])
            if (
                label in candidates
                and np.all(stimulus[start:stop] == label)
                and np.all(repetition[start:stop] == repetition[start])
            ):
                candidates[label].append(emg[:, start:stop].copy())

    rng = np.random.default_rng(seed)
    selected: list[tuple[int, np.ndarray]] = []
    for label, available in sorted(candidates.items()):
        if len(available) < samples_per_label:
            raise ValueError(
                f"subject S{subject} has {len(available)} stable windows for "
                f"label {label}; need {samples_per_label}"
            )
        indices = rng.choice(len(available), size=samples_per_label, replace=False)
        selected.extend((label, available[int(index)]) for index in indices)
    return selected


def _inject(channels: np.ndarray, fault: str) -> np.ndarray:
    signal = channels.copy()
    if fault == "flat_channel":
        signal[0] = 0.0
    elif fault == "clipping_50uV":
        np.clip(signal, -50e-6, 50e-6, out=signal)
    elif fault == "noise_burst_1mV":
        signal[0, 100:120] += np.where(np.arange(20) % 2, 1e-3, -1e-3)
    elif fault == "jump_10x":
        signal[:, WINDOW_SAMPLES // 2 :] *= 10.0
    elif fault != "clean":
        raise ValueError(f"unknown fault: {fault}")
    return signal


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=pathlib.Path, default=pathlib.Path("data/ninapro/DB2"))
    parser.add_argument(
        "--manifest",
        type=pathlib.Path,
        default=pathlib.Path("apps/web/public/replay/ninapro/emg-replay.json"),
    )
    parser.add_argument("--samples-per-label", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20261103)
    parser.add_argument(
        "--out",
        type=pathlib.Path,
        default=pathlib.Path("artifacts/ninapro_quality_fault_probe.json"),
    )
    args = parser.parse_args()
    if args.samples_per_label < 1:
        parser.error("--samples-per-label must be positive")

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if manifest.get("source") != "ninapro_db2" or manifest.get("subject") != "S10":
        raise ValueError("the manifest must contain the recorded S10 NinaPro replay")
    floors = tuple(manifest["signalQuality"]["minChannelRmsByChannel"])
    config = SignalQualityConfig(min_channel_rms_by_channel=floors)
    samples = _recorded_windows(
        args.data_root,
        subject=10,
        samples_per_label=args.samples_per_label,
        seed=args.seed,
    )

    print(
        f"S10 recorded 200 ms windows: {len(samples)} "
        f"({args.samples_per_label} per mapped class; seed {args.seed})"
    )
    results: dict[str, dict[str, object]] = {}
    for fault in FAULTS:
        reasons: Counter[str] = Counter()
        for _, signal in samples:
            result = check_signal_quality(_inject(signal, fault), config)
            if not result.ok:
                reasons[result.reason or "unspecified"] += 1
        rejected = sum(reasons.values())
        reason_summary = ", ".join(
            f"{reason}: {count}" for reason, count in sorted(reasons.items())
        ) or "none"
        results[fault] = {
            "rejected": rejected,
            "total": len(samples),
            "rejection_rate": rejected / len(samples),
            "reasons": dict(sorted(reasons.items())),
        }

    report = {
        "source": "ninapro_db2_s10_controlled_fault_injection",
        "subject": "S10",
        "window_samples": WINDOW_SAMPLES,
        "window_ms": WINDOW_SAMPLES / 2,
        "samples_per_label": args.samples_per_label,
        "n_windows": len(samples),
        "seed": args.seed,
        "signal_quality_config": asdict(config),
        "faults": results,
        "caveat": (
            "Controlled artificial faults on selected recorded windows; these "
            "rates are not validated sensitivity/specificity and are not clinical "
            "or live-electrode validation."
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("fault | rejected | total | rejection rate | rejection reasons")
    for fault, result in results.items():
        reasons = result["reasons"]
        assert isinstance(reasons, dict)
        reason_summary = ", ".join(
            f"{reason}: {count}" for reason, count in reasons.items()
        ) or "none"
        print(
            f"{fault} | {result['rejected']} | {result['total']} | "
            f"{result['rejection_rate']:.1%} | {reason_summary}"
        )
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
