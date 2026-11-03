"""Summarize exploratory rest-RMS screening across locally available DB2 files.

Run from the repository root:

    python -m neurogrip.ninapro_rest_calibration
"""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np

from neurogrip.ninapro import load_session
from neurogrip.signal_quality import (
    MAX_REST_CALIBRATION_RMS,
    assess_rest_calibration_rms,
)

REST_WINDOW_SAMPLES = 400


def _rest_window_percentiles(emg: np.ndarray, labels: np.ndarray) -> dict | None:
    windows = []
    for start in range(0, labels.size - REST_WINDOW_SAMPLES + 1, REST_WINDOW_SAMPLES):
        stop = start + REST_WINDOW_SAMPLES
        if np.all(labels[start:stop] == 0):
            samples = emg[:, start:stop].astype(np.float64)
            windows.append(np.sqrt(np.mean(np.square(samples), axis=1)))
    if not windows:
        return None

    rms = np.stack(windows)
    return {
        "windowMs": 200,
        "nWindows": int(rms.shape[0]),
        "p50Microvolts": (np.percentile(rms, 50, axis=0) * 1e6).tolist(),
        "p95Microvolts": (np.percentile(rms, 95, axis=0) * 1e6).tolist(),
        "p99Microvolts": (np.percentile(rms, 99, axis=0) * 1e6).tolist(),
        "maxMicrovolts": (np.max(rms, axis=0) * 1e6).tolist(),
    }


def build_report(data_root: pathlib.Path) -> dict:
    paths = sorted(pathlib.Path(data_root).glob("S*_E*_A*.mat"))
    if not paths:
        raise FileNotFoundError(f"no NinaPro DB2 session files under {data_root}")

    aggregates: dict[int, dict[str, object]] = {}
    recordings = []
    for path in paths:
        subject_name = path.name.split("_", 1)[0]
        if not subject_name.startswith("S") or not subject_name[1:].isdigit():
            raise ValueError(f"DB2 filename must start with S<number>_: {path.name}")
        subject = int(subject_name[1:])
        emg, labels, _ = load_session(path)
        rest = emg[:, labels == 0].astype(np.float64)
        aggregate = aggregates.setdefault(
            subject,
            {
                "sumSquares": np.zeros(emg.shape[0], dtype=np.float64),
                "samples": 0,
                "sessions": [],
            },
        )
        if rest.shape[1] > 0:
            aggregate["sumSquares"] += np.sum(np.square(rest), axis=1)
            aggregate["samples"] += rest.shape[1]
        aggregate["sessions"].append(path.name)
        recordings.append(
            {
                "file": path.name,
                "subject": f"S{subject}",
                "restSamples": int(rest.shape[1]),
                "channelRms": (
                    np.sqrt(np.mean(np.square(rest), axis=1)).tolist()
                    if rest.shape[1]
                    else []
                ),
                "restWindowRmsPercentiles": _rest_window_percentiles(emg, labels),
            }
        )

    subjects = []
    for subject, aggregate in sorted(aggregates.items()):
        sample_count = int(aggregate["samples"])
        if sample_count < 1:
            raise ValueError(f"subject S{subject} has no rest-labelled samples")
        rms = np.sqrt(aggregate["sumSquares"] / sample_count)
        result = assess_rest_calibration_rms(rms)
        subjects.append(
            {
                "subject": f"S{subject}",
                "sessions": aggregate["sessions"],
                "restSamples": sample_count,
                "channelRms": list(result.channel_rms),
                "flaggedChannels": [index + 1 for index in result.flagged_channels],
                "passed": result.ok,
            }
        )

    return {
        "thresholdVolts": MAX_REST_CALIBRATION_RMS,
        "method": "RMS across all samples labeled rest in each subject's available files.",
        "thresholdCaveat": (
            "Exploratory 50 uV screen selected on the current development subjects; "
            "not a validated electrode-fault or clinical limit."
        ),
        "restWindowMethod": (
            "Non-overlapping 200 ms windows aligned to each file's start; a window "
            "is included only when every sample has rest label 0."
        ),
        "subjects": subjects,
        "recordings": recordings,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=pathlib.Path,
        default=pathlib.Path("data/ninapro/DB2"),
    )
    parser.add_argument(
        "--out",
        type=pathlib.Path,
        default=pathlib.Path("artifacts/ninapro_rest_calibration.json"),
    )
    args = parser.parse_args()
    report = build_report(args.data_root)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"wrote rest calibration for {len(report['subjects'])} subjects to {args.out}")


if __name__ == "__main__":
    main()
