"""Simulator signature: a regression pin on generated sEMG behaviour.

The conformance fixtures in fixtures/conformance/ pin Python-vs-TypeScript
feature agreement. They deliberately store their own input waveforms, so they
are insensitive to simulator changes -- editing a gesture synergy leaves them
untouched and passing.

This file closes that gap. It records per-channel RMS for every gesture at a
fixed seed, so any change to the synergy table, the motor-unit pool, the MUAP
shape, or the volume-conduction model must be re-blessed deliberately rather
than landing silently and quietly invalidating every downstream result.

Regenerate only when a simulator change is intended:
    python -m neurogrip.signature --out fixtures/simulator_signature.json
"""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np

from neurogrip.simulator import GESTURES, SimulatorConfig, make_subject, simulate

SUBJECT_SEED = 4242
SIMULATE_SEED = 99
DURATION_S = 0.5
N_ELECTRODES = 8
SAMPLING_RATE_HZ = 2000

DEFAULT_PATH = pathlib.Path("fixtures/simulator_signature.json")


def channel_rms_by_gesture() -> dict[str, list[float]]:
    """Per-channel RMS for each gesture, at the pinned seeds."""
    config = SimulatorConfig(
        sampling_rate_hz=SAMPLING_RATE_HZ, n_electrodes=N_ELECTRODES
    )
    subject = make_subject("signature", seed=SUBJECT_SEED)

    signature: dict[str, list[float]] = {}
    for gesture in GESTURES:
        signal = simulate(gesture, DURATION_S, config, subject, seed=SIMULATE_SEED)
        rms = np.sqrt(np.mean(signal.astype(np.float64) ** 2, axis=1))
        signature[gesture] = [round(float(value), 9) for value in rms]
    return signature


def build_payload() -> dict:
    return {
        "description": (
            "Per-channel RMS per gesture at a fixed seed. Pins simulator "
            "behaviour so a synergy or model change cannot pass silently. "
            "Regenerate deliberately: python -m neurogrip.signature"
        ),
        "subject_seed": SUBJECT_SEED,
        "simulate_seed": SIMULATE_SEED,
        "duration_s": DURATION_S,
        "n_electrodes": N_ELECTRODES,
        "sampling_rate_hz": SAMPLING_RATE_HZ,
        "channel_rms": channel_rms_by_gesture(),
    }


def write_signature(path: pathlib.Path) -> dict:
    payload = build_payload()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Regenerate the simulator signature")
    parser.add_argument("--out", type=pathlib.Path, default=DEFAULT_PATH)
    args = parser.parse_args()
    payload = write_signature(args.out)
    print(f"wrote signature for {len(payload['channel_rms'])} gestures to {args.out}")


if __name__ == "__main__":
    main()
