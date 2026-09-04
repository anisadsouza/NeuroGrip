"""Golden-vector generator: the Python side of the conformance contract.

Pins both inputs and expected outputs so the TypeScript port has something
unambiguous to be wrong against. Deliberately includes the awkward cases -
silence, DC offset, a pure tone, a signal that touches zero without crossing,
and a twelve-channel case that exposes lexicographic feature ordering.

Regenerate whenever FEATURE_SPEC_VERSION changes:
    python -m neurogrip.fixtures --out fixtures/conformance/golden.json
"""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np

from neurogrip import FEATURE_SPEC_VERSION
from neurogrip.features import FeatureConfig, extract_features, feature_vector
from neurogrip.simulator import SimulatorConfig, make_subject, simulate
from neurogrip.spec import PER_CHANNEL_FEATURES, feature_names

SAMPLING_RATE_HZ = 2000
WINDOW_SAMPLES = 400  # 200 ms at 2 kHz
ZERO_CROSSING_THRESHOLD = 1e-5
SLOPE_SIGN_THRESHOLD = 1e-5


def _simulated(gesture: str, n_electrodes: int, seed: int) -> np.ndarray:
    config = SimulatorConfig(
        sampling_rate_hz=SAMPLING_RATE_HZ, n_electrodes=n_electrodes
    )
    subject = make_subject(f"fixture-{seed}", seed=seed)
    signal = simulate(
        gesture, WINDOW_SAMPLES / SAMPLING_RATE_HZ, config, subject, seed=seed
    )
    return signal[:, :WINDOW_SAMPLES]


def _synthetic_cases() -> dict[str, np.ndarray]:
    t = np.arange(WINDOW_SAMPLES, dtype=np.float64) / SAMPLING_RATE_HZ
    ramp = np.linspace(-0.02, 0.02, WINDOW_SAMPLES)

    # Alternates 0.01, 0, 0.01, 0 ...: touches zero repeatedly but never crosses.
    touching = np.zeros(WINDOW_SAMPLES, dtype=np.float64)
    touching[::2] = 0.01

    return {
        "silent": np.zeros((2, WINDOW_SAMPLES)),
        "dc_offset": np.full((2, WINDOW_SAMPLES), 0.05),
        "pure_tone": np.stack(
            [np.sin(2 * np.pi * 120.0 * t) * 0.01, np.cos(2 * np.pi * 40.0 * t) * 0.01]
        ),
        "touches_zero": np.stack([touching, -touching]),
        "linear_ramp": np.stack([ramp, ramp[::-1]]),
    }


def build_fixture_payload() -> dict:
    config = FeatureConfig(
        sampling_rate_hz=SAMPLING_RATE_HZ,
        zero_crossing_threshold=ZERO_CROSSING_THRESHOLD,
        slope_sign_threshold=SLOPE_SIGN_THRESHOLD,
    )

    sources: list[tuple[str, np.ndarray]] = [
        ("simulated_rest_8ch", _simulated("rest", 8, seed=101)),
        ("simulated_fist_8ch", _simulated("fist", 8, seed=102)),
        ("simulated_pinch_8ch", _simulated("pinch", 8, seed=103)),
        ("simulated_wrist_extension_8ch", _simulated("wrist_extension", 8, seed=104)),
        # Twelve channels: lexicographic ordering would emit ch10 before ch2 here.
        ("twelve_channels", _simulated("spherical_grip", 12, seed=105)),
        ("single_channel", _simulated("fist", 1, seed=106)),
    ]
    sources.extend(_synthetic_cases().items())

    cases = []
    for name, channels in sources:
        matrix = np.asarray(channels, dtype=np.float32)
        n_channels = matrix.shape[0]
        vector = feature_vector(extract_features(matrix, config), n_channels)
        cases.append(
            {
                "name": name,
                "samplingRateHz": SAMPLING_RATE_HZ,
                "channels": [[float(v) for v in row] for row in matrix],
                "featureNames": list(feature_names(n_channels)),
                "featureVector": [float(v) for v in vector],
            }
        )

    return {
        "featureSpecVersion": FEATURE_SPEC_VERSION,
        "perChannelFeatures": list(PER_CHANNEL_FEATURES),
        "config": {
            "samplingRateHz": SAMPLING_RATE_HZ,
            "zeroCrossingThreshold": ZERO_CROSSING_THRESHOLD,
            "slopeSignThreshold": SLOPE_SIGN_THRESHOLD,
        },
        "cases": cases,
    }


def write_fixtures(path: pathlib.Path) -> dict:
    payload = build_fixture_payload()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate conformance golden vectors")
    parser.add_argument(
        "--out",
        type=pathlib.Path,
        default=pathlib.Path("fixtures/conformance/golden.json"),
    )
    args = parser.parse_args()
    payload = write_fixtures(args.out)
    print(
        f"wrote {len(payload['cases'])} cases "
        f"(spec v{payload['featureSpecVersion']}) to {args.out}"
    )


if __name__ == "__main__":
    main()
