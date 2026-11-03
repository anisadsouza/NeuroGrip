"""Simulator-only error attribution and channel-localized feedback.

The router consumes features standardized from training repetitions. Its four
scores are evidence for a cause, not a clinical or physiological diagnosis.
The demonstration cases below inject known causes so the routing logic can be
tested without claiming that those causes can yet be identified in recordings.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from neurogrip.spec import PER_CHANNEL_FEATURES

MIN_CAUSE_SCORE = 0.45
MIN_CAUSE_MARGIN = 0.10
MIN_RAW_EVIDENCE = 0.25
_VARIANCE_FLOOR = 1e-6

CAUSE_LABELS = {
    "physiological": "Physiological overlap",
    "behavioral": "Behavioral deviation",
    "model_capacity": "Model-capacity limitation",
    "drift": "Signal drift",
    "inconclusive": "Inconclusive",
}

ROUTES = {
    "physiological": (
        "Review this gesture pair and consider selecting or synthesizing "
        "a more separable target."
    ),
    "behavioral": (
        "Use a low-pressure repeat drill for the intended gesture, comparing "
        "the attempt with the user's own stable repetitions."
    ),
    "model_capacity": (
        "Compare the deployed decoder with a stronger reference model before "
        "considering model adaptation."
    ),
    "drift": (
        "Pause for recalibration and inspect fatigue or electrode placement "
        "before adapting the decoder."
    ),
    "inconclusive": (
        "Gather more repeated attempts and review the evidence; no single "
        "cause is supported strongly enough to route automatically."
    ),
}

_RING_LOCATIONS = (
    "top",
    "upper-right",
    "right",
    "lower-right",
    "bottom",
    "lower-left",
    "left",
    "upper-left",
)


@dataclass(frozen=True)
class ErrorObservation:
    """One error and its training-derived reference evidence.

    Feature vectors are channel-major and standardized with statistics fitted
    on training repetitions only. `intended_repetitions` and
    `decoded_repetitions` are the user's cleanest available repeat clusters.
    """

    name: str
    intended: str
    decoded: str
    observed: np.ndarray
    intended_repetitions: np.ndarray
    decoded_repetitions: np.ndarray
    reference_prediction: str | None
    drift_cusum: float = 0.0
    drift_threshold: float = 1.0
    mdf_shift_hz: float = 0.0
    mdf_threshold_hz: float = 1.0


def _validate(observation: ErrorObservation) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    observed = np.asarray(observation.observed, dtype=np.float64)
    intended = np.asarray(observation.intended_repetitions, dtype=np.float64)
    decoded = np.asarray(observation.decoded_repetitions, dtype=np.float64)
    if observed.ndim != 1 or intended.ndim != 2 or decoded.ndim != 2:
        raise ValueError("features must be one vector and two repetition matrices")
    if intended.shape[1] != observed.size or decoded.shape[1] != observed.size:
        raise ValueError("observation and repetition feature counts must match")
    if observed.size == 0 or observed.size % len(PER_CHANNEL_FEATURES) != 0:
        raise ValueError(
            "feature count must contain complete channel-major feature groups"
        )
    if intended.shape[0] < 2 or decoded.shape[0] < 2:
        raise ValueError("each class needs at least two reference repetitions")
    if not all(np.isfinite(values).all() for values in (observed, intended, decoded)):
        raise ValueError("features must be finite")
    if observation.drift_cusum < 0 or observation.drift_threshold <= 0:
        raise ValueError("CUSUM must be non-negative with a positive threshold")
    if observation.mdf_threshold_hz <= 0:
        raise ValueError("median-frequency threshold must be positive")
    return observed, intended, decoded


def bhattacharyya_overlap(first: np.ndarray, second: np.ndarray) -> float:
    """Diagonal-Gaussian Bhattacharyya coefficient in [0, 1].

    The coefficient is one for indistinguishable class clusters and approaches
    zero as their means separate. Diagonal covariance keeps the estimate stable
    when the feature count exceeds the number of clean repetitions.
    """
    a = np.asarray(first, dtype=np.float64)
    b = np.asarray(second, dtype=np.float64)
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[1] or a.shape[0] < 2 or b.shape[0] < 2:
        raise ValueError("overlap needs two repetition matrices with at least two rows")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("repetition features must be finite")

    mean_a = a.mean(axis=0)
    mean_b = b.mean(axis=0)
    var_a = np.maximum(a.var(axis=0, ddof=1), _VARIANCE_FLOOR)
    var_b = np.maximum(b.var(axis=0, ddof=1), _VARIANCE_FLOOR)
    average = (var_a + var_b) / 2.0
    mean_term = np.sum(np.square(mean_a - mean_b) / (8.0 * average))
    covariance_term = 0.5 * np.sum(
        np.log(average) - 0.5 * (np.log(var_a) + np.log(var_b))
    )
    return float(np.clip(np.exp(-(mean_term + covariance_term)), 0.0, 1.0))


def _tightest_repeats(samples: np.ndarray, retain_fraction: float = 0.8) -> np.ndarray:
    center = samples.mean(axis=0)
    scale = np.sqrt(np.maximum(samples.var(axis=0, ddof=1), _VARIANCE_FLOOR))
    distance = np.sqrt(np.mean(np.square((samples - center) / scale), axis=1))
    keep = max(2, int(np.ceil(samples.shape[0] * retain_fraction)))
    selected = np.argsort(distance, kind="stable")[:keep]
    return samples[selected]


def _channel_deviations(
    observed: np.ndarray, repetitions: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    channels = observed.size // len(PER_CHANNEL_FEATURES)
    features = len(PER_CHANNEL_FEATURES)
    mean = repetitions.mean(axis=0).reshape(channels, features)
    variance = np.maximum(
        repetitions.var(axis=0, ddof=1).reshape(channels, features),
        _VARIANCE_FLOOR,
    )
    deviation = (observed.reshape(channels, features) - mean) / np.sqrt(variance)
    channel_rms = np.sqrt(np.mean(np.square(deviation), axis=1))

    historical = (repetitions - repetitions.mean(axis=0)).reshape(
        repetitions.shape[0], channels, features
    )
    historical_rms = np.sqrt(
        np.mean(np.square(historical / np.sqrt(variance)[None, :, :]), axis=2)
    )
    spread = float(np.quantile(historical_rms, 0.95))
    return channel_rms, np.sign(deviation.mean(axis=1)), max(spread, 1e-6)


def _coaching(channel_rms: np.ndarray, channel_direction: np.ndarray) -> list[dict[str, object]]:
    peak = float(channel_rms.max())
    if peak <= 0:
        return []
    strongest = np.argsort(channel_rms)[::-1][: min(3, channel_rms.size)]
    return [
        {
            "channel": int(index + 1),
            "ring_location": _RING_LOCATIONS[index % len(_RING_LOCATIONS)],
            "deviation": round(float(channel_rms[index] / peak), 3),
            "direction": "above" if channel_direction[index] >= 0 else "below",
        }
        for index in strongest
    ]


def attribute_error(observation: ErrorObservation) -> dict[str, object]:
    """Score and route one misclassification from independent evidence sources."""
    observed, intended, decoded = _validate(observation)
    if observation.intended == observation.decoded:
        raise ValueError("attribution requires a misclassification")

    overlap = bhattacharyya_overlap(
        _tightest_repeats(intended), _tightest_repeats(decoded)
    )
    channel_rms, channel_direction, historical_spread = _channel_deviations(
        observed, intended
    )
    behavioral_ratio = float(channel_rms.max() / historical_spread)
    behavioral = float(np.clip((behavioral_ratio - 1.0) / (behavioral_ratio + 1.0), 0.0, 1.0))
    model_capacity = float(observation.reference_prediction == observation.intended)
    centroid_shift = float(
        np.clip(observation.drift_cusum / observation.drift_threshold, 0.0, 1.0)
    )
    spectral_shift = float(
        np.clip(abs(observation.mdf_shift_hz) / observation.mdf_threshold_hz, 0.0, 1.0)
    )
    drift = 0.5 * centroid_shift + 0.5 * spectral_shift

    raw = {
        "physiological": overlap,
        "behavioral": behavioral,
        "model_capacity": model_capacity,
        "drift": drift,
    }
    total = sum(raw.values())
    if total <= 1e-12:
        scores = {cause: 0.25 for cause in raw}
        cause = "inconclusive"
    else:
        scores = {cause: value / total for cause, value in raw.items()}
        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        lead = ranked[0][1] - ranked[1][1]
        cause = (
            ranked[0][0]
            if (
                max(raw.values()) >= MIN_RAW_EVIDENCE
                and ranked[0][1] >= MIN_CAUSE_SCORE
                and lead >= MIN_CAUSE_MARGIN
            )
            else "inconclusive"
        )

    channels = _coaching(channel_rms, channel_direction)
    if cause == "behavioral" and channels:
        lead_channel = channels[0]
        coaching = (
            f"For {observation.intended.replace('_', ' ')} practice, compare "
            f"your repeat with channel {lead_channel['channel']} near the "
            f"{lead_channel['ring_location']} of the electrode ring. This is "
            "channel-level feedback; it does not identify a specific muscle "
            "or finger."
        )
    else:
        coaching = None

    return {
        "name": observation.name,
        "intended": observation.intended,
        "decoded": observation.decoded,
        "cause": cause,
        "cause_label": CAUSE_LABELS[cause],
        "scores": {key: round(float(value), 4) for key, value in scores.items()},
        "evidence": {
            "class_overlap": round(overlap, 4),
            "behavioral_deviation_ratio": round(behavioral_ratio, 4),
            "reference_correct_when_deployed_wrong": bool(model_capacity),
            "centroid_cusum_ratio": round(centroid_shift, 4),
            "median_frequency_shift_ratio": round(spectral_shift, 4),
        },
        "routed_action": ROUTES[cause],
        "coaching": coaching,
        "channel_attribution": channels,
    }


def _repetitions(center: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    return center + rng.normal(0.0, 0.25, size=(24, center.size))


def attribution_demo(seed: int = 17) -> dict[str, object]:
    """Create four transparent, controlled fault injections for the demo UI."""
    rng = np.random.default_rng(seed)
    feature_count = 8 * len(PER_CHANNEL_FEATURES)
    channel_features = len(PER_CHANNEL_FEATURES)
    zero = np.zeros(feature_count)
    cases: list[ErrorObservation] = []

    # The cleanest class clusters deliberately overlap; neither deployed nor
    # reference prediction can resolve the ambiguity.
    physiology_target = zero.copy()
    physiology_rival = physiology_target.copy()
    physiology_repeats = _repetitions(physiology_target, rng)
    cases.append(
        ErrorObservation(
            "physiological_overlap",
            "fist",
            "spherical_grip",
            physiology_target + rng.normal(0.0, 0.15, feature_count),
            physiology_repeats,
            physiology_repeats.copy(),
            "spherical_grip",
        )
    )

    # One electrode-ring region is displaced from otherwise separated classes.
    behavioral_target = zero.copy()
    behavioral_rival = zero.copy()
    behavioral_rival[:channel_features] = 4.0
    behavioral_sample = behavioral_target.copy()
    behavioral_sample[3 * channel_features : 4 * channel_features] += 2.5
    cases.append(
        ErrorObservation(
            "behavioral_deviation",
            "point",
            "open_hand",
            behavioral_sample,
            _repetitions(behavioral_target, rng),
            _repetitions(behavioral_rival, rng),
            "open_hand",
        )
    )

    # A stronger reference classifier succeeds on a clean, in-distribution
    # point that the deployed classifier misses.
    capacity_target = zero.copy()
    capacity_rival = zero.copy()
    capacity_rival[:channel_features] = 4.0
    cases.append(
        ErrorObservation(
            "model_capacity",
            "pinch",
            "point",
            capacity_target + rng.normal(0.0, 0.1, feature_count),
            _repetitions(capacity_target, rng),
            _repetitions(capacity_rival, rng),
            "pinch",
        )
    )

    # A moderate feature displacement accompanies strong sequential and
    # spectral drift evidence, as expected for a drift-dominant case.
    drift_target = zero.copy()
    drift_rival = zero.copy()
    drift_rival[:channel_features] = 4.0
    drift_sample = drift_target.copy()
    drift_sample[channel_features : 2 * channel_features] += 0.65
    cases.append(
        ErrorObservation(
            "signal_drift",
            "wrist_flexion",
            "wrist_extension",
            drift_sample,
            _repetitions(drift_target, rng),
            _repetitions(drift_rival, rng),
            "wrist_extension",
            drift_cusum=8.0,
            drift_threshold=8.0,
            mdf_shift_hz=-20.0,
            mdf_threshold_hz=20.0,
        )
    )

    expected_causes = (
        "physiological",
        "behavioral",
        "model_capacity",
        "drift",
    )
    results = []
    for case, expected in zip(cases, expected_causes):
        result = attribute_error(case)
        results.append(
            {
                **result,
                "injected_cause": expected,
                "route_matches_injection": result["cause"] == expected,
            }
        )
    return {
        "source": "synthetic_fault_injections",
        "seed": seed,
        "feature_count": feature_count,
        "channels": feature_count // channel_features,
        "feature_order": "channel-major",
        "method": (
            "P: diagonal-Gaussian Bhattacharyya overlap of each class's tightest "
            "80% repeat cluster; "
            "B: channel-wise standardized deviation from intended-class repeats; "
            "M: stronger reference prediction is correct while deployed prediction "
            "is wrong; D: centroid CUSUM and median-frequency shift. Scores are "
            "normalized, but require minimum raw evidence and a lead margin to route."
        ),
        "injected_causes_routed": sum(
            bool(result["route_matches_injection"]) for result in results
        ),
        "n_injections": len(results),
        "cases": results,
        "caveat": (
            "All four causes are controlled synthetic injections. This checks "
            "routing logic only; it is not attribution accuracy on human or "
            "NinaPro errors. The channel ring localizes deviations to an "
            "electrode position, not a specific muscle or finger. Thresholds "
            "are exploratory, not validated for clinical or live-device use."
        ),
    }
