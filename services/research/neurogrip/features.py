"""Handcrafted sEMG feature extraction.

Mirrored bit-for-bit in packages/core/src/features.ts and pinned by the golden
vectors in fixtures/conformance/. Feature names and ordering come from
neurogrip.spec and must never be re-derived here.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from neurogrip import dsp
from neurogrip.spec import AR_ORDER, PSD_BINS, feature_names


@dataclass(frozen=True)
class FeatureConfig:
    sampling_rate_hz: int = 2000
    zero_crossing_threshold: float = 1e-5
    slope_sign_threshold: float = 1e-5


def extract_features(channels: np.ndarray, config: FeatureConfig) -> dict[str, float]:
    """Extract the 17 spec features per channel from one analysis window.

    Expected input shape is (n_channels, n_samples).
    """
    signal = _as_channel_matrix(channels)
    features: dict[str, float] = {}

    for index, values in enumerate(signal):
        prefix = f"ch{index + 1}"
        window = np.asarray(values, dtype=np.float64)

        features[f"{prefix}_rms"] = float(np.sqrt(np.mean(np.square(window))))
        features[f"{prefix}_mav"] = float(np.mean(np.abs(window)))
        features[f"{prefix}_wl"] = float(np.sum(np.abs(np.diff(window))))
        features[f"{prefix}_zc"] = float(
            _zero_crossings(window, config.zero_crossing_threshold)
        )
        features[f"{prefix}_ssc"] = float(
            _slope_sign_changes(window, config.slope_sign_threshold)
        )

        coefficients = dsp.levinson_durbin(dsp.autocorrelate(window, AR_ORDER), AR_ORDER)
        for order, coefficient in enumerate(coefficients, start=1):
            features[f"{prefix}_ar{order}"] = float(coefficient)

        freqs, psd = dsp.periodogram(window, float(config.sampling_rate_hz))
        for band, power in enumerate(dsp.band_means(psd, PSD_BINS), start=1):
            features[f"{prefix}_psd{band}"] = float(power)

        features[f"{prefix}_mdf"] = dsp.median_frequency(freqs, psd)
        features[f"{prefix}_mnf"] = dsp.mean_frequency(freqs, psd)

    return features


def feature_vector(features: dict[str, float], n_channels: int) -> np.ndarray:
    """Numeric vector in canonical spec order.

    Never sorts. sorted() places ch10_* before ch2_*, silently permuting the
    vector for any recording with ten or more channels.
    """
    return np.array(
        [features[name] for name in feature_names(n_channels)], dtype=np.float32
    )


def _as_channel_matrix(channels: np.ndarray) -> np.ndarray:
    signal = np.asarray(channels, dtype=np.float32)
    if signal.ndim != 2:
        raise ValueError("EMG input must have shape (n_channels, n_samples).")
    if signal.shape[0] < 1 or signal.shape[1] < 3:
        raise ValueError("EMG input needs at least one channel and three samples.")
    return signal


def _zero_crossings(values: np.ndarray, threshold: float) -> int:
    """Count genuine sign changes exceeding an amplitude threshold.

    Uses a strict product-negative test rather than comparing np.sign values:
    np.sign(0.0) is 0, so a sample resting exactly at zero would otherwise be
    counted as crossing against both of its neighbours.
    """
    previous = values[:-1]
    current = values[1:]
    crossed = (previous * current) < 0.0
    large_enough = np.abs(current - previous) >= threshold
    return int(np.count_nonzero(crossed & large_enough))


def _slope_sign_changes(values: np.ndarray, threshold: float) -> int:
    """Count direction reversals whose limbs exceed an amplitude threshold.

    The sign test runs on the product of the two differences (dimensionless);
    the magnitude test runs on the differences themselves, which share units
    with the threshold. Comparing the threshold directly against the product,
    as the previous implementation did, is dimensionally inconsistent: the
    product carries amplitude-squared units, so the same threshold gates almost
    nothing at millivolt amplitudes and almost everything at microvolt ones.
    """
    if values.size < 3:
        return 0
    previous_diff = values[1:-1] - values[:-2]
    next_diff = values[1:-1] - values[2:]
    reversed_direction = (previous_diff * next_diff) > 0.0
    large_enough = np.maximum(np.abs(previous_diff), np.abs(next_diff)) >= threshold
    return int(np.count_nonzero(reversed_direction & large_enough))
