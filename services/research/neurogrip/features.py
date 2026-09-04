from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import welch


@dataclass(frozen=True)
class FeatureConfig:
    sampling_rate_hz: int = 2000
    zero_crossing_threshold: float = 1e-5
    slope_sign_threshold: float = 1e-5
    psd_bins: int = 6


def extract_features(channels: np.ndarray, config: FeatureConfig) -> dict[str, float]:
    """Extract handcrafted time-domain and spectral features from EMG channels.

    Expected input shape is `(n_channels, n_samples)`.
    """
    signal = _as_channel_matrix(channels)
    features: dict[str, float] = {}

    for channel_index, values in enumerate(signal):
        prefix = f"ch{channel_index + 1}"
        features[f"{prefix}_rms"] = float(np.sqrt(np.mean(np.square(values))))
        features[f"{prefix}_mav"] = float(np.mean(np.abs(values)))
        features[f"{prefix}_wl"] = float(np.sum(np.abs(np.diff(values))))
        features[f"{prefix}_zc"] = float(_zero_crossings(values, config.zero_crossing_threshold))
        features[f"{prefix}_ssc"] = float(_slope_sign_changes(values, config.slope_sign_threshold))

        psd = _power_spectral_density(values, config)
        for bin_index, power in enumerate(psd):
            features[f"{prefix}_psd_{bin_index + 1}"] = float(power)

    return features


def feature_vector(features: dict[str, float]) -> np.ndarray:
    """Return a stable numeric vector by sorting feature names."""
    return np.array([features[name] for name in sorted(features)], dtype=np.float32)


def _as_channel_matrix(channels: np.ndarray) -> np.ndarray:
    signal = np.asarray(channels, dtype=np.float32)
    if signal.ndim != 2:
        raise ValueError("EMG input must have shape (n_channels, n_samples).")
    if signal.shape[0] < 1 or signal.shape[1] < 3:
        raise ValueError("EMG input needs at least one channel and three samples.")
    return signal


def _zero_crossings(values: np.ndarray, threshold: float) -> int:
    previous = values[:-1]
    current = values[1:]
    crossed = np.sign(previous) != np.sign(current)
    large_enough = np.abs(current - previous) >= threshold
    return int(np.sum(crossed & large_enough))


def _slope_sign_changes(values: np.ndarray, threshold: float) -> int:
    previous_diff = values[1:-1] - values[:-2]
    next_diff = values[1:-1] - values[2:]
    changed = (previous_diff * next_diff) >= threshold
    return int(np.sum(changed))


def _power_spectral_density(values: np.ndarray, config: FeatureConfig) -> np.ndarray:
    nperseg = min(len(values), 128)
    _, power = welch(values, fs=config.sampling_rate_hz, nperseg=nperseg)
    if len(power) == 0:
        return np.zeros(config.psd_bins, dtype=np.float32)

    chunks = np.array_split(power, config.psd_bins)
    return np.array([np.mean(chunk) if len(chunk) else 0.0 for chunk in chunks], dtype=np.float32)
