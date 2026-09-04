"""Framing a continuous multi-channel stream into overlapping analysis windows.

Mirrored in packages/core/src/windowing.ts. The 20 ms hop (rather than the
100 ms in the project report) is what makes progressive actuation possible:
at a 10 Hz decision rate there is no evidence to accumulate between decisions.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class WindowConfig:
    window_ms: int = 200
    hop_ms: int = 20
    sampling_rate_hz: int = 2000

    @property
    def window_samples(self) -> int:
        if self.window_ms <= 0:
            raise ValueError("window_ms must be positive")
        return int(self.window_ms * self.sampling_rate_hz / 1000)

    @property
    def hop_samples(self) -> int:
        if self.hop_ms <= 0:
            raise ValueError("hop_ms must be positive")
        return int(self.hop_ms * self.sampling_rate_hz / 1000)


def window_start_indices(n_samples: int, config: WindowConfig) -> list[int]:
    """Start offsets of every complete window. A partial tail is dropped."""
    width = config.window_samples
    hop = config.hop_samples
    if n_samples < width:
        return []
    return list(range(0, n_samples - width + 1, hop))


def frame_windows(channels: np.ndarray, config: WindowConfig) -> np.ndarray:
    """Frame (n_channels, n_samples) into (n_windows, n_channels, window_samples).

    A trailing partial window is dropped rather than zero-padded: a zero-filled
    tail yields features no real contraction could produce and would pollute
    both training data and the drift statistics.
    """
    signal = np.asarray(channels, dtype=np.float32)
    if signal.ndim != 2:
        raise ValueError("EMG input must have shape (n_channels, n_samples)")

    n_channels, n_samples = signal.shape
    width = config.window_samples
    starts = window_start_indices(n_samples, config)
    if not starts:
        return np.zeros((0, n_channels, width), dtype=np.float32)

    return np.stack([signal[:, start : start + width] for start in starts])
