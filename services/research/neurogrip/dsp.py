"""Numeric primitives for the feature pipeline.

Every function here is mirrored bit-for-bit in packages/core/src/dsp.ts.
scipy.signal.welch is deliberately avoided: its detrending, segment averaging
and scaling conventions are a porting hazard and buy nothing at a single
200 ms window. The algorithms below are the conformance contract.
"""

from __future__ import annotations

import numpy as np


def next_pow2(n: int) -> int:
    if n < 1:
        raise ValueError("n must be positive")
    power = 1
    while power < n:
        power *= 2
    return power


def hann(n: int) -> np.ndarray:
    """Symmetric Hann window. hann(1) is [1.0]."""
    if n < 1:
        raise ValueError("n must be positive")
    if n == 1:
        return np.ones(1, dtype=np.float64)
    i = np.arange(n, dtype=np.float64)
    return 0.5 - 0.5 * np.cos(2.0 * np.pi * i / (n - 1))


def periodogram(x: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]:
    """One-sided Hann-windowed periodogram, zero-padded to a power of two."""
    signal = np.asarray(x, dtype=np.float64)
    n = signal.size
    if n < 1:
        raise ValueError("signal must not be empty")

    window = hann(n)
    nfft = next_pow2(n)
    spectrum = np.fft.rfft(signal * window, n=nfft)
    scale = fs * float(np.sum(window**2))

    psd = np.abs(spectrum) ** 2 / scale if scale > 0 else np.zeros(spectrum.size)
    # Fold negative-frequency power into the positive bins, excluding DC and Nyquist.
    if psd.size > 2:
        psd[1:-1] *= 2.0

    freqs = np.arange(nfft // 2 + 1, dtype=np.float64) * fs / nfft
    return freqs, psd


def band_means(values: np.ndarray, n_bands: int) -> np.ndarray:
    """Mean of each of n_bands contiguous groups, numpy array_split semantics."""
    if n_bands < 1:
        raise ValueError("n_bands must be positive")
    data = np.asarray(values, dtype=np.float64)
    out = np.zeros(n_bands, dtype=np.float64)
    for index, chunk in enumerate(np.array_split(data, n_bands)):
        out[index] = float(np.mean(chunk)) if chunk.size else 0.0
    return out


def autocorrelate(x: np.ndarray, max_lag: int) -> np.ndarray:
    """Biased autocorrelation r[k] = sum(x[n] * x[n+k]) / N, k = 0..max_lag."""
    if max_lag < 0:
        raise ValueError("max_lag must not be negative")
    signal = np.asarray(x, dtype=np.float64)
    n = signal.size
    out = np.zeros(max_lag + 1, dtype=np.float64)
    for lag in range(min(max_lag, n - 1) + 1):
        out[lag] = float(np.dot(signal[: n - lag], signal[lag:])) / n
    return out


def levinson_durbin(r: np.ndarray, order: int) -> np.ndarray:
    """AR coefficients a[1..order] for x[n] = -sum(a[i] * x[n-i]) + e[n].

    Returns zeros when the signal carries no power or the recursion becomes
    numerically degenerate, which is the correct answer for a silent channel.
    """
    if order < 1:
        raise ValueError("order must be positive")
    acf = np.asarray(r, dtype=np.float64)
    if acf.size < order + 1 or acf[0] <= 0.0:
        return np.zeros(order, dtype=np.float64)

    a = np.zeros(order + 1, dtype=np.float64)
    a[0] = 1.0
    error = float(acf[0])

    for m in range(1, order + 1):
        acc = acf[m] + float(np.dot(a[1:m], acf[m - 1 : 0 : -1]))
        reflection = -acc / error
        previous = a[1:m].copy()
        a[m] = reflection
        if m > 1:
            a[1:m] = previous + reflection * previous[::-1]
        error *= 1.0 - reflection * reflection
        if error <= 0.0:
            return np.zeros(order, dtype=np.float64)

    return a[1:]


def median_frequency(freqs: np.ndarray, psd: np.ndarray) -> float:
    """Frequency at which cumulative power first reaches half the total."""
    power = np.asarray(psd, dtype=np.float64)
    total = float(np.sum(power))
    if total <= 0.0:
        return 0.0
    cumulative = np.cumsum(power)
    index = int(np.searchsorted(cumulative, 0.5 * total, side="left"))
    index = min(index, power.size - 1)
    return float(np.asarray(freqs, dtype=np.float64)[index])


def mean_frequency(freqs: np.ndarray, psd: np.ndarray) -> float:
    """Power-weighted mean frequency."""
    power = np.asarray(psd, dtype=np.float64)
    total = float(np.sum(power))
    if total <= 0.0:
        return 0.0
    return float(np.dot(np.asarray(freqs, dtype=np.float64), power) / total)
