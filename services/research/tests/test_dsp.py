import numpy as np
import pytest

from neurogrip.dsp import (
    autocorrelate,
    band_means,
    hann,
    levinson_durbin,
    mean_frequency,
    median_frequency,
    next_pow2,
    periodogram,
)


def test_next_pow2():
    assert next_pow2(1) == 1
    assert next_pow2(2) == 2
    assert next_pow2(3) == 4
    assert next_pow2(400) == 512


def test_hann_is_symmetric_and_zero_at_the_ends():
    w = hann(9)
    assert w[0] == pytest.approx(0.0, abs=1e-12)
    assert w[-1] == pytest.approx(0.0, abs=1e-12)
    assert w[4] == pytest.approx(1.0)
    assert np.allclose(w, w[::-1])
    assert np.array_equal(hann(1), np.array([1.0]))


def test_periodogram_locates_a_pure_tone():
    fs = 1000.0
    t = np.arange(512) / fs
    x = np.sin(2 * np.pi * 100.0 * t)
    freqs, psd = periodogram(x, fs)
    assert freqs[int(np.argmax(psd))] == pytest.approx(100.0, abs=fs / 512)
    assert len(freqs) == len(psd) == 512 // 2 + 1


def test_band_means_uses_array_split_semantics():
    # 7 items into 3 bands -> sizes 3, 2, 2
    values = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
    assert np.allclose(band_means(values, 3), [2.0, 4.5, 6.5])


def test_band_means_pads_with_zero_when_shorter_than_bands():
    assert np.allclose(band_means(np.array([5.0]), 3), [5.0, 0.0, 0.0])


def test_autocorrelate_is_biased_and_peaks_at_zero_lag():
    x = np.array([1.0, 2.0, 3.0, 4.0])
    r = autocorrelate(x, 2)
    assert len(r) == 3
    assert r[0] == pytest.approx(np.sum(x**2) / 4)
    assert r[0] >= r[1] >= r[2]


def test_levinson_recovers_a_known_ar2_process():
    rng = np.random.default_rng(7)
    n = 4000
    a1, a2 = -1.2, 0.5  # x[n] = 1.2 x[n-1] - 0.5 x[n-2] + e
    x = np.zeros(n)
    e = rng.standard_normal(n) * 0.1
    for i in range(2, n):
        x[i] = -a1 * x[i - 1] - a2 * x[i - 2] + e[i]
    coeffs = levinson_durbin(autocorrelate(x, 2), 2)
    assert coeffs[0] == pytest.approx(a1, abs=0.05)
    assert coeffs[1] == pytest.approx(a2, abs=0.05)


def test_levinson_returns_zeros_for_a_silent_signal():
    assert np.allclose(levinson_durbin(np.zeros(5), 4), np.zeros(4))


def test_frequency_moments_on_a_flat_spectrum():
    freqs = np.array([0.0, 10.0, 20.0, 30.0, 40.0])
    psd = np.ones(5)
    assert mean_frequency(freqs, psd) == pytest.approx(20.0)
    assert median_frequency(freqs, psd) == pytest.approx(20.0)


def test_frequency_moments_are_zero_for_no_power():
    freqs = np.array([0.0, 10.0, 20.0])
    psd = np.zeros(3)
    assert mean_frequency(freqs, psd) == 0.0
    assert median_frequency(freqs, psd) == 0.0
