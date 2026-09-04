import numpy as np
import pytest

from neurogrip.windowing import WindowConfig, frame_windows, window_start_indices


def test_sample_counts_derive_from_milliseconds_and_rate():
    config = WindowConfig(window_ms=200, hop_ms=20, sampling_rate_hz=2000)
    assert config.window_samples == 400
    assert config.hop_samples == 40


def test_start_indices_are_hop_spaced_and_drop_the_partial_tail():
    config = WindowConfig(window_ms=100, hop_ms=50, sampling_rate_hz=100)
    # window = 10 samples, hop = 5 samples
    assert window_start_indices(22, config) == [0, 5, 10]
    assert window_start_indices(9, config) == []


def test_frame_windows_shape_and_content():
    config = WindowConfig(window_ms=100, hop_ms=50, sampling_rate_hz=100)
    channels = np.tile(np.arange(20, dtype=np.float32), (2, 1))
    windows = frame_windows(channels, config)
    assert windows.shape == (3, 2, 10)
    assert np.allclose(windows[0, 0], np.arange(10))
    assert np.allclose(windows[1, 0], np.arange(5, 15))


def test_frame_windows_returns_empty_when_signal_is_too_short():
    config = WindowConfig(window_ms=200, hop_ms=20, sampling_rate_hz=2000)
    windows = frame_windows(np.zeros((2, 100), dtype=np.float32), config)
    assert windows.shape == (0, 2, 400)


def test_rejects_non_2d_input():
    config = WindowConfig()
    with pytest.raises(ValueError):
        frame_windows(np.zeros(400, dtype=np.float32), config)


def test_rejects_non_positive_geometry():
    with pytest.raises(ValueError):
        WindowConfig(window_ms=0).window_samples
    with pytest.raises(ValueError):
        WindowConfig(hop_ms=0).hop_samples
