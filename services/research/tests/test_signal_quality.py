import numpy as np

from neurogrip.signal_quality import SignalQualityConfig, check_signal_quality


def test_quality_rejects_flat_channel():
    channels = np.array([[0.0, 0.0, 0.0], [0.1, 0.2, 0.1]], dtype=np.float32)

    result = check_signal_quality(channels)

    assert not result.ok
    assert result.failed_channel == 0


def test_quality_accepts_valid_signal():
    channels = np.array([[0.1, 0.2, 0.1], [0.2, 0.1, -0.1]], dtype=np.float32)

    result = check_signal_quality(channels)

    assert result.ok


def test_quality_applies_recorded_per_channel_floors():
    channels = np.array(
        [[1e-6, -1e-6, 1e-6], [0.2e-6, -0.2e-6, 0.2e-6]],
        dtype=np.float32,
    )
    config = SignalQualityConfig(
        min_channel_rms_by_channel=(0.5e-6, 0.3e-6)
    )

    result = check_signal_quality(channels, config)

    assert not result.ok
    assert result.reason == "channel RMS below calibrated minimum"
    assert result.failed_channel == 1


def test_quality_rejects_recorded_calibration_with_wrong_channel_count():
    channels = np.array([[0.1, 0.2], [0.2, 0.1]], dtype=np.float32)
    config = SignalQualityConfig(min_channel_rms_by_channel=(1e-6,))

    result = check_signal_quality(channels, config)

    assert not result.ok
    assert result.reason == "quality calibration channel count mismatch"


def test_quality_rejects_repeated_extreme_samples_suggesting_clipping():
    time = np.arange(400, dtype=np.float32)
    signal = np.stack(
        [
            100e-6 * np.sin(2 * np.pi * 35 * time / 2000 + channel)
            for channel in range(8)
        ]
    ).astype(np.float32)

    clipped = np.clip(signal, -50e-6, 50e-6)

    result = check_signal_quality(clipped)

    assert not result.ok
    assert result.reason == "repeated extreme samples suggest clipping"


def test_quality_rejects_repeated_large_steps_suggesting_transient_noise():
    time = np.arange(400, dtype=np.float32)
    signal = np.stack(
        [
            100e-6 * np.sin(2 * np.pi * 35 * time / 2000 + channel)
            for channel in range(8)
        ]
    ).astype(np.float32)
    burst = signal.copy()
    burst[0, 100:120] += np.where(np.arange(20) % 2, 1e-3, -1e-3)

    result = check_signal_quality(burst)

    assert not result.ok
    assert result.reason == "repeated large sample steps suggest transient noise"


def test_quality_rejects_abrupt_half_window_gain_shift():
    time = np.arange(400, dtype=np.float32)
    signal = np.stack(
        [
            100e-6 * np.sin(2 * np.pi * 35 * time / 2000 + channel)
            for channel in range(8)
        ]
    ).astype(np.float32)
    jumped = signal.copy()
    jumped[:, 200:] *= 10

    result = check_signal_quality(jumped)

    assert not result.ok
    assert result.reason == "abrupt half-window RMS change suggests gain shift"


def test_quality_accepts_recorded_sized_signal_without_fault_patterns():
    time = np.arange(400, dtype=np.float32)
    signal = np.stack(
        [
            100e-6 * np.sin(2 * np.pi * 35 * time / 2000 + channel)
            for channel in range(8)
        ]
    ).astype(np.float32)

    assert check_signal_quality(signal).ok
