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


def test_quality_gate_does_not_detect_clipping_noise_burst_or_gain_jump():
    time = np.arange(400, dtype=np.float32)
    signal = np.stack(
        [
            100e-6 * np.sin(2 * np.pi * 35 * time / 2000 + channel)
            for channel in range(8)
        ]
    ).astype(np.float32)

    clipped = np.clip(signal, -50e-6, 50e-6)
    burst = signal.copy()
    burst[0, 100:120] += np.where(np.arange(20) % 2, 1e-3, -1e-3)
    jumped = signal.copy()
    jumped[:, 200:] *= 10

    assert check_signal_quality(clipped).ok
    assert check_signal_quality(burst).ok
    assert check_signal_quality(jumped).ok
