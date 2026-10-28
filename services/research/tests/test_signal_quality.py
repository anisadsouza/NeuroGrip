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
