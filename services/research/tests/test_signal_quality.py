import numpy as np

from neurogrip.signal_quality import check_signal_quality


def test_quality_rejects_flat_channel():
    channels = np.array([[0.0, 0.0, 0.0], [0.1, 0.2, 0.1]], dtype=np.float32)

    result = check_signal_quality(channels)

    assert not result.ok
    assert result.failed_channel == 0


def test_quality_accepts_valid_signal():
    channels = np.array([[0.1, 0.2, 0.1], [0.2, 0.1, -0.1]], dtype=np.float32)

    result = check_signal_quality(channels)

    assert result.ok
