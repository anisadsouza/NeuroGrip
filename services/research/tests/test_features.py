import numpy as np
import pytest

from neurogrip.features import FeatureConfig, extract_features, feature_vector
from neurogrip.spec import feature_count, feature_names


def _config(**kwargs) -> FeatureConfig:
    return FeatureConfig(**kwargs)


def test_extracts_every_spec_feature_for_every_channel():
    channels = np.random.default_rng(0).standard_normal((3, 400)).astype(np.float32) * 0.01
    features = extract_features(channels, _config())
    assert set(features) == set(feature_names(3))
    assert len(features) == feature_count(3)


def test_time_domain_values_are_correct():
    channels = np.array([[3.0, -4.0, 3.0, -4.0]], dtype=np.float32)
    features = extract_features(channels, _config(zero_crossing_threshold=0.0))
    assert features["ch1_rms"] == pytest.approx(np.sqrt((9 + 16 + 9 + 16) / 4))
    assert features["ch1_mav"] == pytest.approx(3.5)
    assert features["ch1_wl"] == pytest.approx(7.0 + 7.0 + 7.0)


def test_a_sample_resting_exactly_at_zero_is_not_a_crossing():
    """np.sign(0) == 0 made the old implementation count this as two crossings."""
    channels = np.array([[1.0, 0.0, 1.0]], dtype=np.float32)
    features = extract_features(channels, _config(zero_crossing_threshold=0.0))
    assert features["ch1_zc"] == 0.0


def test_a_genuine_sign_change_is_counted_once():
    channels = np.array([[1.0, -1.0, 1.0]], dtype=np.float32)
    features = extract_features(channels, _config(zero_crossing_threshold=0.0))
    assert features["ch1_zc"] == 2.0


def test_zero_crossing_threshold_suppresses_tiny_wiggles():
    channels = np.array([[1e-9, -1e-9, 1e-9, -1e-9]], dtype=np.float32)
    features = extract_features(channels, _config(zero_crossing_threshold=1e-5))
    assert features["ch1_zc"] == 0.0


def test_slope_sign_threshold_is_scale_consistent():
    """The threshold is in amplitude units, so scaling the signal by k and the
    threshold by k must leave the SSC count unchanged. The old implementation
    compared the threshold against a squared product and failed this."""
    base = np.array([[0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0]], dtype=np.float32)
    small = extract_features(base * 1e-3, _config(slope_sign_threshold=1e-3 * 0.5))
    large = extract_features(base * 1e3, _config(slope_sign_threshold=1e3 * 0.5))
    assert small["ch1_ssc"] == large["ch1_ssc"]
    assert small["ch1_ssc"] > 0


def test_silent_channel_yields_zeroed_ar_and_spectral_features():
    channels = np.zeros((1, 400), dtype=np.float32)
    features = extract_features(channels, _config())
    for name in ("ch1_ar1", "ch1_ar2", "ch1_ar3", "ch1_ar4", "ch1_mdf", "ch1_mnf"):
        assert features[name] == 0.0


def test_feature_vector_follows_spec_order_and_is_float32():
    channels = np.random.default_rng(1).standard_normal((2, 400)).astype(np.float32) * 0.01
    features = extract_features(channels, _config())
    vector = feature_vector(features, 2)
    assert vector.dtype == np.float32
    assert vector.shape == (feature_count(2),)
    assert vector[0] == pytest.approx(features["ch1_rms"], rel=1e-6)
    assert vector[17] == pytest.approx(features["ch2_rms"], rel=1e-6)


def test_rejects_malformed_input():
    with pytest.raises(ValueError):
        extract_features(np.zeros(400, dtype=np.float32), _config())
    with pytest.raises(ValueError):
        extract_features(np.zeros((1, 2), dtype=np.float32), _config())
