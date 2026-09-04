import numpy as np

from pclm.features import FeatureConfig, extract_features, feature_vector


def test_extract_features_returns_stable_vector():
    channels = np.array(
        [
            [0.0, 0.1, -0.1, 0.2, -0.2],
            [0.2, 0.1, 0.0, -0.1, -0.2],
        ],
        dtype=np.float32,
    )

    features = extract_features(channels, FeatureConfig(psd_bins=3))
    vector = feature_vector(features)

    assert "ch1_rms" in features
    assert "ch2_wl" in features
    assert len(features) == 2 * 8
    assert vector.shape == (len(features),)
    assert np.all(np.isfinite(vector))
