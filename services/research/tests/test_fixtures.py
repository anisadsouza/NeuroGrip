import json

import numpy as np

from neurogrip import FEATURE_SPEC_VERSION
from neurogrip.features import FeatureConfig, extract_features, feature_vector
from neurogrip.fixtures import build_fixture_payload, write_fixtures
from neurogrip.spec import PER_CHANNEL_FEATURES


def test_payload_declares_the_spec_version_and_feature_table():
    payload = build_fixture_payload()
    assert payload["featureSpecVersion"] == FEATURE_SPEC_VERSION
    assert tuple(payload["perChannelFeatures"]) == PER_CHANNEL_FEATURES


def test_payload_covers_the_awkward_cases():
    names = {case["name"] for case in build_fixture_payload()["cases"]}
    for required in (
        "silent",
        "dc_offset",
        "pure_tone",
        "touches_zero",
        "twelve_channels",
    ):
        assert required in names


def test_every_case_vector_matches_a_fresh_extraction():
    """The fixture must be self-consistent or it pins the wrong answer."""
    payload = build_fixture_payload()
    for case in payload["cases"]:
        channels = np.asarray(case["channels"], dtype=np.float32)
        config = FeatureConfig(
            sampling_rate_hz=case["samplingRateHz"],
            zero_crossing_threshold=payload["config"]["zeroCrossingThreshold"],
            slope_sign_threshold=payload["config"]["slopeSignThreshold"],
        )
        expected = feature_vector(extract_features(channels, config), channels.shape[0])
        assert np.allclose(expected, case["featureVector"], rtol=1e-6, atol=1e-12)
        assert len(case["featureVector"]) == len(case["featureNames"])


def test_generation_is_deterministic():
    assert build_fixture_payload() == build_fixture_payload()


def test_write_fixtures_round_trips(tmp_path):
    target = tmp_path / "golden.json"
    written = write_fixtures(target)
    assert json.loads(target.read_text(encoding="utf-8")) == written
