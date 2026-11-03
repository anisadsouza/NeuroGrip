import numpy as np
import pytest

from neurogrip.attribution import (
    ErrorObservation,
    attribution_demo,
    attribute_error,
    bhattacharyya_overlap,
)


def test_identical_repetition_clusters_have_full_overlap():
    rng = np.random.default_rng(4)
    samples = rng.normal(size=(20, 136))

    assert bhattacharyya_overlap(samples, samples) == pytest.approx(1.0)


def test_separated_repetition_clusters_have_less_overlap():
    rng = np.random.default_rng(5)
    first = rng.normal(size=(20, 136))
    second = first + np.r_[np.full(17, 5.0), np.zeros(119)]

    assert bhattacharyya_overlap(first, second) < 0.1


@pytest.mark.parametrize(
    ("case_name", "expected_cause"),
    [
        ("physiological_overlap", "physiological"),
        ("behavioral_deviation", "behavioral"),
        ("model_capacity", "model_capacity"),
        ("signal_drift", "drift"),
    ],
)
def test_injected_causes_route_to_their_expected_remedy(case_name, expected_cause):
    report = attribution_demo(seed=17)
    result = next(case for case in report["cases"] if case["name"] == case_name)

    assert result["cause"] == expected_cause
    assert result["routed_action"]
    assert sum(result["scores"].values()) == pytest.approx(1.0)


def test_demo_only_claims_routing_of_controlled_injections():
    report = attribution_demo()

    assert report["source"] == "synthetic_fault_injections"
    assert report["injected_causes_routed"] == report["n_injections"] == 4
    assert "not attribution accuracy" in report["caveat"]


def test_behavioral_coaching_localizes_an_electrode_region_without_naming_a_muscle():
    result = next(
        case
        for case in attribution_demo()["cases"]
        if case["name"] == "behavioral_deviation"
    )

    assert result["coaching"]
    assert result["channel_attribution"][0]["channel"] == 4
    assert "does not identify a specific muscle or finger" in result["coaching"]


def test_no_supported_cause_is_reported_as_inconclusive():
    rng = np.random.default_rng(9)
    target = rng.normal(0.0, 0.3, size=(24, 136))
    decoded = rng.normal(0.0, 0.3, size=(24, 136))
    decoded[:, :17] += 5.0
    sample = target.mean(axis=0)
    result = attribute_error(
        ErrorObservation(
            name="ambiguous",
            intended="fist",
            decoded="open_hand",
            observed=sample,
            intended_repetitions=target,
            decoded_repetitions=decoded,
            reference_prediction="open_hand",
        )
    )

    assert result["cause"] == "inconclusive"
    assert result["cause_label"] == "Inconclusive"
    assert result["coaching"] is None


def test_invalid_or_non_error_observations_are_rejected():
    repeats = np.zeros((3, 136))
    with pytest.raises(ValueError, match="misclassification"):
        attribute_error(
            ErrorObservation("correct", "fist", "fist", repeats[0], repeats, repeats, "fist")
        )
    with pytest.raises(ValueError, match="complete channel-major"):
        attribute_error(
            ErrorObservation("bad", "fist", "open_hand", np.zeros(16), np.zeros((3, 16)), np.zeros((3, 16)), None)
        )
