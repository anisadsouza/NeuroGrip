"""End-to-end routing on simulated sessions.

These tests pin the accounting and the evidence statistics, not the routing
outcome: whether each session reaches its injected cause is the experiment's
result, recorded in `artifacts/router_session.json`, and a test that demanded
4/4 would be an invitation to tune thresholds until it passed.
"""

import numpy as np
import pytest

from neurogrip.router_session import cusum, run_session


def test_cusum_stays_at_zero_for_in_distribution_scores():
    rng = np.random.default_rng(1)
    statistic = cusum(rng.normal(0.0, 0.3, size=200))

    assert statistic.min() >= 0.0
    assert statistic.max() < 2.0


def test_cusum_grows_under_a_sustained_shift():
    statistic = cusum(np.full(20, 1.5))

    assert np.all(np.diff(statistic) > 0)
    assert statistic[-1] == pytest.approx(20 * (1.5 - 0.5))


def test_unknown_fault_is_refused():
    with pytest.raises(ValueError):
        run_session("sunspots")


@pytest.mark.parametrize("fault", ["behavioral", "drift", "physiological", "model_capacity"])
def test_every_error_window_is_routed_somewhere(fault):
    session = run_session(fault, seed=0, reps=3)

    assert session["n_error_windows"] == sum(session["routed"].values())
    assert 0 <= session["deployed_accuracy"] <= 1
    assert session["injected_cause"] == fault


def test_behavioural_routes_carry_anatomical_coaching():
    session = run_session("behavioral", seed=0, reps=3)
    behavioural = [item for item in session["examples"] if item["cause"] == "behavioral"]

    for item in behavioural:
        assert set(item["anatomy"]["muscle_groups"]) == {
            "digit_flexor",
            "digit_extensor",
            "wrist_flexor",
            "wrist_extensor",
        }
        assert item["coaching"].startswith("Your ")


def test_drift_session_shows_the_spectral_shift_it_was_given():
    drifted = run_session("drift", seed=0, reps=3)
    clean = run_session("behavioral", seed=0, reps=3)

    # Fatigue widens MUAPs, so median frequency must fall, and by more than in
    # a session that was not fatigued.
    assert drifted["final_mdf_shift_hz"] < 0
    assert abs(drifted["final_mdf_shift_hz"]) > abs(clean["final_mdf_shift_hz"])


def test_a_clean_session_accumulates_far_less_drift_evidence_than_a_fatigued_one():
    # The first version compared fresh windows with in-sample training
    # distances, which are smaller by construction, and a clean session
    # reached 5x its alarm. Leave-one-repetition-out removed most of that. It
    # does not reliably stay below 1: windows within one attempt share its
    # execution jitter, so z-scores arrive in correlated blocks and CUSUM,
    # which assumes independence, still inflates. Documented, not hidden.
    clean = run_session("model_capacity", seed=0, reps=4)
    fatigued = run_session("drift", seed=0, reps=4)

    assert clean["final_cusum_ratio"] < 3.0
    assert fatigued["final_cusum_ratio"] > 4 * clean["final_cusum_ratio"]


def test_drift_evidence_for_an_error_ignores_the_gesture_in_error():
    # Drift moves every gesture; a behavioural fault moves one. Evidence for
    # drift is therefore read from the other cued gestures only.
    from neurogrip.router_session import other_gesture_cusum

    labels = np.array([1] * 5 + [2] * 5)
    z = np.r_[np.zeros(5), np.full(5, 3.0)]

    assert other_gesture_cusum(z, labels, index=9, intended=2) == 0.0
    assert other_gesture_cusum(z, labels, index=9, intended=1) > 0.0
