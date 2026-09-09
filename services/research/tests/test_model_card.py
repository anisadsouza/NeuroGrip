import pytest

from neurogrip.model_card import render


@pytest.fixture
def comparison() -> dict:
    return {
        "best_model": "rbf_svm",
        "corpus": {
            "n_subjects": 8,
            "reps_per_gesture": 4,
            "rep_seconds": 1.0,
            "n_windows": 13120,
            "n_features": 136,
            "window_ms": 200,
            "hop_ms": 20,
            "max_fatigue": 0.5,
        },
        "models": {
            "lda": {
                "mean_accuracy": 0.9133,
                "std_accuracy": 0.1108,
                "min_accuracy": 0.6396,
                "ci95_low": 0.82,
                "ci95_high": 1.0,
                "baseline_accuracy": 0.1,
                "expected_calibration_error": 0.242,
                "per_gesture_recall": {"rest": 1.0, "fist": 0.85},
            },
            "rbf_svm": {
                "mean_accuracy": 0.9512,
                "std_accuracy": 0.0597,
                "min_accuracy": 0.8348,
                "ci95_low": 0.9013,
                "ci95_high": 1.0,
                "baseline_accuracy": 0.1,
                "expected_calibration_error": 0.0494,
                "per_gesture_recall": {"rest": 1.0, "fist": 0.846},
            },
        },
        "paired_significance": {
            "rbf_svm_vs_lda": {
                "statistic": 4.0,
                "p_value": 0.132812,
                "mean_difference": 0.0379,
            }
        },
    }


@pytest.fixture
def decoder() -> dict:
    return {
        "model": "rbf_svm",
        "feature_spec_version": 1,
        "n_channels": 8,
        "sampling_rate_hz": 2000,
        "gestures": ["rest", "fist"],
        "onnx_parity": {
            "ok": True,
            "max_abs_diff": 1.2e-07,
            "n_samples": 2000,
            "rtol": 0.0001,
            "atol": 1e-05,
        },
        "latency": {
            "n_windows": 1000,
            "budget_ms": 10.0,
            "meets_budget": True,
            "feature_extraction": {
                "mean_ms": 0.74, "p50_ms": 0.73, "p95_ms": 0.79,
                "p99_ms": 0.85, "max_ms": 0.9,
            },
            "onnx_inference": {
                "mean_ms": 0.2, "p50_ms": 0.19, "p95_ms": 0.25,
                "p99_ms": 0.3, "max_ms": 0.4,
            },
            "end_to_end": {
                "mean_ms": 0.94, "p50_ms": 0.92, "p95_ms": 1.04,
                "p99_ms": 1.15, "max_ms": 1.3,
            },
        },
    }


def test_card_leads_with_the_not_a_medical_device_disclaimer(comparison, decoder):
    card = render(comparison, decoder, "2026-09-04")
    header = card.split("## ")[0]
    assert "Not a medical device" in header


def test_card_marks_the_selected_model(comparison, decoder):
    card = render(comparison, decoder, "2026-09-04")
    assert "`rbf_svm` **(selected)**" in card
    assert "`lda` |" in card  # the loser is still reported, not hidden


def test_card_reports_the_simulator_caveat(comparison, decoder):
    card = render(comparison, decoder, "2026-09-04")
    assert "simulated" in card.lower()
    assert "NinaPro" in card


def test_card_states_the_absent_amputee_stratum(comparison, decoder):
    """The report treats amputee-stratum reporting as a first-class requirement.
    The card must say plainly that it is not satisfied yet."""
    card = render(comparison, decoder, "2026-09-04")
    assert "amputee" in card.lower()


def test_card_reports_latency_verdict(comparison, decoder):
    card = render(comparison, decoder, "2026-09-04")
    assert "**PASS**" in card
    assert "1.04 ms" in card


def test_card_reports_a_latency_failure_when_the_budget_is_missed(comparison, decoder):
    decoder["latency"]["meets_budget"] = False
    card = render(comparison, decoder, "2026-09-04")
    assert "**FAIL**" in card


def test_card_includes_the_significance_table(comparison, decoder):
    card = render(comparison, decoder, "2026-09-04")
    assert "0.132812" in card
    assert "Wilcoxon" in card


def test_card_reports_parity(comparison, decoder):
    card = render(comparison, decoder, "2026-09-04")
    assert "1.20e-07" in card
    assert "Within tolerance." in card


@pytest.fixture
def ttum() -> dict:
    return {
        "nTrials": 320,
        "hopsPerTrial": 41,
        "hopMs": 20,
        "definition": "Time to Useful Motion: hops from a trial first decodable window ...",
        "reachability": "Each trial is 41 hops, and the urgency timeout fires at 75 ...",
        "caveat": "Every repetition holds a constant excitation for its whole duration ...",
        "anyMotion": {
            "nMoved": 320,
            "nCensored": 0,
            "p50Ms": 40,
            "p90Ms": 100,
            "p95Ms": 100,
            "maxMs": 240,
        },
        "correctMotion": {
            "nMoved": 318,
            "nCensored": 2,
            "p50Ms": 40,
            "p90Ms": 100,
            "p95Ms": 140,
            "maxMs": 720,
        },
        "latch": {
            "nLatched": 309,
            "nCensored": 11,
            "nTimedOut": 0,
            "accuracyAtLatch": 0.9871,
            "p50Ms": 260,
            "p95Ms": 420,
        },
        "perGesture": {
            "fist": {"nTrials": 32, "nMoved": 32, "p50Ms": 80, "risk": 1.0},
            "open_hand": {"nTrials": 32, "nMoved": 32, "p50Ms": 40, "risk": 0.1},
        },
    }


def test_ttum_section_appears_when_the_artifact_is_present(comparison, decoder, ttum):
    card = render(comparison, decoder, "2026-09-09", ttum)
    assert "### Time to useful motion (TTUM)" in card
    assert "320 out-of-fold trials" in card


def test_the_card_renders_without_a_ttum_run(comparison, decoder):
    """The TTUM run needs the Node toolchain; the card must not.

    A researcher regenerating the card from a Python checkout gets a card
    without the section, rather than a traceback.
    """
    card = render(comparison, decoder, "2026-09-09")
    assert "Time to useful motion" not in card
    assert "### Latency" in card


def test_the_ttum_section_orders_gestures_by_commit_cost(comparison, decoder, ttum):
    """The risk weighting is the mechanism's claim, and a reader should be able
    to see it as a trend down the table rather than take it on trust."""
    card = render(comparison, decoder, "2026-09-09", ttum)
    assert card.index("| `fist` | 1.0") < card.index("| `open_hand` | 0.1")


def test_the_ttum_section_reports_censored_trials_and_the_caveat(comparison, decoder, ttum):
    """Both are load-bearing. Censored trials are the slow ones by definition,
    so a summary that omitted them would read better than the system behaves;
    and the constant-excitation caveat is what stops the figure being quoted as
    a wearer's reaction time."""
    card = render(comparison, decoder, "2026-09-09", ttum)
    assert "Censored" in card
    assert "constant excitation" in card


def test_ttum_never_prints_a_missing_measurement_as_zero(comparison, decoder, ttum):
    """A censored measurement and an instantaneous one are opposite findings."""
    ttum["anyMotion"] = {**ttum["anyMotion"], "nMoved": 0, "nCensored": 320, "p50Ms": None,
                         "p90Ms": None, "p95Ms": None, "maxMs": None}
    card = render(comparison, decoder, "2026-09-09", ttum)
    assert "| Any motion | 0 | 320 | — | — | — | — |" in card
