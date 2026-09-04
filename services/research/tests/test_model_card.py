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
