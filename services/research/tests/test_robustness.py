import numpy as np
import pytest

from neurogrip.datasets import CorpusConfig, build_corpus
from neurogrip.robustness import robustness_curve, evaluate_shift
from neurogrip.train import model_by_name


@pytest.fixture(scope="module")
def tiny_config():
    return CorpusConfig(n_subjects=3, reps_per_gesture=2, rep_seconds=0.3)


def test_evaluate_shift_returns_one_fold_per_subject(tiny_config):
    corpus = build_corpus(tiny_config)
    result = evaluate_shift(corpus, corpus, model_by_name("lda"), calibration_folds=2)

    assert len(result.loso.folds) == corpus.n_subjects
    assert result.mean_accuracy > 0.0


def test_evaluate_shift_rejects_misaligned_corpora(tiny_config):
    base = build_corpus(tiny_config)
    shifted = build_corpus(tiny_config)
    shifted = type(shifted)(
        features=shifted.features,
        labels=shifted.labels[::-1],
        subjects=shifted.subjects,
        reps=shifted.reps,
        fatigue=shifted.fatigue,
        feature_names=shifted.feature_names,
        gestures=shifted.gestures,
    )

    with pytest.raises(ValueError):
        evaluate_shift(base, shifted, model_by_name("lda"), calibration_folds=2)


def test_robustness_curve_reports_degradation(tiny_config):
    payload = robustness_curve(
        tiny_config,
        model_by_name("lda"),
        shifts_rad=(0.0, 0.2),
        calibration_folds=2,
    )

    assert payload["model"] == "lda"
    assert payload["baseline_accuracy"] > 0.0
    assert len(payload["shifts"]) == 2
    assert payload["shifts"][0]["shift_rad"] == 0.0
    assert "degradation_pp" in payload["shifts"][1]
    assert isinstance(payload["acceptance"]["meets_bound"], bool)


def test_robustness_curve_requires_at_least_one_shift(tiny_config):
    with pytest.raises(ValueError):
        robustness_curve(tiny_config, model_by_name("lda"), shifts_rad=())
