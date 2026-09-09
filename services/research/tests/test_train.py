import numpy as np
import pytest

from neurogrip.datasets import CorpusConfig, build_corpus
from neurogrip.evaluation import LosoResult
from neurogrip.train import (
    MODEL_ZOO,
    evaluate_model,
    fit_final_model,
    model_by_name,
    select_best,
)


@pytest.fixture(scope="module")
def tiny_corpus():
    return build_corpus(
        CorpusConfig(n_subjects=3, reps_per_gesture=2, rep_seconds=0.3, n_electrodes=8)
    )


def test_zoo_covers_the_families_worth_comparing():
    names = {spec.name for spec in MODEL_ZOO}
    assert {"lda", "linear_svm", "rbf_svm"} <= names


def test_every_spec_builds_a_fresh_unfitted_estimator():
    for spec in MODEL_ZOO:
        first, second = spec.build(), spec.build()
        assert first is not second
        assert not hasattr(first, "classes_")


def test_model_by_name_rejects_unknown():
    assert model_by_name("lda").name == "lda"
    with pytest.raises(KeyError):
        model_by_name("transformer")


def test_evaluate_model_returns_one_fold_per_subject(tiny_corpus):
    result = evaluate_model(tiny_corpus, model_by_name("lda"), calibration_folds=2)
    assert isinstance(result.loso, LosoResult)
    assert len(result.loso.folds) == tiny_corpus.n_subjects
    assert {fold.subject for fold in result.loso.folds} == {0, 1, 2}


def test_evaluate_model_produces_well_formed_probabilities(tiny_corpus):
    result = evaluate_model(tiny_corpus, model_by_name("lda"), calibration_folds=2)
    probs = result.probabilities
    assert probs.shape == (tiny_corpus.n_windows, len(tiny_corpus.gestures))
    assert np.all(probs >= 0.0) and np.all(probs <= 1.0)
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, rtol=1e-5)


def test_out_of_fold_predictions_cover_every_window_exactly_once(tiny_corpus):
    result = evaluate_model(tiny_corpus, model_by_name("lda"), calibration_folds=2)
    assert result.predictions.shape == (tiny_corpus.n_windows,)
    assert not np.any(result.predictions < 0)


def test_evaluation_beats_the_majority_baseline(tiny_corpus):
    result = evaluate_model(tiny_corpus, model_by_name("lda"), calibration_folds=2)
    assert result.loso.mean_accuracy > result.baseline_accuracy


def test_select_best_picks_the_highest_mean_accuracy():
    class Stub:
        def __init__(self, name, mean):
            self.name = name
            self._mean = mean

        @property
        def loso(self):
            class L:
                mean_accuracy = self._mean

            return L

    results = {"a": Stub("a", 0.7), "b": Stub("b", 0.9), "c": Stub("c", 0.8)}
    assert select_best(results) == "b"


def test_fit_final_model_predicts_on_new_windows(tiny_corpus):
    model = fit_final_model(tiny_corpus, model_by_name("lda"), calibration_folds=2)
    probs = model.predict_proba(tiny_corpus.features[:5])
    assert probs.shape == (5, len(tiny_corpus.gestures))
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, rtol=1e-5)
