import numpy as np
import pytest

from neurogrip.datasets import Corpus
from neurogrip.ninapro_feature_attribution import (
    _contributions,
    run_feature_attribution,
)


def test_linear_contributions_exactly_reconstruct_predicted_vs_truth_margin():
    features = np.array([[0.5, -1.0, 2.0], [-0.5, 0.25, 1.5]])
    truth = np.array([0, 1])
    predicted = np.array([2, 0])
    classes = np.array([0, 1, 2])
    coefficients = np.array([[1.0, 2.0, -1.0], [-2.0, 0.5, 1.0], [0.25, -1.0, 2.0]])
    intercepts = np.array([0.3, -0.5, 1.2])

    contributions, residual = _contributions(
        features, truth, predicted, classes, coefficients, intercepts
    )

    for index in range(features.shape[0]):
        actual = predicted[index]
        expected = truth[index]
        margin = (
            features[index] @ coefficients[actual]
            + intercepts[actual]
            - features[index] @ coefficients[expected]
            - intercepts[expected]
        )
        reconstructed = (
            contributions[index].sum()
            + intercepts[actual]
            - intercepts[expected]
        )
        assert reconstructed == pytest.approx(margin)
    assert residual == pytest.approx(0.0)


def test_subject_heldout_report_compares_trained_models_and_attributes_errors():
    rng = np.random.default_rng(42)
    features = []
    labels = []
    subjects = []
    for subject in (1, 10, 11):
        for label, center in enumerate((0.0, 1.0, 2.0)):
            features.append(rng.normal(center + subject / 20, 0.35, (12, 4)))
            labels.extend([label] * 12)
            subjects.extend([subject] * 12)
    corpus = Corpus(
        features=np.vstack(features).astype(np.float32),
        labels=np.asarray(labels, dtype=np.int16),
        subjects=np.asarray(subjects, dtype=np.int16),
        reps=np.tile(np.repeat(np.arange(4), 3), 9).astype(np.int16),
        fatigue=np.zeros(108, dtype=np.float32),
        feature_names=("ch1_rms", "ch1_mav", "ch2_rms", "ch2_mav"),
        gestures=("rest", "fist", "open_hand"),
    )

    report = run_feature_attribution(corpus, svm_samples_per_class=8)

    assert report["subjects"] == [1, 10, 11]
    assert len(report["folds"]) == 3
    assert 0 <= report["primary_accuracy"] <= 1
    assert set(report["reference_accuracy"]) == {
        "linear_svm",
        "rbf_svm",
        "random_forest",
    }
    assert all(0 <= score <= 1 for score in report["reference_accuracy"].values())
    assert report["attribution"]["n_errors_attributed"] == report["primary_error_count"]
    assert report["attribution"]["max_score_decomposition_residual"] < 1e-9
    assert all(
        item["feature"] in corpus.feature_names
        for item in report["attribution"]["top_features"]
    )
