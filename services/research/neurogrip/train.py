"""Model comparison under leave-one-subject-out, with in-fold calibration.

The project report names an RBF SVM as the handcrafted-feature decoder. This
module does not assume that: it evaluates a small zoo and selects on evidence,
because an early characterisation run found plain LDA beating an untuned RBF
SVM on this feature set. Which family wins is a project finding, not a premise.

Calibration matters as much as accuracy here. A prosthetic controller that
abstains below a confidence threshold is only safe if its confidences mean
something, so every model is wrapped in a calibrator fitted strictly inside the
training fold. Fitting calibration on data the fold has seen would inflate both
accuracy and the calibration score.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.calibration import CalibratedClassifierCV
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC

from neurogrip.datasets import Corpus
from neurogrip.evaluation import (
    FoldResult,
    LosoResult,
    expected_calibration_error,
    loso_splits,
    majority_class_accuracy,
)
from neurogrip.simulator import GESTURES


@dataclass(frozen=True)
class ModelSpec:
    name: str
    description: str
    build: Callable[[], BaseEstimator]


def _lda() -> BaseEstimator:
    return LinearDiscriminantAnalysis()


def _lda_uniform_priors() -> BaseEstimator:
    n_classes = len(GESTURES)
    return LinearDiscriminantAnalysis(priors=np.full(n_classes, 1 / n_classes))


def _linear_svm() -> BaseEstimator:
    return LinearSVC(C=1.0, dual="auto", max_iter=5000)


def _rbf_svm() -> BaseEstimator:
    return SVC(C=10.0, gamma="scale", kernel="rbf")


MODEL_ZOO: tuple[ModelSpec, ...] = (
    ModelSpec(
        "lda",
        "Linear discriminant analysis. Cheapest at inference; a strong baseline "
        "whenever the feature space is close to linearly separable.",
        _lda,
    ),
    ModelSpec(
        "lda_uniform",
        "Linear discriminant analysis with uniform class priors. Tests whether "
        "the training-set class frequencies bias predictions toward rest.",
        _lda_uniform_priors,
    ),
    ModelSpec(
        "linear_svm",
        "Linear support vector machine. Tests whether a max-margin linear "
        "boundary beats LDA's Gaussian assumption.",
        _linear_svm,
    ),
    ModelSpec(
        "rbf_svm",
        "RBF-kernel support vector machine. The decoder named in the project "
        "report; the most expensive of the three at inference.",
        _rbf_svm,
    ),
)


def model_by_name(name: str) -> ModelSpec:
    for spec in MODEL_ZOO:
        if spec.name == name:
            return spec
    raise KeyError(f"unknown model '{name}'; known: {[s.name for s in MODEL_ZOO]}")


def _calibrated_pipeline(
    spec: ModelSpec, calibration_folds: int, ensemble: bool = False
) -> Pipeline:
    """Scale, then classify, then calibrate the classifier's scores.

    Sigmoid (Platt) calibration rather than isotonic: isotonic needs far more
    data per class to avoid overfitting, and the per-fold, per-class counts here
    are modest.

    `ensemble=False` by default. The ensemble form fits one classifier per
    calibration fold and averages them, which for an RBF SVM multiplies the
    stored support vectors by the fold count. Measured on the full corpus, the
    single form was better on every axis at once -- accuracy 0.9563 vs 0.9512,
    ECE 0.0225 vs 0.0494, model 2.15 MB vs 5.16 MB, inference P95 0.94 ms vs
    2.17 ms -- so there is no trade-off to weigh. Size and inference cost matter
    here because this model is downloaded and run inside a browser.
    """
    return make_pipeline(
        StandardScaler(),
        CalibratedClassifierCV(
            spec.build(), method="sigmoid", cv=calibration_folds, ensemble=ensemble
        ),
    )


@dataclass(frozen=True)
class ModelEvaluation:
    name: str
    loso: LosoResult
    predictions: np.ndarray  # out-of-fold, one per window
    probabilities: np.ndarray  # out-of-fold, (n_windows, n_classes)
    baseline_accuracy: float
    calibration_error: float

    def summary(self) -> dict[str, object]:
        low, high = self.loso.confidence_interval()
        return {
            "model": self.name,
            "mean_accuracy": round(self.loso.mean_accuracy, 4),
            "std_accuracy": round(self.loso.std_accuracy, 4),
            "min_accuracy": round(self.loso.min_accuracy, 4),
            "ci95_low": round(low, 4),
            "ci95_high": round(high, 4),
            "baseline_accuracy": round(self.baseline_accuracy, 4),
            "expected_calibration_error": round(self.calibration_error, 4),
            "per_fold_accuracy": [round(a, 4) for a in self.loso.accuracies.tolist()],
            "per_gesture_recall": {
                gesture: (None if np.isnan(value) else round(value, 4))
                for gesture, value in self.loso.per_gesture_recall().items()
            },
        }


@dataclass(frozen=True)
class RepetitionEvaluation:
    """NinaPro's prescribed repetition-held-out evaluation."""

    accuracy: float
    calibration_error: float
    n_train: int
    n_test: int
    train_repetitions: tuple[int, ...]
    test_repetitions: tuple[int, ...]
    confusion: np.ndarray

    def summary(self, gestures: tuple[str, ...]) -> dict[str, object]:
        support = self.confusion.sum(axis=1)
        recall = {
            name: (None if support[index] == 0 else round(float(self.confusion[index, index] / support[index]), 4))
            for index, name in enumerate(gestures)
        }
        return {
            "accuracy": round(self.accuracy, 4),
            "expected_calibration_error": round(self.calibration_error, 4),
            "n_train": self.n_train,
            "n_test": self.n_test,
            "train_repetitions": list(self.train_repetitions),
            "test_repetitions": list(self.test_repetitions),
            "per_gesture_recall": recall,
            "confusion_matrix": self.confusion.tolist(),
        }


def evaluate_model(
    corpus: Corpus,
    spec: ModelSpec,
    calibration_folds: int = 3,
    calibration_ensemble: bool = False,
) -> ModelEvaluation:
    """Leave-one-subject-out evaluation, collecting out-of-fold probabilities."""
    n_classes = len(corpus.gestures)
    predictions = np.full(corpus.n_windows, -1, dtype=np.int16)
    probabilities = np.zeros((corpus.n_windows, n_classes), dtype=np.float64)
    folds: list[FoldResult] = []

    for train, test in loso_splits(corpus.subjects):
        # A fresh pipeline per fold. Reusing a fitted one would leak the
        # held-out subject into the scaler and the calibrator.
        model = _calibrated_pipeline(spec, calibration_folds, calibration_ensemble)
        model.fit(corpus.features[train], corpus.labels[train])

        fold_probabilities = model.predict_proba(corpus.features[test])
        fold_predictions = fold_probabilities.argmax(axis=1)

        probabilities[test] = fold_probabilities
        predictions[test] = fold_predictions

        held = int(corpus.subjects[test][0])
        folds.append(
            FoldResult(
                subject=held,
                accuracy=float((fold_predictions == corpus.labels[test]).mean()),
                n_test=int(test.size),
                confusion=confusion_matrix(
                    corpus.labels[test], fold_predictions, labels=range(n_classes)
                ),
            )
        )

    return ModelEvaluation(
        name=spec.name,
        loso=LosoResult(folds=tuple(folds), gestures=corpus.gestures),
        predictions=predictions,
        probabilities=probabilities,
        baseline_accuracy=majority_class_accuracy(corpus.labels),
        calibration_error=expected_calibration_error(probabilities, corpus.labels),
    )


def evaluate_repetition_split(
    corpus: Corpus,
    spec: ModelSpec,
    train_repetitions: tuple[int, ...] = (1, 3, 4, 6),
    test_repetitions: tuple[int, ...] = (2, 5),
    calibration_folds: int = 3,
) -> RepetitionEvaluation:
    """Evaluate the DB2 convention without overlapping train/test repetitions.

    DB2 marks rest with repetition zero. It is excluded here because it cannot
    be assigned to a performed repetition without leaking long pauses into both
    sides of the split.
    """
    train = np.flatnonzero(np.isin(corpus.reps, train_repetitions))
    test = np.flatnonzero(np.isin(corpus.reps, test_repetitions))
    if not train.size or not test.size:
        raise ValueError("repetition split has an empty train or test partition")
    model = _calibrated_pipeline(spec, calibration_folds)
    model.fit(corpus.features[train], corpus.labels[train])
    probabilities = model.predict_proba(corpus.features[test])
    predictions = probabilities.argmax(axis=1)
    truth = corpus.labels[test]
    return RepetitionEvaluation(
        accuracy=float((predictions == truth).mean()),
        calibration_error=expected_calibration_error(probabilities, truth),
        n_train=int(train.size),
        n_test=int(test.size),
        train_repetitions=train_repetitions,
        test_repetitions=test_repetitions,
        confusion=confusion_matrix(truth, predictions, labels=range(len(corpus.gestures))),
    )


def select_best(results: dict[str, ModelEvaluation]) -> str:
    """Name of the model with the highest mean LOSO accuracy."""
    if not results:
        raise ValueError("no results to select from")
    return max(results, key=lambda name: results[name].loso.mean_accuracy)


def fit_final_model(
    corpus: Corpus,
    spec: ModelSpec,
    calibration_folds: int = 3,
    calibration_ensemble: bool = False,
) -> Pipeline:
    """Fit on the whole corpus, for export.

    The accuracy this model should be reported with is the leave-one-subject-out
    figure from `evaluate_model`, never a score measured on this fit.
    """
    model = _calibrated_pipeline(spec, calibration_folds, calibration_ensemble)
    model.fit(corpus.features, corpus.labels)
    return model
