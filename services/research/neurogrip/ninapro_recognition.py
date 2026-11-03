"""Reproducible, subject-held-out NinaPro recognition experiments.

Run from the repository root:

    python -m neurogrip.ninapro_recognition \
        --data-root data/ninapro/DB2 \
        --out artifacts/ninapro_recognition_experiment.json

The outer folds hold out an entire subject. Calibration folds are grouped by
subject and repetition so overlapping windows from one repetition cannot occur
on both sides of an internal calibration split. SVM runs use a seeded,
per-subject/per-class cap to keep the nonlinear comparison bounded.
"""

from __future__ import annotations

import argparse
import json
import pathlib
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.calibration import CalibratedClassifierCV
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC

from neurogrip.datasets import Corpus
from neurogrip.evaluation import (
    expected_calibration_error,
    loso_splits,
    majority_class_accuracy,
)
from neurogrip.ninapro import build_ninapro_corpus

DEFAULT_SEED = 20261104


@dataclass(frozen=True)
class Strategy:
    name: str
    build: Callable[[], BaseEstimator]
    calibrated: bool
    balance: Literal["none", "rest_downsample"]
    svm_cap: bool


def _calibration_groups(subjects: np.ndarray, repetitions: np.ndarray) -> np.ndarray:
    """Keep all windows from one subject/repetition in one calibration group."""
    return np.asarray(subjects, dtype=np.int64) * 10 + np.asarray(
        repetitions, dtype=np.int64
    )


def _grouped_calibration_splits(
    groups: np.ndarray, labels: np.ndarray, n_splits: int
) -> list[tuple[np.ndarray, np.ndarray]]:
    if np.unique(groups).size < n_splits:
        raise ValueError(
            f"grouped calibration needs at least {n_splits} groups; "
            f"got {np.unique(groups).size}"
        )
    splits = list(
        GroupKFold(n_splits=n_splits).split(np.zeros(len(groups)), labels, groups)
    )
    expected_classes = set(np.unique(labels))
    for train, validation in splits:
        if set(groups[train]).intersection(groups[validation]):
            raise RuntimeError("calibration split leaked a subject/repetition group")
        missing = expected_classes - set(np.unique(labels[train]))
        if missing:
            raise ValueError(
                f"calibration training partition is missing classes: {sorted(missing)}"
            )
    return splits


def _downsample_rest(
    indices: np.ndarray,
    labels: np.ndarray,
    n_classes: int,
    seed: int,
) -> np.ndarray:
    """Retain all movement examples and cap rest to median movement support."""
    movement_counts = np.asarray(
        [np.count_nonzero(labels[indices] == label) for label in range(1, n_classes)],
        dtype=np.int64,
    )
    if np.any(movement_counts == 0):
        raise ValueError("rest balancing requires examples from every movement class")
    target = int(np.median(movement_counts))
    rest = indices[labels[indices] == 0]
    movement = indices[labels[indices] != 0]
    if rest.size <= target:
        return indices.copy()
    rng = np.random.default_rng(seed)
    chosen_rest = rng.choice(rest, size=target, replace=False)
    return np.sort(np.concatenate((movement, chosen_rest)))


def _cap_per_subject_class(
    indices: np.ndarray,
    subjects: np.ndarray,
    labels: np.ndarray,
    cap: int,
    seed: int,
) -> np.ndarray:
    if cap < 1:
        raise ValueError("per-class sample cap must be positive")
    rng = np.random.default_rng(seed)
    selected: list[np.ndarray] = []
    for subject in np.unique(subjects[indices]):
        subject_indices = indices[subjects[indices] == subject]
        for label in np.unique(labels[subject_indices]):
            class_indices = subject_indices[labels[subject_indices] == label]
            if class_indices.size > cap:
                class_indices = rng.choice(class_indices, size=cap, replace=False)
            selected.append(class_indices)
    return np.sort(np.concatenate(selected))


def _strategies(n_classes: int) -> list[Strategy]:
    return [
        Strategy(
            name="lda_empirical_raw",
            build=LinearDiscriminantAnalysis,
            calibrated=False,
            balance="none",
            svm_cap=False,
        ),
        Strategy(
            name="lda_empirical_sigmoid",
            build=LinearDiscriminantAnalysis,
            calibrated=True,
            balance="none",
            svm_cap=False,
        ),
        Strategy(
            name="lda_uniform_raw",
            build=lambda: LinearDiscriminantAnalysis(
                priors=np.full(n_classes, 1.0 / n_classes)
            ),
            calibrated=False,
            balance="none",
            svm_cap=False,
        ),
        Strategy(
            name="lda_uniform_sigmoid",
            build=lambda: LinearDiscriminantAnalysis(
                priors=np.full(n_classes, 1.0 / n_classes)
            ),
            calibrated=True,
            balance="none",
            svm_cap=False,
        ),
        Strategy(
            name="lda_rest_downsampled_sigmoid",
            build=lambda: LinearDiscriminantAnalysis(
                priors=np.full(n_classes, 1.0 / n_classes)
            ),
            calibrated=True,
            balance="rest_downsample",
            svm_cap=False,
        ),
        Strategy(
            name="linear_svm_sampled_sigmoid",
            build=lambda: LinearSVC(C=1.0, dual="auto", max_iter=5000),
            calibrated=True,
            balance="none",
            svm_cap=True,
        ),
        Strategy(
            name="linear_svm_weighted_sigmoid",
            build=lambda: LinearSVC(
                C=1.0, class_weight="balanced", dual="auto", max_iter=5000
            ),
            calibrated=True,
            balance="none",
            svm_cap=True,
        ),
        Strategy(
            name="rbf_svm_weighted_sigmoid",
            build=lambda: SVC(C=10.0, class_weight="balanced", gamma="scale"),
            calibrated=True,
            balance="none",
            svm_cap=True,
        ),
    ]


def _fit_predict(
    strategy: Strategy,
    corpus: Corpus,
    train: np.ndarray,
    test: np.ndarray,
    *,
    seed: int,
    svm_samples_per_class: int,
    calibration_folds: int,
) -> np.ndarray:
    fit_indices = train
    if strategy.balance == "rest_downsample":
        fit_indices = _downsample_rest(
            fit_indices, corpus.labels, len(corpus.gestures), seed
        )
    if strategy.svm_cap:
        fit_indices = _cap_per_subject_class(
            fit_indices,
            corpus.subjects,
            corpus.labels,
            svm_samples_per_class,
            seed,
        )

    estimator = make_pipeline(StandardScaler(), strategy.build())
    if strategy.calibrated:
        groups = _calibration_groups(
            corpus.subjects[fit_indices], corpus.reps[fit_indices]
        )
        splits = _grouped_calibration_splits(
            groups, corpus.labels[fit_indices], calibration_folds
        )
        model = CalibratedClassifierCV(
            estimator=estimator,
            method="sigmoid",
            cv=splits,
            ensemble=False,
        )
    else:
        model = estimator

    model.fit(corpus.features[fit_indices], corpus.labels[fit_indices])
    probabilities = model.predict_proba(corpus.features[test])
    return np.asarray(probabilities, dtype=np.float64)


def _fold_metrics(
    corpus: Corpus,
    test: np.ndarray,
    probabilities: np.ndarray,
) -> dict[str, object]:
    truth = corpus.labels[test]
    predictions = probabilities.argmax(axis=1)
    matrix = confusion_matrix(
        truth, predictions, labels=np.arange(len(corpus.gestures))
    )
    support = matrix.sum(axis=1)
    recall = np.divide(
        np.diag(matrix),
        support,
        out=np.full(len(support), np.nan, dtype=np.float64),
        where=support > 0,
    )
    movement_recall = recall[1:]
    return {
        "subject": int(corpus.subjects[test[0]]),
        "n_test": int(test.size),
        "accuracy": float(np.mean(predictions == truth)),
        "majority_baseline": majority_class_accuracy(truth),
        "expected_calibration_error": expected_calibration_error(probabilities, truth),
        "macro_movement_recall": float(np.nanmean(movement_recall)),
        "rest_recall": None if np.isnan(recall[0]) else float(recall[0]),
        "true_rest_fraction": float(np.mean(truth == 0)),
        "predicted_rest_fraction": float(np.mean(predictions == 0)),
        "per_gesture_recall": {
            name: None if np.isnan(value) else float(value)
            for name, value in zip(corpus.gestures, recall, strict=True)
        },
        "confusion_matrix": matrix.tolist(),
    }


def run_recognition_experiment(
    corpus: Corpus,
    *,
    seed: int = DEFAULT_SEED,
    svm_samples_per_class: int = 250,
    calibration_folds: int = 3,
) -> dict[str, object]:
    """Compare fixed recognition strategies across subject-held-out folds."""
    if calibration_folds < 2:
        raise ValueError("calibration_folds must be at least 2")
    strategies = _strategies(len(corpus.gestures))
    subjects = sorted(int(subject) for subject in np.unique(corpus.subjects))
    oof: dict[str, np.ndarray] = {
        strategy.name: np.zeros(
            (corpus.n_windows, len(corpus.gestures)), dtype=np.float64
        )
        for strategy in strategies
    }
    fold_results: dict[str, list[dict[str, object]]] = {
        strategy.name: [] for strategy in strategies
    }

    for fold_number, (train, test) in enumerate(loso_splits(corpus.subjects)):
        held_subject = int(corpus.subjects[test[0]])
        for strategy in strategies:
            name = strategy.name
            probabilities = _fit_predict(
                strategy,
                corpus,
                train,
                test,
                seed=seed + fold_number,
                svm_samples_per_class=svm_samples_per_class,
                calibration_folds=calibration_folds,
            )
            oof[name][test] = probabilities
            fold_results[name].append(
                _fold_metrics(corpus, test, probabilities)
            )
        print(f"completed subject-held-out fold S{held_subject}", flush=True)

    summaries: dict[str, object] = {}
    for strategy in strategies:
        name = strategy.name
        folds = fold_results[name]
        pooled = confusion_matrix(
            corpus.labels,
            oof[name].argmax(axis=1),
            labels=np.arange(len(corpus.gestures)),
        )
        support = pooled.sum(axis=1)
        recalls = np.divide(
            np.diag(pooled),
            support,
            out=np.full(len(support), np.nan, dtype=np.float64),
            where=support > 0,
        )
        summaries[name] = {
            "calibration": (
                "sigmoid, grouped by subject and repetition"
                if strategy.calibrated
                else "none; raw LDA probabilities"
            ),
            "training_adjustment": (
                "rest downsampled to median movement-class support"
                if strategy.balance == "rest_downsample"
                else (
                    f"at most {svm_samples_per_class} windows per class per "
                    "training subject"
                    if strategy.svm_cap
                    else "none"
                )
            ),
            "class_weight": (
                "balanced"
                if "weighted" in name
                else "not set"
            ),
            "mean_accuracy": float(np.mean([fold["accuracy"] for fold in folds])),
            "std_accuracy": float(np.std([fold["accuracy"] for fold in folds])),
            "mean_majority_baseline": float(
                np.mean([fold["majority_baseline"] for fold in folds])
            ),
            "mean_macro_movement_recall": float(
                np.mean([fold["macro_movement_recall"] for fold in folds])
            ),
            "pooled_macro_movement_recall": float(np.nanmean(recalls[1:])),
            "mean_predicted_rest_fraction": float(
                np.mean([fold["predicted_rest_fraction"] for fold in folds])
            ),
            "expected_calibration_error": expected_calibration_error(
                oof[name], corpus.labels
            ),
            "per_gesture_recall": {
                gesture: None if np.isnan(value) else float(value)
                for gesture, value in zip(corpus.gestures, recalls, strict=True)
            },
            "confusion_matrix": pooled.tolist(),
            "folds": folds,
        }

    return {
        "source": "ninapro_db2",
        "evaluation": "leave-one-subject-out; each subject held out exactly once",
        "subjects": subjects,
        "n_subjects": len(subjects),
        "n_windows": corpus.n_windows,
        "n_features": corpus.n_features,
        "gestures": list(corpus.gestures),
        "seed": seed,
        "svm_samples_per_class_per_training_subject": svm_samples_per_class,
        "calibration_folds": calibration_folds,
        "calibration_split": (
            "GroupKFold by subject × repetition inside each outer training fold; "
            "overlapping windows from a repetition are kept together"
        ),
        "caveat": (
            "Exploratory results from three local intact DB2 subjects. S1, S10, "
            "and S11 have already informed development; these folds are not an "
            "untouched final evaluation and do not establish clinical validity."
        ),
        "models": summaries,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=pathlib.Path, default=pathlib.Path("data/ninapro/DB2"))
    parser.add_argument(
        "--out",
        type=pathlib.Path,
        default=pathlib.Path("artifacts/ninapro_recognition_experiment.json"),
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--svm-samples-per-class", type=int, default=250)
    parser.add_argument("--calibration-folds", type=int, default=3)
    args = parser.parse_args()
    if args.svm_samples_per_class < 1:
        parser.error("--svm-samples-per-class must be positive")

    corpus = build_ninapro_corpus(args.data_root)
    result = run_recognition_experiment(
        corpus,
        seed=args.seed,
        svm_samples_per_class=args.svm_samples_per_class,
        calibration_folds=args.calibration_folds,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"wrote {args.out}")
    for name, metrics in result["models"].items():
        print(
            f"{name:<34} accuracy={metrics['mean_accuracy']:.4f} "
            f"movement_recall={metrics['mean_macro_movement_recall']:.4f} "
            f"predicted_rest={metrics['mean_predicted_rest_fraction']:.4f} "
            f"ECE={metrics['expected_calibration_error']:.4f}"
        )


if __name__ == "__main__":
    main()
