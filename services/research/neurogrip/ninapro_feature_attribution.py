"""Explain held-out errors with an exact linear-model score decomposition.

Run from the repository root:

    python -m neurogrip.ninapro_feature_attribution \
        --data-root data/ninapro/DB2 \
        --out artifacts/ninapro_feature_attribution.json

The primary research model is uncalibrated uniform-prior LDA, not the deployed
ONNX decoder. For each primary-model error, feature contributions exactly
decompose the standardized linear score margin between the predicted and
intended classes. A separately trained, class-weighted RBF SVM is a reference
comparator; it is not assumed to be better.
"""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from neurogrip.datasets import Corpus
from neurogrip.evaluation import loso_splits
from neurogrip.ninapro import build_ninapro_corpus
from neurogrip.ninapro_recognition import _cap_per_subject_class

REFERENCE_SVM_CAP = 250
TOP_FEATURES = 20


def _contributions(
    standardized: np.ndarray,
    truth: np.ndarray,
    predicted: np.ndarray,
    classes: np.ndarray,
    coefficients: np.ndarray,
    intercepts: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Return exact per-feature contributions to each predicted-vs-true margin."""
    standardized = np.asarray(standardized, dtype=np.float64)
    coefficients = np.asarray(coefficients, dtype=np.float64)
    intercepts = np.asarray(intercepts, dtype=np.float64)
    class_index = {int(label): index for index, label in enumerate(classes)}
    contributions = np.empty_like(standardized, dtype=np.float64)
    max_residual = 0.0
    for row, (actual, guess) in enumerate(zip(truth, predicted, strict=True)):
        actual_index = class_index[int(actual)]
        guess_index = class_index[int(guess)]
        contributions[row] = standardized[row] * (
            coefficients[guess_index] - coefficients[actual_index]
        )
        margin = (
            standardized[row] @ coefficients[guess_index]
            + intercepts[guess_index]
            - standardized[row] @ coefficients[actual_index]
            - intercepts[actual_index]
        )
        reconstructed = (
            contributions[row].sum()
            + intercepts[guess_index]
            - intercepts[actual_index]
        )
        max_residual = max(max_residual, abs(float(margin - reconstructed)))
    return contributions, max_residual


def run_feature_attribution(
    corpus: Corpus,
    *,
    seed: int = 20261104,
    svm_samples_per_class: int = REFERENCE_SVM_CAP,
) -> dict[str, object]:
    if corpus.n_subjects < 2:
        raise ValueError("subject-held-out attribution needs at least two subjects")
    if corpus.features.shape[1] != len(corpus.feature_names):
        raise ValueError("feature names do not match the corpus feature matrix")
    if svm_samples_per_class < 1:
        raise ValueError("SVM per-class sample cap must be positive")

    classes = np.unique(corpus.labels)
    primary_truth: list[np.ndarray] = []
    primary_predictions: list[np.ndarray] = []
    reference_predictions: list[np.ndarray] = []
    feature_abs_sum = np.zeros(corpus.n_features, dtype=np.float64)
    feature_signed_sum = np.zeros(corpus.n_features, dtype=np.float64)
    error_count = 0
    max_residual = 0.0
    folds: list[dict[str, object]] = []

    for fold_number, (train, test) in enumerate(loso_splits(corpus.subjects)):
        scaler = StandardScaler()
        train_features = scaler.fit_transform(corpus.features[train])
        test_features = scaler.transform(corpus.features[test])

        primary = LinearDiscriminantAnalysis(
            priors=np.full(classes.size, 1.0 / classes.size)
        )
        primary.fit(train_features, corpus.labels[train])
        primary_prediction = primary.predict(test_features)
        reference_prediction = np.empty(test.size, dtype=corpus.labels.dtype)

        capped = _cap_per_subject_class(
            train,
            corpus.subjects,
            corpus.labels,
            svm_samples_per_class,
            seed + fold_number,
        )
        reference_scaler = StandardScaler()
        reference_train = reference_scaler.fit_transform(corpus.features[capped])
        reference_test = reference_scaler.transform(corpus.features[test])
        reference = SVC(C=10.0, class_weight="balanced", gamma="scale")
        reference.fit(reference_train, corpus.labels[capped])
        reference_prediction[:] = reference.predict(reference_test)

        truth = corpus.labels[test]
        errors = primary_prediction != truth
        if np.any(errors):
            scores = primary.decision_function(test_features[errors])
            score_prediction = primary.classes_[np.argmax(scores, axis=1)]
            if not np.array_equal(score_prediction, primary_prediction[errors]):
                raise RuntimeError("LDA score decomposition disagrees with model predictions")
            contributions, residual = _contributions(
                test_features[errors],
                truth[errors],
                primary_prediction[errors],
                primary.classes_,
                primary.coef_,
                primary.intercept_,
            )
            feature_abs_sum += np.abs(contributions).sum(axis=0)
            feature_signed_sum += contributions.sum(axis=0)
            error_count += int(errors.sum())
            max_residual = max(max_residual, residual)

        primary_truth.append(truth)
        primary_predictions.append(primary_prediction)
        reference_predictions.append(reference_prediction)
        folds.append(
            {
                "subject": int(corpus.subjects[test[0]]),
                "n_test": int(test.size),
                "primary_accuracy": float(np.mean(primary_prediction == truth)),
                "reference_accuracy": float(np.mean(reference_prediction == truth)),
                "primary_errors": int(errors.sum()),
                "reference_correct_on_primary_errors": (
                    float(np.mean(reference_prediction[errors] == truth[errors]))
                    if np.any(errors)
                    else None
                ),
            }
        )

    all_truth = np.concatenate(primary_truth)
    all_primary = np.concatenate(primary_predictions)
    all_reference = np.concatenate(reference_predictions)
    order = np.argsort(feature_abs_sum, kind="stable")[::-1][:TOP_FEATURES]
    top_features = [
        {
            "feature": corpus.feature_names[int(index)],
            "meanAbsoluteMarginContribution": float(
                feature_abs_sum[index] / max(1, error_count)
            ),
            "meanSignedMarginContribution": float(
                feature_signed_sum[index] / max(1, error_count)
            ),
        }
        for index in order
    ]
    primary_error_mask = all_primary != all_truth
    return {
        "source": "ninapro_db2",
        "evaluation": "leave-one-subject-out; each subject held out exactly once",
        "subjects": sorted(int(value) for value in np.unique(corpus.subjects)),
        "n_windows": corpus.n_windows,
        "primary_model": (
            "research comparator: uncalibrated uniform-prior LDA with "
            "training-fold standardization; not the deployed ONNX decoder"
        ),
        "reference_model": (
            "RBF SVM (C=10, class_weight=balanced, gamma=scale), "
            f"at most {svm_samples_per_class} samples per class per training subject"
        ),
        "primary_accuracy": float(np.mean(all_primary == all_truth)),
        "reference_accuracy": float(np.mean(all_reference == all_truth)),
        "primary_error_count": int(primary_error_mask.sum()),
        "reference_correct_on_primary_errors": (
            float(np.mean(all_reference[primary_error_mask] == all_truth[primary_error_mask]))
            if np.any(primary_error_mask)
            else None
        ),
        "attribution": {
            "method": (
                "For each misclassification, decompose the standardized LDA "
                "predicted-class minus intended-class score margin into exact "
                "per-feature contributions; report mean absolute and signed "
                "contributions over out-of-fold errors."
            ),
            "n_errors_attributed": error_count,
            "max_score_decomposition_residual": max_residual,
            "top_features": top_features,
            "caveat": (
                "Exploratory explanation of this LDA model's held-out score "
                "margins, not causal physiology, clinical coaching, or evidence "
                "that changing a feature will improve recognition. The RBF SVM "
                "is a trained comparator, not presumed superior. All available "
                "subjects have informed development. This uncalibrated research "
                "model is not the deployed ONNX decoder."
            ),
        },
        "folds": folds,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=pathlib.Path, default=pathlib.Path("data/ninapro/DB2"))
    parser.add_argument(
        "--out",
        type=pathlib.Path,
        default=pathlib.Path("artifacts/ninapro_feature_attribution.json"),
    )
    parser.add_argument("--seed", type=int, default=20261104)
    parser.add_argument("--svm-samples-per-class", type=int, default=REFERENCE_SVM_CAP)
    args = parser.parse_args()
    report = run_feature_attribution(
        build_ninapro_corpus(args.data_root),
        seed=args.seed,
        svm_samples_per_class=args.svm_samples_per_class,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        f"wrote {args.out}; primary={report['primary_accuracy']:.3f}, "
        f"reference={report['reference_accuracy']:.3f}, "
        f"attributed errors={report['attribution']['n_errors_attributed']}"
    )


if __name__ == "__main__":
    main()
