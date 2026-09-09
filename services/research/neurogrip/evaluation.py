"""Subject-independent evaluation and calibration metrics.

Leave-one-subject-out is the only credible way to report accuracy here.
A subject-dependent split would train and test on windows from the same
recording session and overstate accuracy substantially, because consecutive
200 ms windows at a 20 ms hop overlap by 90% and are near-duplicates of one
another.

`Corpus.subjects` carries the grouping, so folds are built from it directly
rather than from any positional assumption about how the corpus was ordered.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
from scipy import stats


def loso_splits(subjects: np.ndarray) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield (train_indices, test_indices) holding out one subject at a time."""
    groups = np.asarray(subjects)
    unique = np.unique(groups)
    if unique.size < 2:
        raise ValueError(
            "leave-one-subject-out needs at least two subjects; "
            f"got {unique.size}"
        )
    for held in unique:
        test = np.flatnonzero(groups == held)
        train = np.flatnonzero(groups != held)
        yield train, test


def majority_class_accuracy(labels: np.ndarray) -> float:
    """Accuracy of always predicting the most common class."""
    values = np.asarray(labels)
    if values.size == 0:
        return 0.0
    _, counts = np.unique(values, return_counts=True)
    return float(counts.max() / values.size)


def expected_calibration_error(
    probabilities: np.ndarray, labels: np.ndarray, n_bins: int = 10
) -> float:
    """Expected Calibration Error over the predicted-class confidence.

    Bins predictions by their maximum class probability and measures the gap
    between stated confidence and observed accuracy in each bin, weighted by
    bin occupancy. A model that says "90% sure" should be right 90% of the time.
    """
    probs = np.asarray(probabilities, dtype=np.float64)
    truth = np.asarray(labels)
    if probs.ndim != 2:
        raise ValueError("probabilities must have shape (n_samples, n_classes)")
    if probs.shape[0] != truth.shape[0]:
        raise ValueError(
            f"probabilities has {probs.shape[0]} rows but labels has {truth.shape[0]}"
        )
    if probs.shape[0] == 0:
        return 0.0

    confidence = probs.max(axis=1)
    predicted = probs.argmax(axis=1)
    correct = (predicted == truth).astype(np.float64)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    error = 0.0
    for low, high in zip(edges[:-1], edges[1:]):
        # Include the left edge on the first bin so confidence exactly 0 counts.
        in_bin = (confidence > low) & (confidence <= high)
        if low == 0.0:
            in_bin |= confidence == 0.0
        occupancy = float(in_bin.sum())
        if occupancy == 0.0:
            continue
        gap = abs(correct[in_bin].mean() - confidence[in_bin].mean())
        error += (occupancy / probs.shape[0]) * gap
    return float(error)


def paired_fold_test(
    first: np.ndarray, second: np.ndarray
) -> tuple[float | None, float]:
    """Wilcoxon signed-rank test over paired per-fold accuracies.

    Non-parametric, which suits a handful of folds far better than a t-test.
    Returns (statistic, p_value); the statistic is None when every pair is
    identical, in which case there is no difference to test and p is 1.0.
    """
    a = np.asarray(first, dtype=np.float64)
    b = np.asarray(second, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError("paired test needs equal-length accuracy vectors")
    if np.allclose(a, b):
        return None, 1.0
    result = stats.wilcoxon(a, b)
    return float(result.statistic), float(result.pvalue)


@dataclass(frozen=True)
class FoldResult:
    subject: int
    accuracy: float
    n_test: int
    confusion: np.ndarray


@dataclass(frozen=True)
class LosoResult:
    folds: tuple[FoldResult, ...]
    gestures: tuple[str, ...]

    @property
    def accuracies(self) -> np.ndarray:
        return np.array([fold.accuracy for fold in self.folds], dtype=np.float64)

    @property
    def mean_accuracy(self) -> float:
        return float(self.accuracies.mean())

    @property
    def std_accuracy(self) -> float:
        return float(self.accuracies.std())

    @property
    def min_accuracy(self) -> float:
        return float(self.accuracies.min())

    @property
    def confusion(self) -> np.ndarray:
        return np.sum([fold.confusion for fold in self.folds], axis=0)

    def confidence_interval(self, level: float = 0.95) -> tuple[float, float]:
        """Student-t interval over per-fold accuracies.

        Reported instead of a bare mean because with a handful of subjects the
        spread between them matters more than the average, and a point estimate
        invites overclaiming.
        """
        values = self.accuracies
        if values.size < 2:
            return float(values[0]), float(values[0])
        margin = stats.t.ppf(
            0.5 + level / 2.0, df=values.size - 1
        ) * values.std(ddof=1) / np.sqrt(values.size)
        # Accuracy is a proportion, so the interval is clipped to [0, 1]. A
        # symmetric t-interval on a mean near 1.0 with few folds routinely
        # exceeds 1.0, and reporting an accuracy above 100% would be plainly
        # wrong. Clipping is the honest presentation of a wide interval, not a
        # narrowing of it: when a bound is clipped the interval was already
        # wide enough to touch the limit of the scale.
        low = float(np.clip(values.mean() - margin, 0.0, 1.0))
        high = float(np.clip(values.mean() + margin, 0.0, 1.0))
        return low, high

    def per_gesture_recall(self) -> dict[str, float]:
        """Recall per gesture. NaN where a gesture never appeared as a label."""
        matrix = self.confusion
        support = matrix.sum(axis=1)
        recall: dict[str, float] = {}
        for index, gesture in enumerate(self.gestures):
            recall[gesture] = (
                float(matrix[index, index] / support[index])
                if support[index] > 0
                else float("nan")
            )
        return recall
