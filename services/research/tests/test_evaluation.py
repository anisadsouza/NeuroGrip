import numpy as np
import pytest

from neurogrip.evaluation import (
    FoldResult,
    LosoResult,
    expected_calibration_error,
    loso_splits,
    majority_class_accuracy,
    paired_fold_test,
)


def test_loso_yields_one_fold_per_subject_with_no_leakage():
    subjects = np.array([0, 0, 1, 1, 2, 2])
    folds = list(loso_splits(subjects))
    assert len(folds) == 3
    for held, (train, test) in zip(sorted(set(subjects.tolist())), folds):
        assert set(subjects[test].tolist()) == {held}
        assert held not in set(subjects[train].tolist())
        assert len(set(train.tolist()) & set(test.tolist())) == 0
        assert len(train) + len(test) == len(subjects)


def test_loso_requires_at_least_two_subjects():
    with pytest.raises(ValueError):
        list(loso_splits(np.array([0, 0, 0])))


def test_majority_class_accuracy():
    assert majority_class_accuracy(np.array([0, 0, 0, 1])) == pytest.approx(0.75)
    assert majority_class_accuracy(np.array([0, 1, 2, 3])) == pytest.approx(0.25)


def _fold(subject: int, correct: int, wrong: int, n_classes: int = 2) -> FoldResult:
    confusion = np.zeros((n_classes, n_classes), dtype=np.int64)
    confusion[0, 0] = correct
    confusion[0, 1] = wrong
    return FoldResult(
        subject=subject,
        accuracy=correct / (correct + wrong),
        n_test=correct + wrong,
        confusion=confusion,
    )


def test_loso_result_aggregates_folds():
    result = LosoResult(
        folds=(_fold(0, 8, 2), _fold(1, 6, 4)), gestures=("a", "b")
    )
    assert result.mean_accuracy == pytest.approx(0.7)
    assert result.min_accuracy == pytest.approx(0.6)
    assert result.std_accuracy == pytest.approx(0.1)
    assert result.confusion.sum() == 20


def test_loso_result_reports_per_gesture_recall():
    result = LosoResult(folds=(_fold(0, 8, 2),), gestures=("a", "b"))
    recall = result.per_gesture_recall()
    assert recall["a"] == pytest.approx(0.8)
    assert np.isnan(recall["b"])  # class never appeared as a true label


def test_confidence_interval_widens_with_variance():
    tight = LosoResult(
        folds=(_fold(0, 8, 2), _fold(1, 8, 2), _fold(2, 8, 2)), gestures=("a", "b")
    )
    loose = LosoResult(
        folds=(_fold(0, 10, 0), _fold(1, 5, 5), _fold(2, 8, 2)), gestures=("a", "b")
    )
    tight_low, tight_high = tight.confidence_interval()
    loose_low, loose_high = loose.confidence_interval()
    assert (tight_high - tight_low) < (loose_high - loose_low)


def test_expected_calibration_error_is_zero_for_perfect_confidence():
    # Always predicts class 1 with probability 1.0, and is always right.
    probabilities = np.tile(np.array([0.0, 1.0]), (50, 1))
    labels = np.ones(50, dtype=int)
    assert expected_calibration_error(probabilities, labels) == pytest.approx(0.0)


def test_expected_calibration_error_penalises_overconfidence():
    # Claims 100% confidence but is right only half the time.
    probabilities = np.tile(np.array([0.0, 1.0]), (50, 1))
    labels = np.array([1] * 25 + [0] * 25)
    assert expected_calibration_error(probabilities, labels) == pytest.approx(0.5)


def test_expected_calibration_error_rejects_shape_mismatch():
    with pytest.raises(ValueError):
        expected_calibration_error(np.zeros((3, 2)), np.zeros(4, dtype=int))


def test_paired_fold_test_detects_a_consistent_difference():
    better = np.array([0.90, 0.92, 0.88, 0.91, 0.89, 0.93])
    worse = np.array([0.80, 0.82, 0.78, 0.81, 0.79, 0.83])
    statistic, p_value = paired_fold_test(better, worse)
    assert p_value < 0.05
    assert statistic is not None


def test_paired_fold_test_finds_no_difference_between_identical_folds():
    same = np.array([0.90, 0.92, 0.88, 0.91])
    _, p_value = paired_fold_test(same, same.copy())
    assert p_value == 1.0


def test_confidence_interval_is_clipped_to_the_probability_scale():
    """A symmetric t-interval on a near-perfect mean exceeds 1.0. Reporting an
    accuracy above 100% would be plainly wrong."""
    near_perfect = LosoResult(
        folds=(_fold(0, 99, 1), _fold(1, 98, 2), _fold(2, 100, 0)),
        gestures=("a", "b"),
    )
    low, high = near_perfect.confidence_interval()
    assert high <= 1.0
    assert low >= 0.0


def test_confidence_interval_lower_bound_cannot_go_negative():
    near_zero = LosoResult(
        folds=(_fold(0, 1, 99), _fold(1, 0, 100), _fold(2, 2, 98)),
        gestures=("a", "b"),
    )
    low, _ = near_zero.confidence_interval()
    assert low >= 0.0


def test_single_fold_confidence_interval_is_the_point_estimate():
    single = LosoResult(folds=(_fold(0, 8, 2),), gestures=("a", "b"))
    assert single.confidence_interval() == (0.8, 0.8)
