import numpy as np

from neurogrip.ninapro_recognition import (
    _calibration_groups,
    _cap_per_subject_class,
    _downsample_rest,
    _grouped_calibration_splits,
)


def test_grouped_calibration_splits_keep_subject_repetitions_together():
    subjects = np.repeat([1, 10], 60)
    repetitions = np.tile(np.repeat(np.arange(6), 10), 2)
    groups = _calibration_groups(subjects, repetitions)
    labels = np.tile(np.repeat(np.arange(3), 20), 2)

    splits = _grouped_calibration_splits(groups, labels, n_splits=3)

    validation_indices = []
    for train, validation in splits:
        assert not set(groups[train]).intersection(groups[validation])
        assert set(labels[train]) == set(np.unique(labels))
        validation_indices.extend(validation.tolist())
    assert sorted(validation_indices) == list(range(len(labels)))


def test_rest_downsampling_preserves_movements_and_is_seeded():
    labels = np.array([0] * 30 + [1] * 10 + [2] * 20 + [3] * 15)
    indices = np.arange(labels.size)

    first = _downsample_rest(indices, labels, n_classes=4, seed=7)
    second = _downsample_rest(indices, labels, n_classes=4, seed=7)

    np.testing.assert_array_equal(first, second)
    assert np.count_nonzero(labels[first] == 0) == 15
    np.testing.assert_array_equal(
        np.sort(labels[first][labels[first] != 0]),
        np.sort(labels[indices][labels[indices] != 0]),
    )


def test_per_subject_class_cap_is_deterministic_and_respects_the_cap():
    subjects = np.repeat([1, 10], 80)
    labels = np.tile(np.repeat([0, 1], 40), 2)
    indices = np.arange(labels.size)

    first = _cap_per_subject_class(indices, subjects, labels, cap=7, seed=19)
    second = _cap_per_subject_class(indices, subjects, labels, cap=7, seed=19)

    np.testing.assert_array_equal(first, second)
    for subject in np.unique(subjects):
        for label in np.unique(labels):
            selected = (subjects[first] == subject) & (labels[first] == label)
            assert selected.sum() == 7
