import numpy as np
import pytest

from neurogrip.datasets import Corpus, CorpusConfig, build_corpus
from neurogrip.simulator import GESTURES
from neurogrip.spec import feature_count, feature_names


@pytest.fixture(scope="module")
def small_corpus() -> Corpus:
    return build_corpus(
        CorpusConfig(n_subjects=2, reps_per_gesture=1, rep_seconds=0.4, n_electrodes=8)
    )


def test_corpus_shapes_are_consistent(small_corpus):
    c = small_corpus
    assert c.features.ndim == 2
    assert c.features.shape[1] == feature_count(8)
    assert c.labels.shape == (c.n_windows,)
    assert c.subjects.shape == (c.n_windows,)
    assert c.reps.shape == (c.n_windows,)
    assert c.fatigue.shape == (c.n_windows,)
    assert c.features.dtype == np.float32


def test_corpus_carries_spec_feature_names(small_corpus):
    assert small_corpus.feature_names == feature_names(8)
    assert small_corpus.gestures == GESTURES


def test_every_gesture_and_subject_is_represented(small_corpus):
    assert set(small_corpus.labels.tolist()) == set(range(len(GESTURES)))
    assert set(small_corpus.subjects.tolist()) == {0, 1}


def test_windows_per_repetition_follow_the_window_geometry(small_corpus):
    # 0.4 s at 2 kHz = 800 samples; 400-sample window, 40-sample hop -> 11 windows
    expected = 11 * len(GESTURES) * 1 * 2
    assert small_corpus.n_windows == expected


def test_corpus_is_deterministic():
    config = CorpusConfig(n_subjects=1, reps_per_gesture=1, rep_seconds=0.3)
    a = build_corpus(config)
    b = build_corpus(config)
    assert np.array_equal(a.features, b.features)
    assert np.array_equal(a.labels, b.labels)


def test_different_subjects_produce_different_features():
    config = CorpusConfig(n_subjects=2, reps_per_gesture=1, rep_seconds=0.3)
    corpus = build_corpus(config)
    first = corpus.features[corpus.subjects == 0].mean(axis=0)
    second = corpus.features[corpus.subjects == 1].mean(axis=0)
    assert not np.allclose(first, second, rtol=0.05)


def test_fatigue_accumulates_across_repetitions():
    corpus = build_corpus(
        CorpusConfig(n_subjects=1, reps_per_gesture=3, rep_seconds=0.3, max_fatigue=0.6)
    )
    levels = sorted(set(np.round(corpus.fatigue, 6).tolist()))
    assert levels[0] == 0.0
    assert levels[-1] == pytest.approx(0.6)
    assert len(levels) == 3


def test_electrode_shift_override_changes_the_features():
    base = CorpusConfig(n_subjects=1, reps_per_gesture=1, rep_seconds=0.3)
    shifted = CorpusConfig(
        n_subjects=1, reps_per_gesture=1, rep_seconds=0.3, electrode_shift_rad=0.4
    )
    assert not np.allclose(
        build_corpus(base).features, build_corpus(shifted).features, rtol=0.05
    )


def test_npz_round_trip(tmp_path, small_corpus):
    target = tmp_path / "corpus.npz"
    small_corpus.to_npz(target)
    loaded = Corpus.from_npz(target)
    assert np.array_equal(loaded.features, small_corpus.features)
    assert np.array_equal(loaded.labels, small_corpus.labels)
    assert np.array_equal(loaded.subjects, small_corpus.subjects)
    assert loaded.feature_names == small_corpus.feature_names
    assert loaded.gestures == small_corpus.gestures


def test_rejects_degenerate_configuration():
    with pytest.raises(ValueError):
        build_corpus(CorpusConfig(n_subjects=0))
    with pytest.raises(ValueError):
        build_corpus(CorpusConfig(reps_per_gesture=0))
