"""The trial grouping is an invariant of how the corpus is built, not a fact
the arrays record.

`build_corpus` appends window by window through nested loops, so one
repetition's windows are adjacent and in time order. A shuffle added upstream
for any good reason -- balancing folds, breaking a correlation -- would leave
every shape intact and silently turn each "trial" into a bag of windows from
across the session. TTUM computed over that would still be a number, and it
would be meaningless.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from neurogrip import FEATURE_SPEC_VERSION
from neurogrip.datasets import CorpusConfig, build_corpus
from neurogrip.posteriors import group_trials, write_posteriors


@pytest.fixture(scope="module")
def corpus():
    # Two subjects, one repetition: enough to have several trials without
    # paying for the full sweep.
    return build_corpus(CorpusConfig(n_subjects=2, reps_per_gesture=1, rep_seconds=0.5))


def test_trials_cover_every_window_exactly_once(corpus):
    trials = group_trials(corpus)
    covered = np.zeros(corpus.n_windows, dtype=np.int32)
    for trial in trials:
        covered[trial["start"] : trial["start"] + trial["length"]] += 1
    assert covered.min() == 1
    assert covered.max() == 1


def test_each_trial_is_one_subject_one_gesture_one_repetition(corpus):
    for trial in group_trials(corpus):
        span = slice(trial["start"], trial["start"] + trial["length"])
        assert set(corpus.subjects[span].tolist()) == {trial["subject"]}
        assert set(corpus.labels[span].tolist()) == {trial["label"]}
        assert set(corpus.reps[span].tolist()) == {trial["rep"]}


def test_there_is_one_trial_per_subject_gesture_repetition(corpus):
    """Contiguity, stated as a count.

    If the corpus were shuffled, the same (subject, gesture, rep) would appear
    in many separate runs and this count would exceed the number of
    repetitions actually performed.
    """
    trials = group_trials(corpus)
    expected = corpus.n_subjects * len(corpus.gestures) * 1
    assert len(trials) == expected


def test_trial_windows_are_in_time_order(corpus):
    """Fatigue rises through a repetition, so it is a clock the corpus carries.

    Windows within a trial must be non-decreasing in fatigue; a shuffle would
    break the ordering without changing any other property.
    """
    for trial in group_trials(corpus):
        span = slice(trial["start"], trial["start"] + trial["length"])
        fatigue = corpus.fatigue[span]
        assert np.all(np.diff(fatigue) >= -1e-6), trial


def test_written_posteriors_round_trip(corpus, tmp_path):
    rng = np.random.default_rng(0)
    probabilities = rng.random((corpus.n_windows, len(corpus.gestures)))
    probabilities /= probabilities.sum(axis=1, keepdims=True)

    manifest = write_posteriors(corpus, probabilities, "rbf_svm", tmp_path / "posteriors")

    written = np.frombuffer((tmp_path / "posteriors.bin").read_bytes(), dtype=np.float32)
    assert written.size == corpus.n_windows * len(corpus.gestures)
    np.testing.assert_allclose(
        written.reshape(probabilities.shape), probabilities, rtol=1e-6, atol=1e-7
    )

    on_disk = json.loads((tmp_path / "posteriors.json").read_text(encoding="utf-8"))
    assert on_disk == manifest
    assert on_disk["featureSpecVersion"] == FEATURE_SPEC_VERSION
    assert on_disk["outOfFold"] is True


def test_refuses_probabilities_that_do_not_cover_the_corpus(corpus, tmp_path):
    """A silent truncation here would shift every trial's window offsets."""
    short = np.zeros((corpus.n_windows - 1, len(corpus.gestures)))
    with pytest.raises(ValueError):
        write_posteriors(corpus, short, "rbf_svm", tmp_path / "posteriors")


def test_the_caveat_names_the_constant_excitation(corpus, tmp_path):
    """The number must not travel without the sentence that qualifies it.

    The simulator holds excitation constant for a whole repetition, so TTUM
    derived from these sequences measures evidence accrual from an already
    active contraction, not a wearer's reaction time.
    """
    probabilities = np.full((corpus.n_windows, len(corpus.gestures)), 0.1)
    manifest = write_posteriors(corpus, probabilities, "rbf_svm", tmp_path / "posteriors")
    assert "onset envelope" in manifest["caveat"]
    assert "lower bound" in manifest["caveat"]
