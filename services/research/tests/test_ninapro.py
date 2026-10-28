from __future__ import annotations

import numpy as np
import pytest
from scipy.io import savemat

from neurogrip.ninapro import build_ninapro_corpus, load_session
from neurogrip.simulator import GESTURES


def _write_session(path, *, channels: int = 12) -> None:
    # Four 200 ms homogeneous sections at DB2's 2 kHz rate.  Values are mV so
    # the loader's unit conversion is asserted as part of this fixture.
    n = 400
    labels = np.repeat(np.array([0, 1, 6, 1], dtype=np.int16), n)
    reps = np.repeat(np.array([0, 1, 2, 2], dtype=np.int16), n)
    emg = np.full((labels.size, channels), 0.02, dtype=np.float32)
    emg += np.arange(channels, dtype=np.float32)[None, :] * 0.001
    savemat(path, {"emg": emg, "restimulus": labels[:, None], "rerepetition": reps[:, None]})


def test_load_session_keeps_eight_channels_and_converts_mv_to_volts(tmp_path):
    path = tmp_path / "S1_E1_A1.mat"
    _write_session(path)
    emg, labels, reps = load_session(path)

    assert emg.shape == (8, 1600)
    assert emg.dtype == np.float32
    assert emg[0, 0] == pytest.approx(20e-6)
    assert labels.tolist()[:2] == [0, 0]
    assert reps.tolist()[-2:] == [2, 2]


def test_build_ninapro_corpus_drops_transition_windows_and_subsamples_rest(tmp_path):
    _write_session(tmp_path / "S1_E1_A1.mat")
    corpus = build_ninapro_corpus(
        tmp_path, subjects=[1], mapping={"thumb_up": 1, "fist": 6}, rest_stride=2
    )

    # One 200 ms window per homogeneous block; the one rest window is retained
    # only every second time, while movement labels preserve their DB2 reps.
    assert corpus.n_windows == 3
    assert corpus.features.shape == (3, 136)
    assert corpus.labels.tolist() == [GESTURES.index("thumb_up"), GESTURES.index("fist"), GESTURES.index("thumb_up")]
    assert corpus.reps.tolist() == [1, 2, 2]
    assert corpus.subjects.tolist() == [1, 1, 1]
    assert np.array_equal(corpus.fatigue, np.zeros(3, dtype=np.float32))


def test_load_session_rejects_a_file_without_required_variables(tmp_path):
    path = tmp_path / "S1_E1_A1.mat"
    savemat(path, {"emg": np.zeros((10, 12), dtype=np.float32)})
    with pytest.raises(ValueError, match="missing NinaPro variables"):
        load_session(path)
