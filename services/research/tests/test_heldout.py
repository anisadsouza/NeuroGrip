import numpy as np

from neurogrip.datasets import Corpus
from neurogrip.heldout import _quality_gate_summary, _timeline
from neurogrip.ninapro import NinaProWindow
from neurogrip.simulator import GESTURES


def test_quality_gate_summary_scores_each_labeled_window(monkeypatch, tmp_path):
    emg = np.full((8, 800), 2e-6, dtype=np.float32)
    emg[2, 400:] = 0
    monkeypatch.setattr(
        "neurogrip.heldout.load_ninapro_session",
        lambda _path: (emg, np.zeros(800, dtype=np.int16), np.zeros(800, dtype=np.int16)),
    )
    provenance = tuple(
        NinaProWindow(
            subject=10,
            recording="S10_E1_A1.mat",
            sample_start=start,
            label=raw_label,
            repetition=1,
        )
        for start, raw_label in ((0, 0), (400, 6))
    )

    result = _quality_gate_summary(
        tmp_path,
        provenance,
        np.array([0, 1]),
        np.array([GESTURES.index("rest"), GESTURES.index("fist")]),
        np.array([GESTURES.index("rest"), GESTURES.index("rest")]),
        (1e-6,) * 8,
    )

    assert result["windows_tested"] == 2
    assert result["accepted"] == 1
    assert result["rejected"] == 1
    assert result["top1_accuracy_before_gate"] == 0.5
    assert result["top1_accuracy_on_accepted"] == 1.0
    assert result["rejected_by_performed_gesture"]["fist"] == 1


def test_timeline_preserves_performed_trials_and_compresses_decoded_runs():
    indices = [
        GESTURES.index("thumb_up"),
        GESTURES.index("thumb_up"),
        GESTURES.index("thumb_up"),
        GESTURES.index("fist"),
    ]
    corpus = Corpus(
        features=np.zeros((4, 1), dtype=np.float32),
        labels=np.asarray(indices, dtype=np.int16),
        subjects=np.full(4, 10, dtype=np.int16),
        reps=np.ones(4, dtype=np.int16),
        fatigue=np.zeros(4, dtype=np.float32),
        feature_names=("feature",),
        gestures=GESTURES,
    )
    provenance = tuple(
        NinaProWindow(
            subject=10,
            recording="S10_E1_A1.mat",
            sample_start=start,
            label=raw_label,
            repetition=1,
        )
        for start, raw_label in zip((400, 440, 480, 520), (1, 1, 1, 6), strict=True)
    )

    timeline = _timeline(
        corpus,
        provenance,
        np.arange(4),
        np.asarray(
            [
                GESTURES.index("thumb_up"),
                GESTURES.index("point"),
                GESTURES.index("point"),
                GESTURES.index("fist"),
            ],
            dtype=np.int16,
        ),
    )

    assert len(timeline) == 2
    assert timeline[0]["performed"] == "Thumb up"
    assert timeline[0]["start_ms"] == 200
    assert timeline[0]["end_ms"] == 440
    assert timeline[0]["decoded"] == [
        {"gesture": "Thumb up", "windows": 1},
        {"gesture": "Pointing index", "windows": 2},
    ]
    assert timeline[1]["performed"] == "Fingers flexed together in fist"
    assert timeline[1]["decoded"] == [
        {"gesture": "Fingers flexed together in fist", "windows": 1}
    ]
