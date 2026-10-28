import numpy as np

from neurogrip.datasets import Corpus
from neurogrip.heldout import _timeline
from neurogrip.ninapro import NinaProWindow
from neurogrip.simulator import GESTURES


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
