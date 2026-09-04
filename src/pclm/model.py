from __future__ import annotations

from dataclasses import dataclass

import numpy as np


GESTURES = (
    "rest",
    "fist",
    "open_hand",
    "pinch",
    "point",
    "wrist_flexion",
    "wrist_extension",
    "thumb_up",
    "two_finger",
    "spherical_grip",
)


@dataclass(frozen=True)
class GesturePrediction:
    label: str
    confidence: float
    abstained: bool


class PlaceholderGestureModel:
    """Deterministic demo model used until trained SVM/LSTM artifacts exist."""

    def predict(self, vector: np.ndarray, confidence_threshold: float) -> GesturePrediction:
        if vector.size == 0:
            return GesturePrediction("uncertain", 0.0, True)

        energy = float(np.mean(np.abs(vector)))
        if energy < 1e-4:
            return GesturePrediction("uncertain", 0.0, True)

        index = int(abs(np.sum(vector) * 1000)) % (len(GESTURES) - 1) + 1
        confidence = min(0.95, 0.5 + np.log1p(energy) / 3)

        if confidence < confidence_threshold:
            return GesturePrediction("uncertain", confidence, True)

        return GesturePrediction(GESTURES[index], float(confidence), False)
