"""NinaPro DB2 ingestion, without bundling or redistributing its recordings.

DB2 is sampled at 2 kHz, matching the NeuroGrip feature pipeline.  The loader
keeps its first eight electrode channels so the recorded-data decoder has the
same input layout as the browser's eight-channel prosthesis interface.

The movement mapping is deliberately explicit and can be replaced by callers:
DB2 labels must be verified against the release notes before publishing a real
result.  Raw recordings belong under ``data/ninapro/DB2`` and are gitignored.
"""

from __future__ import annotations

import pathlib
from collections.abc import Mapping, Sequence

import numpy as np
from scipy.io import loadmat

from neurogrip.datasets import Corpus
from neurogrip.features import FeatureConfig, extract_features, feature_vector
from neurogrip.simulator import GESTURES
from neurogrip.spec import feature_names
from neurogrip.windowing import WindowConfig

# Provisional DB2 global labels.  Verify these against the DB2 movement list
# supplied with the exact download before treating an experiment as evidence.
DEFAULT_MAPPING: dict[str, int] = {
    "thumb_up": 1,
    "two_finger": 2,
    "open_hand": 5,
    "fist": 6,
    "point": 7,
    "wrist_flexion": 13,
    "wrist_extension": 14,
    "spherical_grip": 27,
    "pinch": 32,
}


def _as_vector(value: np.ndarray, name: str) -> np.ndarray:
    vector = np.asarray(value).squeeze()
    if vector.ndim != 1:
        raise ValueError(f"NinaPro {name!r} must be one-dimensional")
    return vector


def _scale_to_volts(emg: np.ndarray, labels: np.ndarray) -> float:
    """Conservative unit heuristic for DB2 MATLAB files.

    DB2 files encountered in the wild may store amplitudes in volts, mV, or
    microvolts.  The inferred scale is intentionally based on the rest region
    where available.  A caller can inspect it via ``infer_emg_scale_to_volts``
    before reporting a result; tests use values with an unambiguous scale.
    """
    rest = emg[labels == 0]
    reference = rest if rest.size else emg
    rms = float(np.sqrt(np.mean(np.square(reference, dtype=np.float64))))
    if rms > 1.0:
        return 1e-6  # microvolts
    if rms > 1e-3:
        return 1e-3  # millivolts
    return 1.0  # volts


def infer_emg_scale_to_volts(emg: np.ndarray, labels: np.ndarray) -> float:
    """Return the inferred multiplier that converts a DB2 matrix to volts."""
    return _scale_to_volts(np.asarray(emg), np.asarray(labels).reshape(-1))


def load_session(path: pathlib.Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load one DB2 ``S*_E*_A1.mat`` file as channels × samples in volts.

    ``restimulus`` is used rather than ``stimulus`` because it is DB2's
    realigned performed-movement label.  Only channels 1–8 are retained: the
    browser and exported decoder are intentionally fixed to the forearm ring.
    """
    raw = loadmat(path)
    missing = {"emg", "restimulus", "rerepetition"} - raw.keys()
    if missing:
        raise ValueError(f"{path} is missing NinaPro variables: {sorted(missing)}")

    emg = np.asarray(raw["emg"], dtype=np.float32)
    labels = _as_vector(raw["restimulus"], "restimulus").astype(np.int16)
    reps = _as_vector(raw["rerepetition"], "rerepetition").astype(np.int16)
    if emg.ndim != 2 or emg.shape[0] != labels.size or labels.size != reps.size:
        raise ValueError("emg, restimulus and rerepetition have incompatible shapes")
    if emg.shape[1] < 8:
        raise ValueError(f"{path} has {emg.shape[1]} EMG channels; NeuroGrip needs 8")

    scale = _scale_to_volts(emg, labels)
    return (emg[:, :8].T * scale).astype(np.float32), labels, reps


def _subject_number(path: pathlib.Path) -> int:
    token = path.name.split("_", 1)[0]
    if not token.startswith("S") or not token[1:].isdigit():
        raise ValueError(f"DB2 filename must start with S<number>_: {path.name}")
    return int(token[1:])


def build_ninapro_corpus(
    root: pathlib.Path,
    subjects: Sequence[int] | None = None,
    mapping: Mapping[str, int] = DEFAULT_MAPPING,
    rest_stride: int = 8,
) -> Corpus:
    """Feature-extract mapped DB2 movements into the common :class:`Corpus`.

    A 200 ms window is emitted every 20 ms only when every sample shares one
    ``restimulus`` label and repetition.  Rest is subsampled because its long
    pauses would otherwise dominate both fitting and reported accuracy.  Real
    recording fatigue is unknown, so its corpus field is zero by definition.
    """
    root = pathlib.Path(root)
    if rest_stride < 1:
        raise ValueError("rest_stride must be at least 1")
    selected = None if subjects is None else set(subjects)
    paths = sorted(root.glob("S*_E*_A*.mat"))
    if not paths:
        raise FileNotFoundError(f"no NinaPro DB2 session files under {root}")

    label_to_class = {0: 0}
    for gesture, label in mapping.items():
        if gesture not in GESTURES:
            raise ValueError(f"unknown NeuroGrip gesture in mapping: {gesture}")
        label_to_class[int(label)] = GESTURES.index(gesture)

    features: list[np.ndarray] = []
    labels_out: list[int] = []
    subject_out: list[int] = []
    reps_out: list[int] = []
    window = WindowConfig(window_ms=200, hop_ms=20, sampling_rate_hz=2000)
    feature_config = FeatureConfig(sampling_rate_hz=2000)
    rest_seen = 0

    for path in paths:
        subject = _subject_number(path)
        if selected is not None and subject not in selected:
            continue
        emg, labels, reps = load_session(path)
        for start in range(0, emg.shape[1] - window.window_samples + 1, window.hop_samples):
            stop = start + window.window_samples
            label_slice, rep_slice = labels[start:stop], reps[start:stop]
            if not np.all(label_slice == label_slice[0]) or not np.all(rep_slice == rep_slice[0]):
                continue
            raw_label, repetition = int(label_slice[0]), int(rep_slice[0])
            if raw_label not in label_to_class:
                continue
            if raw_label == 0:
                rest_seen += 1
                if rest_seen % rest_stride:
                    continue
            features.append(feature_vector(extract_features(emg[:, start:stop], feature_config), 8))
            labels_out.append(label_to_class[raw_label])
            subject_out.append(subject)
            reps_out.append(repetition)

    if not features:
        raise ValueError("no complete mapped NinaPro windows were found")
    return Corpus(
        features=np.asarray(features, dtype=np.float32),
        labels=np.asarray(labels_out, dtype=np.int16),
        subjects=np.asarray(subject_out, dtype=np.int16),
        reps=np.asarray(reps_out, dtype=np.int16),
        # DB2 does not provide a per-window fatigue ground truth.
        fatigue=np.zeros(len(features), dtype=np.float32),
        feature_names=feature_names(8),
        gestures=GESTURES,
    )
