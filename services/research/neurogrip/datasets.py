"""Labelled corpus construction from the simulator.

Turns the simulator into a model-ready dataset: subjects perform each gesture
for several repetitions, each repetition is framed into overlapping windows, and
each window becomes one feature vector with a label, a subject id, a repetition
index and a fatigue level.

Subject id is retained through every stage so leave-one-subject-out folds can be
built without leakage, and so an anomalous result can be traced back to the
recording that produced it. Fatigue is retained because the Pillar 3 drift
estimator needs a ground-truth fatigue axis to be validated against.

Everything is seeded from `CorpusConfig.base_seed`; two builds with the same
config are bit-identical.
"""

from __future__ import annotations

import pathlib
from dataclasses import dataclass

import numpy as np

from neurogrip.features import FeatureConfig, extract_features, feature_vector
from neurogrip.simulator import (
    GESTURES,
    SimulatorConfig,
    SubjectProfile,
    make_subject,
    simulate,
)
from neurogrip.spec import feature_names
from neurogrip.windowing import WindowConfig, frame_windows


@dataclass(frozen=True)
class CorpusConfig:
    n_subjects: int = 8
    reps_per_gesture: int = 4
    rep_seconds: float = 1.0
    sampling_rate_hz: int = 2000
    n_electrodes: int = 8
    window_ms: int = 200
    hop_ms: int = 20
    #: Peak fatigue reached on the final repetition. Fatigue ramps linearly
    #: across repetitions, as it would within a real recording session.
    max_fatigue: float = 0.5
    #: When set, overrides every subject's own electrode shift. Used to build
    #: the shifted corpora that the electrode-shift robustness curve needs.
    electrode_shift_rad: float | None = None
    base_seed: int = 1000


@dataclass(frozen=True)
class Corpus:
    features: np.ndarray  # (n_windows, n_features) float32
    labels: np.ndarray  # (n_windows,) int16 - index into `gestures`
    subjects: np.ndarray  # (n_windows,) int16
    reps: np.ndarray  # (n_windows,) int16
    fatigue: np.ndarray  # (n_windows,) float32
    feature_names: tuple[str, ...]
    gestures: tuple[str, ...]

    @property
    def n_windows(self) -> int:
        return int(self.features.shape[0])

    @property
    def n_features(self) -> int:
        return int(self.features.shape[1])

    @property
    def n_subjects(self) -> int:
        return int(np.unique(self.subjects).size)

    def to_npz(self, path: pathlib.Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            features=self.features,
            labels=self.labels,
            subjects=self.subjects,
            reps=self.reps,
            fatigue=self.fatigue,
            feature_names=np.array(self.feature_names),
            gestures=np.array(self.gestures),
        )

    @classmethod
    def from_npz(cls, path: pathlib.Path) -> Corpus:
        with np.load(path, allow_pickle=False) as data:
            return cls(
                features=data["features"],
                labels=data["labels"],
                subjects=data["subjects"],
                reps=data["reps"],
                fatigue=data["fatigue"],
                feature_names=tuple(str(name) for name in data["feature_names"]),
                gestures=tuple(str(name) for name in data["gestures"]),
            )


def _subject_for(index: int, config: CorpusConfig) -> SubjectProfile:
    subject = make_subject(f"s{index:02d}", seed=config.base_seed + index)
    if config.electrode_shift_rad is not None:
        # SubjectProfile is intentionally mutable so a robustness sweep can hold
        # every other subject characteristic fixed while varying only the shift.
        subject.electrode_shift_rad = config.electrode_shift_rad
    return subject


def build_corpus(config: CorpusConfig = CorpusConfig()) -> Corpus:
    """Build a labelled, windowed, feature-extracted corpus from the simulator."""
    if config.n_subjects < 1:
        raise ValueError("n_subjects must be at least 1")
    if config.reps_per_gesture < 1:
        raise ValueError("reps_per_gesture must be at least 1")
    if config.rep_seconds <= 0:
        raise ValueError("rep_seconds must be positive")

    sim_config = SimulatorConfig(
        sampling_rate_hz=config.sampling_rate_hz, n_electrodes=config.n_electrodes
    )
    win_config = WindowConfig(
        window_ms=config.window_ms,
        hop_ms=config.hop_ms,
        sampling_rate_hz=config.sampling_rate_hz,
    )
    feat_config = FeatureConfig(sampling_rate_hz=config.sampling_rate_hz)

    features: list[np.ndarray] = []
    labels: list[int] = []
    subjects: list[int] = []
    reps: list[int] = []
    fatigues: list[float] = []

    denominator = max(1, config.reps_per_gesture - 1)

    for subject_index in range(config.n_subjects):
        subject = _subject_for(subject_index, config)
        for label, gesture in enumerate(GESTURES):
            for rep in range(config.reps_per_gesture):
                fatigue = config.max_fatigue * rep / denominator
                seed = (
                    config.base_seed * 10
                    + subject_index * 1000
                    + label * 10
                    + rep
                )
                raw = simulate(
                    gesture, config.rep_seconds, sim_config, subject, seed, fatigue
                )
                for window in frame_windows(raw, win_config):
                    features.append(
                        feature_vector(
                            extract_features(window, feat_config), config.n_electrodes
                        )
                    )
                    labels.append(label)
                    subjects.append(subject_index)
                    reps.append(rep)
                    fatigues.append(fatigue)

    if not features:
        raise ValueError(
            "corpus is empty: rep_seconds is shorter than one analysis window"
        )

    return Corpus(
        features=np.asarray(features, dtype=np.float32),
        labels=np.asarray(labels, dtype=np.int16),
        subjects=np.asarray(subjects, dtype=np.int16),
        reps=np.asarray(reps, dtype=np.int16),
        fatigue=np.asarray(fatigues, dtype=np.float32),
        feature_names=feature_names(config.n_electrodes),
        gestures=GESTURES,
    )
