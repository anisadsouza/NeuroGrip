"""Electrode-shift robustness evaluation.

The classifier is trained on the unshifted corpus and tested on shifted copies
of the held-out subject. This measures the clinical failure mode the report
names: a decoder that works only for one exact electrode placement is brittle.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import confusion_matrix

from neurogrip.datasets import Corpus, CorpusConfig, build_corpus
from neurogrip.evaluation import FoldResult, LosoResult, loso_splits
from neurogrip.train import ModelSpec, _calibrated_pipeline


@dataclass(frozen=True)
class ShiftResult:
    shift_rad: float
    loso: LosoResult
    baseline_accuracy: float

    @property
    def mean_accuracy(self) -> float:
        return self.loso.mean_accuracy

    @property
    def degradation(self) -> float:
        return self.baseline_accuracy - self.mean_accuracy

    def summary(self) -> dict[str, object]:
        low, high = self.loso.confidence_interval()
        return {
            "shift_rad": round(self.shift_rad, 4),
            "mean_accuracy": round(self.mean_accuracy, 4),
            "std_accuracy": round(self.loso.std_accuracy, 4),
            "min_accuracy": round(self.loso.min_accuracy, 4),
            "ci95_low": round(low, 4),
            "ci95_high": round(high, 4),
            "degradation_pp": round(self.degradation * 100, 2),
            "per_fold_accuracy": [
                round(value, 4) for value in self.loso.accuracies.tolist()
            ],
        }


def _shifted_config(config: CorpusConfig, shift_rad: float) -> CorpusConfig:
    return CorpusConfig(
        n_subjects=config.n_subjects,
        reps_per_gesture=config.reps_per_gesture,
        rep_seconds=config.rep_seconds,
        sampling_rate_hz=config.sampling_rate_hz,
        n_electrodes=config.n_electrodes,
        window_ms=config.window_ms,
        hop_ms=config.hop_ms,
        max_fatigue=config.max_fatigue,
        electrode_shift_rad=shift_rad,
        base_seed=config.base_seed,
    )


def _assert_same_index(base: Corpus, shifted: Corpus) -> None:
    if not np.array_equal(base.labels, shifted.labels):
        raise ValueError("shifted corpus labels do not match base corpus")
    if not np.array_equal(base.subjects, shifted.subjects):
        raise ValueError("shifted corpus subjects do not match base corpus")
    if base.feature_names != shifted.feature_names:
        raise ValueError("shifted corpus feature layout differs from base corpus")


def evaluate_shift(
    base: Corpus,
    shifted: Corpus,
    spec: ModelSpec,
    calibration_folds: int = 3,
) -> ShiftResult:
    """LOSO train on base placements, test on shifted held-out placements."""
    _assert_same_index(base, shifted)

    n_classes = len(base.gestures)
    folds: list[FoldResult] = []

    for train, test in loso_splits(base.subjects):
        model = _calibrated_pipeline(spec, calibration_folds)
        model.fit(base.features[train], base.labels[train])
        probabilities = model.predict_proba(shifted.features[test])
        predictions = probabilities.argmax(axis=1)
        held = int(base.subjects[test][0])
        folds.append(
            FoldResult(
                subject=held,
                accuracy=float((predictions == base.labels[test]).mean()),
                n_test=int(test.size),
                confusion=confusion_matrix(
                    base.labels[test], predictions, labels=range(n_classes)
                ),
            )
        )

    return ShiftResult(
        shift_rad=0.0,
        loso=LosoResult(folds=tuple(folds), gestures=base.gestures),
        baseline_accuracy=float("nan"),
    )


def robustness_curve(
    config: CorpusConfig,
    spec: ModelSpec,
    shifts_rad: tuple[float, ...],
    calibration_folds: int = 3,
) -> dict[str, object]:
    if not shifts_rad:
        raise ValueError("at least one shift value is required")

    base = build_corpus(config)
    baseline = evaluate_shift(base, base, spec, calibration_folds)
    baseline_accuracy = baseline.mean_accuracy

    results: list[dict[str, object]] = []
    for shift in shifts_rad:
        shifted = build_corpus(_shifted_config(config, shift))
        result = evaluate_shift(base, shifted, spec, calibration_folds)
        result = ShiftResult(
            shift_rad=shift,
            loso=result.loso,
            baseline_accuracy=baseline_accuracy,
        )
        results.append(result.summary())

    return {
        "model": spec.name,
        "corpus": {
            "n_subjects": config.n_subjects,
            "reps_per_gesture": config.reps_per_gesture,
            "rep_seconds": config.rep_seconds,
            "n_electrodes": config.n_electrodes,
            "window_ms": config.window_ms,
            "hop_ms": config.hop_ms,
            "max_fatigue": config.max_fatigue,
            "base_seed": config.base_seed,
            "n_windows": base.n_windows,
            "n_features": base.n_features,
        },
        "baseline_accuracy": round(baseline_accuracy, 4),
        "shifts": results,
        "acceptance": {
            "max_allowed_degradation_pp": 8.0,
            "meets_bound": all(
                float(result["degradation_pp"]) <= 8.0 for result in results
            ),
        },
        "caveat": (
            "Shift is simulated by overriding electrode placement in the "
            "motor-unit simulator. This is a robustness screen, not a substitute "
            "for repeated real electrode placement on human subjects."
        ),
    }
