"""Train, export, and evaluate a subject-held-out NinaPro DB2 decoder."""

from __future__ import annotations

import argparse
import json
import pathlib
import time
from collections import defaultdict

import numpy as np
from sklearn.metrics import confusion_matrix

from neurogrip import FEATURE_SPEC_VERSION
from neurogrip.datasets import Corpus
from neurogrip.evaluation import expected_calibration_error, majority_class_accuracy
from neurogrip.export import export_to_onnx, load_session, predict_onnx, verify_parity
from neurogrip.ninapro import (
    DATASET_MOVEMENTS,
    DEFAULT_MAPPING,
    MAPPING_SOURCES,
    NinaProWindow,
    build_ninapro_corpus_with_provenance,
    load_session as load_ninapro_session,
)
from neurogrip.replay import write_bundle
from neurogrip.signal_quality import SignalQualityConfig, check_signal_quality
from neurogrip.simulator import GESTURES
from neurogrip.train import fit_final_model, model_by_name


def _subset(corpus: Corpus, indices: np.ndarray) -> Corpus:
    return Corpus(
        features=corpus.features[indices],
        labels=corpus.labels[indices],
        subjects=corpus.subjects[indices],
        reps=corpus.reps[indices],
        fatigue=corpus.fatigue[indices],
        feature_names=corpus.feature_names,
        gestures=corpus.gestures,
    )


def _percentiles(samples: np.ndarray) -> dict[str, float]:
    return {
        "p50_ms": round(float(np.percentile(samples, 50)), 4),
        "p95_ms": round(float(np.percentile(samples, 95)), 4),
        "p99_ms": round(float(np.percentile(samples, 99)), 4),
        "max_ms": round(float(samples.max()), 4),
    }


def _benchmark_onnx(path: pathlib.Path, features: np.ndarray) -> dict[str, object]:
    session = load_session(path)
    input_name = session.get_inputs()[0].name
    inputs = features[: min(features.shape[0], 1000)]
    for row in inputs[: min(inputs.shape[0], 20)]:
        session.run(None, {input_name: row.reshape(1, -1)})

    elapsed_ms = np.empty(inputs.shape[0], dtype=np.float64)
    for index, row in enumerate(inputs):
        start = time.perf_counter()
        session.run(None, {input_name: row.reshape(1, -1)})
        elapsed_ms[index] = (time.perf_counter() - start) * 1000

    return {
        "runtime": "onnxruntime CPUExecutionProvider",
        "n_windows": int(inputs.shape[0]),
        "batched": False,
        "scope": (
            "ONNX inference only on pre-extracted held-out recording windows; "
            "excludes windowing, feature extraction, worker messaging, and UI."
        ),
        **_percentiles(elapsed_ms),
    }


def _quality_gate_summary(
    data_root: pathlib.Path,
    provenance: tuple[NinaProWindow, ...],
    test_indices: np.ndarray,
    truth: np.ndarray,
    predicted: np.ndarray,
    min_rms_by_channel: tuple[float, ...],
) -> dict[str, object]:
    """Score the recorded-only quality gate against the held-out labeled windows."""
    config = SignalQualityConfig(min_channel_rms_by_channel=min_rms_by_channel)
    positions_by_recording: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for output_index, corpus_index in enumerate(test_indices):
        source = provenance[int(corpus_index)]
        positions_by_recording[source.recording].append(
            (output_index, source.sample_start)
        )

    accepted = np.zeros(test_indices.size, dtype=bool)
    for recording, positions in positions_by_recording.items():
        emg, _, _ = load_ninapro_session(data_root / recording)
        for output_index, sample_start in positions:
            window = emg[:, sample_start : sample_start + 400]
            accepted[output_index] = (
                window.shape[1] == 400
                and check_signal_quality(window, config).ok
            )

    accepted_count = int(accepted.sum())
    rejected_count = int(accepted.size - accepted_count)
    correct = predicted == truth
    rejected_by_gesture = {
        gesture: int(np.count_nonzero((truth == index) & ~accepted))
        for index, gesture in enumerate(GESTURES)
    }
    return {
        "method": (
            "Same per-channel floors as the NinaPro replay, evaluated on every "
            "held-out labeled 200 ms window; this is a signal-quality result, "
            "not a prediction."
        ),
        "windows_tested": int(accepted.size),
        "accepted": accepted_count,
        "rejected": rejected_count,
        "acceptance_rate": round(float(accepted.mean()), 4),
        "top1_accuracy_before_gate": round(float(correct.mean()), 4),
        "top1_accuracy_on_accepted": (
            None
            if accepted_count == 0
            else round(float(correct[accepted].mean()), 4)
        ),
        "rejected_by_performed_gesture": rejected_by_gesture,
    }


def _timeline(
    corpus: Corpus,
    provenance: tuple[NinaProWindow, ...],
    indices: np.ndarray,
    predictions: np.ndarray,
    hop_ms: int = 20,
) -> list[dict[str, object]]:
    """Compress each performed gesture/repetition into decoded label runs."""
    events: list[dict[str, object]] = []
    current: dict[str, object] | None = None
    run_class = -1
    run_gesture = ""
    run_windows = 0

    def finish_run() -> None:
        nonlocal run_class, run_windows
        if current is None or run_windows == 0:
            return
        decoded = current["decoded"]
        assert isinstance(decoded, list)
        decoded.append({
            "gesture": DATASET_MOVEMENTS.get(run_gesture, run_gesture),
            "windows": run_windows,
        })
        run_class = -1
        run_windows = 0

    def finish_event() -> None:
        nonlocal current
        finish_run()
        if current is not None:
            events.append(current)
            current = None

    for index, predicted in zip(indices.tolist(), predictions.tolist(), strict=True):
        source = provenance[index]
        if source.label == 0:
            finish_event()
            continue

        key = (source.recording, source.repetition, source.label)
        if current is not None:
            previous_start = int(current["last_sample"])
            same_event = current["key"] == key
            contiguous = source.sample_start == previous_start + 40
            if not same_event or not contiguous:
                finish_event()

        if current is None:
            current = {
                "recording": source.recording,
                "repetition": source.repetition,
                "performed": DATASET_MOVEMENTS[
                    GESTURES[int(corpus.labels[index])]
                ],
                "start_ms": round(source.sample_start / 2, 1),
                "end_ms": round((source.sample_start + 400) / 2, 1),
                "windows": 0,
                "decoded": [],
                "key": key,
                "last_sample": source.sample_start,
            }

        if run_windows and int(predicted) == run_class:
            run_windows += 1
        else:
            finish_run()
            run_class = int(predicted)
            run_gesture = GESTURES[int(predicted)]
            run_windows = 1

        current["windows"] = int(current["windows"]) + 1
        current["end_ms"] = round((source.sample_start + 400) / 2, 1)
        current["last_sample"] = source.sample_start

    finish_event()
    for event in events:
        event.pop("key")
        event.pop("last_sample")
        event.pop("decoded_runs", None)
    return events


def train_evaluate_heldout(
    data_root: pathlib.Path,
    train_subjects: tuple[int, ...],
    test_subject: int,
    out_dir: pathlib.Path,
    replay_dir: pathlib.Path,
    model_name: str = "lda",
) -> dict[str, object]:
    """Fit on the named subjects and evaluate only on the excluded subject."""
    training = set(train_subjects)
    if not training:
        raise ValueError("at least one --train-subjects value is required")
    if test_subject in training:
        raise ValueError("test subject must not appear in --train-subjects")

    corpus, provenance = build_ninapro_corpus_with_provenance(data_root)
    available = set(int(value) for value in np.unique(corpus.subjects))
    missing = (training | {test_subject}) - available
    if missing:
        raise ValueError(f"requested subjects are absent from the corpus: {sorted(missing)}")

    train_indices = np.flatnonzero(np.isin(corpus.subjects, tuple(sorted(training))))
    test_indices = np.flatnonzero(corpus.subjects == test_subject)
    train_corpus = _subset(corpus, train_indices)
    missing_classes = [
        corpus.gestures[index]
        for index in range(len(corpus.gestures))
        if not np.any(train_corpus.labels == index)
    ]
    if missing_classes:
        raise ValueError(
            "training subjects do not contain every mapped class: "
            + ", ".join(missing_classes)
        )

    spec = model_by_name(model_name)
    model = fit_final_model(train_corpus, spec)
    model_dir = out_dir / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = export_to_onnx(model, train_corpus.n_features, model_dir / "decoder.onnx")

    parity_count = min(2000, test_indices.size)
    parity = verify_parity(model, onnx_path, corpus.features[test_indices[:parity_count]])
    if not parity.ok:
        raise RuntimeError(
            "ONNX probabilities diverge from sklearn on held-out data; refusing to publish"
        )

    heldout_features = corpus.features[test_indices]
    predicted, probabilities = predict_onnx(load_session(onnx_path), heldout_features)
    truth = corpus.labels[test_indices]
    predicted_classes = predicted.astype(np.int16)
    matrix = confusion_matrix(truth, predicted_classes, labels=range(len(GESTURES)))
    support = matrix.sum(axis=1)
    recall = {
        gesture: (
            None
            if support[index] == 0
            else round(float(matrix[index, index] / support[index]), 4)
        )
        for index, gesture in enumerate(GESTURES)
    }
    latency = _benchmark_onnx(onnx_path, heldout_features)

    timeline = _timeline(corpus, provenance, test_indices, predicted_classes)
    replay_manifest = write_bundle(
        replay_dir,
        source="ninapro",
        root=data_root,
        subject=test_subject,
        trained_subjects=tuple(sorted(training)),
    )
    min_rms_by_channel = tuple(
        float(value)
        for value in replay_manifest["signalQuality"]["minChannelRmsByChannel"]
    )
    quality_gate = _quality_gate_summary(
        data_root,
        provenance,
        test_indices,
        truth,
        predicted_classes,
        min_rms_by_channel,
    )

    model_metadata = {
        "model": spec.name,
        "description": spec.description,
        "feature_spec_version": FEATURE_SPEC_VERSION,
        "n_features": train_corpus.n_features,
        "n_channels": 8,
        "feature_names": list(train_corpus.feature_names),
        "gestures": list(train_corpus.gestures),
        "gesture_labels": {"rest": "Rest", **DATASET_MOVEMENTS},
        "window_ms": 200,
        "hop_ms": 20,
        "sampling_rate_hz": 2000,
        "trained_on": "ninapro_db2",
        "training_subjects": sorted(training),
        "excluded_test_subject": test_subject,
        "onnx_parity": parity.summary(),
        "latency": latency,
        "caveat": (
            "Research prototype, not a medical device. Trained on recorded "
            "NinaPro DB2 data and tested on a separate subject; not clinically "
            "validated. Latency is CPU ONNX inference only."
        ),
    }
    (model_dir / "decoder.json").write_text(
        json.dumps(model_metadata, indent=2), encoding="utf-8"
    )

    result: dict[str, object] = {
        "source": "ninapro_db2",
        "model": spec.name,
        "training_subjects": sorted(training),
        "test_subject": test_subject,
        "subject_excluded_from_training": test_subject not in training,
        "n_train": int(train_indices.size),
        "n_test": int(test_indices.size),
        "gesture_order": list(GESTURES),
        "dataset_movements": DATASET_MOVEMENTS,
        "label_mapping": {gesture: label for gesture, label in DEFAULT_MAPPING.items()},
        "mapping_sources": MAPPING_SOURCES,
        "accuracy": round(float(np.mean(predicted_classes == truth)), 4),
        "baseline_accuracy": round(majority_class_accuracy(truth), 4),
        "expected_calibration_error": round(
            expected_calibration_error(probabilities, truth), 4
        ),
        "per_gesture_recall": recall,
        "confusion_matrix": matrix.tolist(),
        "quality_gate": quality_gate,
        "onnx_parity": parity.summary(),
        "latency": latency,
        "replay": {
            "source": replay_manifest["source"],
            "subject": replay_manifest["subject"],
            "trained_subjects": replay_manifest["trainedSubjects"],
            "path": str(replay_dir),
        },
        "timeline": timeline,
        "caveat": (
            "Exploratory subject-held-out result from the supplied NinaPro DB2 "
            "recordings. Only the listed training subject(s) were used; the test "
            "subject was excluded from fitting. The movement-name mapping must "
            "be interpreted using dataset_movements: DB2 label 5 is finger "
            "abduction and is only a proxy for the app's open-hand class. "
            "The same test subject has been used for model diagnostics, so these "
            "scores are development results rather than an untouched final test. "
            "Not clinical validation."
        ),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "evaluation.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=pathlib.Path, default=pathlib.Path("data/ninapro/DB2"))
    parser.add_argument("--train-subjects", required=True, help="comma-separated subject numbers")
    parser.add_argument("--test-subject", type=int, required=True)
    parser.add_argument("--model", default="lda")
    parser.add_argument("--out", type=pathlib.Path, default=pathlib.Path("artifacts/ninapro_holdout"))
    parser.add_argument(
        "--replay-out",
        type=pathlib.Path,
        default=pathlib.Path("apps/web/public/replay/ninapro"),
    )
    args = parser.parse_args()
    try:
        train_subjects = tuple(
            sorted({int(item) for item in args.train_subjects.split(",") if item.strip()})
        )
    except ValueError as exc:
        parser.error(f"--train-subjects must be comma-separated integers: {exc}")

    try:
        result = train_evaluate_heldout(
            args.data_root,
            train_subjects,
            args.test_subject,
            args.out,
            args.replay_out,
            args.model,
        )
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))

    print(
        f"{result['model']} held-out accuracy: {result['accuracy']:.4f} "
        f"({result['n_test']} windows, subject S{result['test_subject']})"
    )
    print(f"model: {args.out / 'model' / 'decoder.onnx'}")
    print(f"results: {args.out / 'evaluation.json'}")
    print(f"replay: {args.replay_out}")


if __name__ == "__main__":
    main()
