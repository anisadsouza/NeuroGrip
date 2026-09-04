"""Reproducible experiment runners.

Kept out of the test suite because a full leave-one-subject-out sweep takes
minutes, not milliseconds. Results are written to JSON so the report and the
model card quote a recorded run rather than a remembered number.

    python -m neurogrip.experiments compare --out artifacts/model_comparison.json
"""

from __future__ import annotations

import argparse
import json
import pathlib
import time

import numpy as np

from neurogrip import FEATURE_SPEC_VERSION
from neurogrip.datasets import CorpusConfig, build_corpus
from neurogrip.evaluation import paired_fold_test
from neurogrip.export import export_to_onnx, load_session, predict_onnx, verify_parity
from neurogrip.features import FeatureConfig, extract_features, feature_vector
from neurogrip.simulator import SimulatorConfig, make_subject, simulate
from neurogrip.train import MODEL_ZOO, evaluate_model, fit_final_model, model_by_name, select_best
from neurogrip.windowing import WindowConfig, frame_windows


def compare_models(config: CorpusConfig, calibration_folds: int = 3) -> dict:
    started = time.perf_counter()
    print(
        f"building corpus: {config.n_subjects} subjects x "
        f"{config.reps_per_gesture} reps x {config.rep_seconds}s ..."
    )
    corpus = build_corpus(config)
    print(
        f"  {corpus.n_windows} windows x {corpus.n_features} features "
        f"({time.perf_counter() - started:.1f}s)"
    )

    results = {}
    for spec in MODEL_ZOO:
        t0 = time.perf_counter()
        print(f"evaluating {spec.name} ...", flush=True)
        results[spec.name] = evaluate_model(corpus, spec, calibration_folds)
        elapsed = time.perf_counter() - t0
        summary = results[spec.name].summary()
        print(
            f"  {spec.name:<12} acc {summary['mean_accuracy']:.4f} "
            f"+/-{summary['std_accuracy']:.4f}  min {summary['min_accuracy']:.4f}  "
            f"ECE {summary['expected_calibration_error']:.4f}  ({elapsed:.1f}s)"
        )

    best = select_best(results)

    # Is the winner actually better than the others, or is the gap noise?
    significance = {}
    for name, result in results.items():
        if name == best:
            continue
        statistic, p_value = paired_fold_test(
            results[best].loso.accuracies, result.loso.accuracies
        )
        significance[f"{best}_vs_{name}"] = {
            "statistic": statistic,
            "p_value": round(p_value, 6),
            "mean_difference": round(
                results[best].loso.mean_accuracy - result.loso.mean_accuracy, 4
            ),
        }

    return {
        "corpus": {
            "n_subjects": config.n_subjects,
            "reps_per_gesture": config.reps_per_gesture,
            "rep_seconds": config.rep_seconds,
            "n_electrodes": config.n_electrodes,
            "window_ms": config.window_ms,
            "hop_ms": config.hop_ms,
            "max_fatigue": config.max_fatigue,
            "base_seed": config.base_seed,
            "n_windows": corpus.n_windows,
            "n_features": corpus.n_features,
        },
        "calibration_folds": calibration_folds,
        "models": {name: result.summary() for name, result in results.items()},
        "best_model": best,
        "paired_significance": significance,
        "source": "simulator",
        "caveat": (
            "Accuracy here is measured on simulated sEMG. The simulator has no "
            "motion artefact, no skin-impedance drift and no cross-session "
            "electrode replacement, so these figures are optimistic relative to "
            "recorded data such as NinaPro. Report them as simulator results."
        ),
        "elapsed_seconds": round(time.perf_counter() - started, 1),
    }


def _percentiles(samples: np.ndarray) -> dict[str, float]:
    return {
        "mean_ms": round(float(samples.mean()), 4),
        "p50_ms": round(float(np.percentile(samples, 50)), 4),
        "p95_ms": round(float(np.percentile(samples, 95)), 4),
        "p99_ms": round(float(np.percentile(samples, 99)), 4),
        "max_ms": round(float(samples.max()), 4),
    }


def benchmark_latency(
    onnx_path: pathlib.Path, config: CorpusConfig, n_windows: int = 1000
) -> dict:
    """End-to-end latency: raw window -> features -> ONNX inference.

    Measured one window at a time, never batched. The deployed system decodes a
    single window every 20 ms hop, so a batched throughput number would flatter
    the result and say nothing about the latency a wearer actually experiences.
    """
    sim_config = SimulatorConfig(
        sampling_rate_hz=config.sampling_rate_hz, n_electrodes=config.n_electrodes
    )
    win_config = WindowConfig(
        window_ms=config.window_ms,
        hop_ms=config.hop_ms,
        sampling_rate_hz=config.sampling_rate_hz,
    )
    feat_config = FeatureConfig(sampling_rate_hz=config.sampling_rate_hz)

    subject = make_subject("bench", seed=777)
    windows: list[np.ndarray] = []
    gesture_cycle = ["fist", "pinch", "open_hand", "point", "rest"]
    seed = 5000
    while len(windows) < n_windows:
        gesture = gesture_cycle[len(windows) % len(gesture_cycle)]
        raw = simulate(gesture, 1.0, sim_config, subject, seed)
        windows.extend(frame_windows(raw, win_config))
        seed += 1
    windows = windows[:n_windows]

    session = load_session(onnx_path)
    input_name = session.get_inputs()[0].name

    # Warm up: first calls pay one-off allocation and kernel-selection costs.
    for window in windows[:20]:
        vector = feature_vector(extract_features(window, feat_config), config.n_electrodes)
        session.run(None, {input_name: vector.reshape(1, -1)})

    feature_ms, inference_ms, total_ms = [], [], []
    for window in windows:
        start = time.perf_counter()
        vector = feature_vector(extract_features(window, feat_config), config.n_electrodes)
        mid = time.perf_counter()
        session.run(None, {input_name: vector.reshape(1, -1)})
        end = time.perf_counter()
        feature_ms.append((mid - start) * 1000)
        inference_ms.append((end - mid) * 1000)
        total_ms.append((end - start) * 1000)

    total = np.asarray(total_ms)
    return {
        "n_windows": n_windows,
        "batched": False,
        "feature_extraction": _percentiles(np.asarray(feature_ms)),
        "onnx_inference": _percentiles(np.asarray(inference_ms)),
        "end_to_end": _percentiles(total),
        "budget_ms": 10.0,
        "meets_budget": bool(np.percentile(total, 95) <= 10.0),
    }


def export_best(
    config: CorpusConfig, model_name: str, out_dir: pathlib.Path
) -> dict:
    print(f"building corpus for final fit ...")
    corpus = build_corpus(config)
    spec = model_by_name(model_name)

    print(f"fitting {spec.name} on all {corpus.n_windows} windows ...")
    model = fit_final_model(corpus, spec)

    onnx_path = out_dir / "decoder.onnx"
    export_to_onnx(model, corpus.n_features, onnx_path)
    print(f"exported {onnx_path} ({onnx_path.stat().st_size / 1024:.0f} KB)")

    parity = verify_parity(model, onnx_path, corpus.features[:2000])
    print(f"parity: ok={parity.ok} max_abs_diff={parity.max_abs_diff:.3e}")
    if not parity.ok:
        raise RuntimeError("ONNX output diverges from sklearn; refusing to ship")

    print("benchmarking latency ...")
    latency = benchmark_latency(onnx_path, config)
    print(
        f"  end-to-end p95 {latency['end_to_end']['p95_ms']:.2f} ms "
        f"(budget 10 ms) -> {'PASS' if latency['meets_budget'] else 'FAIL'}"
    )

    metadata = {
        "model": spec.name,
        "description": spec.description,
        "feature_spec_version": FEATURE_SPEC_VERSION,
        "n_features": corpus.n_features,
        "n_channels": config.n_electrodes,
        "feature_names": list(corpus.feature_names),
        "gestures": list(corpus.gestures),
        "window_ms": config.window_ms,
        "hop_ms": config.hop_ms,
        "sampling_rate_hz": config.sampling_rate_hz,
        "trained_on": "simulator",
        "onnx_parity": parity.summary(),
        "latency": latency,
        "caveat": (
            "Trained and evaluated on simulated sEMG. Not validated on recorded "
            "data. Research prototype, not a medical device."
        ),
    }
    (out_dir / "decoder.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="NeuroGrip experiments")
    parser.add_argument("experiment", choices=["compare", "export"])
    parser.add_argument(
        "--out",
        type=pathlib.Path,
        default=None,
        help="compare: output JSON path. export: output directory.",
    )
    parser.add_argument("--subjects", type=int, default=8)
    parser.add_argument("--reps", type=int, default=4)
    parser.add_argument("--seconds", type=float, default=1.0)
    parser.add_argument("--model", default="rbf_svm")
    args = parser.parse_args()

    config = CorpusConfig(
        n_subjects=args.subjects,
        reps_per_gesture=args.reps,
        rep_seconds=args.seconds,
    )

    if args.experiment == "compare":
        out = args.out or pathlib.Path("artifacts/model_comparison.json")
        if out.suffix != ".json":
            parser.error(f"compare --out must be a .json path, got {out}")
        payload = compare_models(config)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print()
        print(f"best model: {payload['best_model']}")
        print(f"written to {out}")
        return

    out_dir = args.out or pathlib.Path("artifacts")
    if out_dir.suffix:
        parser.error(f"export --out must be a directory, got {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    export_best(config, args.model, out_dir)
    print(f"written to {out_dir / 'decoder.onnx'} and {out_dir / 'decoder.json'}")


if __name__ == "__main__":
    main()
