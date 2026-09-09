"""Model card generation.

Renders a markdown model card from recorded experiment artifacts rather than
from prose, so the numbers in the card cannot drift away from the numbers the
experiments actually produced. Every figure it prints is traceable to a JSON
file written by `neurogrip.experiments`.

    python -m neurogrip.model_card --artifacts artifacts --out docs/model_card.md
"""

from __future__ import annotations

import argparse
import json
import pathlib
from datetime import date


def _fmt_ms(value: float | None) -> str:
    """Milliseconds, or an em dash where a trial never got there.

    Never a zero: a censored measurement and an instantaneous one are opposite
    findings, and printing both as 0 ms would merge them.
    """
    return "—" if value is None else f"{value:.0f} ms"


def _fmt_pct(value: float | None) -> str:
    return "-" if value is None else f"{value * 100:.1f}%"


def render(
    comparison: dict,
    decoder: dict,
    generated_on: str,
    ttum: dict | None = None,
) -> str:
    best = comparison["best_model"]
    models = comparison["models"]
    chosen = models[best]
    corpus = comparison["corpus"]
    latency = decoder["latency"]

    lines: list[str] = []
    add = lines.append

    add("# Model Card — NeuroGrip Gesture Decoder")
    add("")
    add("**Research prototype. Not a medical device. Not validated for clinical use.**")
    add("")
    add(f"Generated {generated_on} from recorded experiment artifacts.")
    add("")

    add("## Intended use")
    add("")
    add(
        "Decodes a 200 ms window of multi-channel surface EMG into one of "
        f"{len(decoder['gestures'])} hand-gesture classes with a calibrated "
        "confidence score, for driving a prosthetic-hand controller."
    )
    add("")
    add(
        "**Out of scope:** any use on a person, any clinical decision, and any "
        "deployment connected to real actuation. The decoder has never been "
        "evaluated on recorded human sEMG."
    )
    add("")

    add("## Training data")
    add("")
    add("| Property | Value |")
    add("| --- | --- |")
    add(f"| Source | Simulated sEMG (`neurogrip.simulator`) |")
    add(f"| Subjects | {corpus['n_subjects']} |")
    add(f"| Repetitions per gesture | {corpus['reps_per_gesture']} |")
    add(f"| Windows | {corpus['n_windows']:,} |")
    add(f"| Features per window | {corpus['n_features']} |")
    add(f"| Window / hop | {corpus['window_ms']} ms / {corpus['hop_ms']} ms |")
    add(f"| Channels | {decoder['n_channels']} |")
    add(f"| Sampling rate | {decoder['sampling_rate_hz']} Hz |")
    add(f"| Peak simulated fatigue | {corpus['max_fatigue']} |")
    add(f"| Feature spec version | {decoder['feature_spec_version']} |")
    add("")

    add("## Model selection")
    add("")
    add(
        "Three families were compared under leave-one-subject-out. Selection was "
        "made on evidence, not on the architecture named in the project plan."
    )
    add("")
    add("| Model | LOSO accuracy | SD | Worst subject | ECE |")
    add("| --- | --- | --- | --- | --- |")
    for name, result in models.items():
        marker = " **(selected)**" if name == best else ""
        add(
            f"| `{name}`{marker} | {_fmt_pct(result['mean_accuracy'])} | "
            f"{result['std_accuracy']:.3f} | {_fmt_pct(result['min_accuracy'])} | "
            f"{result['expected_calibration_error']:.3f} |"
        )
    add("")
    add(f"Majority-class baseline: {_fmt_pct(chosen['baseline_accuracy'])}.")
    add("")

    if comparison.get("paired_significance"):
        add("### Is the winner actually better?")
        add("")
        add("Wilcoxon signed-rank over paired per-fold accuracies.")
        add("")
        add("| Comparison | Mean difference | p |")
        add("| --- | --- | --- |")
        for key, value in comparison["paired_significance"].items():
            add(
                f"| {key.replace('_vs_', ' vs ')} | "
                f"{value['mean_difference']:+.4f} | {value['p_value']} |"
            )
        add("")

    add("## Performance")
    add("")
    add(
        f"Leave-one-subject-out accuracy **{_fmt_pct(chosen['mean_accuracy'])}** "
        f"(95% CI {_fmt_pct(chosen['ci95_low'])}–{_fmt_pct(chosen['ci95_high'])}), "
        f"worst subject {_fmt_pct(chosen['min_accuracy'])}."
    )
    add("")
    add("### Per-gesture recall")
    add("")
    add("| Gesture | Recall |")
    add("| --- | --- |")
    for gesture, recall in chosen["per_gesture_recall"].items():
        add(f"| `{gesture}` | {_fmt_pct(recall)} |")
    add("")

    add("### Latency")
    add("")
    add(
        "Measured one window at a time on CPU, never batched: the deployed "
        "system decodes a single window every hop, so a batched throughput "
        "figure would say nothing about experienced latency."
    )
    add("")
    add("| Stage | Mean | P50 | P95 | P99 | Max |")
    add("| --- | --- | --- | --- | --- | --- |")
    for label, key in (
        ("Feature extraction", "feature_extraction"),
        ("ONNX inference", "onnx_inference"),
        ("**End to end**", "end_to_end"),
    ):
        stage = latency[key]
        add(
            f"| {label} | {stage['mean_ms']:.2f} ms | {stage['p50_ms']:.2f} ms | "
            f"{stage['p95_ms']:.2f} ms | {stage['p99_ms']:.2f} ms | "
            f"{stage['max_ms']:.2f} ms |"
        )
    add("")
    verdict = "PASS" if latency["meets_budget"] else "FAIL"
    add(
        f"Budget {latency['budget_ms']:.0f} ms at P95 over "
        f"{latency['n_windows']:,} windows — **{verdict}**."
    )
    add("")

    if ttum is not None:
        add("### Time to useful motion (TTUM)")
        add("")
        add(
            "Accuracy says whether the decoder was right. TTUM says how long the "
            "wearer pushed before the hand did anything, which is the quantity "
            "progressive actuation exists to improve and the one accuracy cannot "
            "see."
        )
        add("")
        add(ttum["definition"])
        add("")
        add(
            f"Measured over {ttum['nTrials']:,} out-of-fold trials of "
            f"{ttum['hopsPerTrial']} hops each, replayed through the deployed "
            f"evidence accumulator."
        )
        add("")
        add("| | Moved | Censored | P50 | P90 | P95 | Max |")
        add("| --- | --- | --- | --- | --- | --- | --- |")
        for label, key in (("Any motion", "anyMotion"), ("Correct motion", "correctMotion")):
            block = ttum[key]
            add(
                f"| {label} | {block['nMoved']} | {block['nCensored']} | "
                f"{_fmt_ms(block['p50Ms'])} | {_fmt_ms(block['p90Ms'])} | "
                f"{_fmt_ms(block['p95Ms'])} | {_fmt_ms(block['maxMs'])} |"
            )
        add("")
        latch = ttum["latch"]
        add(
            f"Commitment latched in {latch['nLatched']} of {ttum['nTrials']} trials "
            f"at a median of {_fmt_ms(latch['p50Ms'])}, correct in "
            f"{_fmt_pct(latch['accuracyAtLatch'])} of those; "
            f"{latch['nTimedOut']} fell back to rest."
        )
        add("")
        add(
            "Ordered by commit cost, so the risk weighting is visible as a trend "
            "rather than asserted. A costly gesture has a further boundary, so it "
            "reaches the motion onset later on the same quality of evidence."
        )
        add("")
        add("| Gesture | Commit cost | Trials | Moved | P50 |")
        add("| --- | --- | --- | --- | --- |")
        for name, block in sorted(
            ttum["perGesture"].items(), key=lambda kv: (-kv[1]["risk"], kv[0])
        ):
            add(
                f"| `{name}` | {block['risk']} | {block['nTrials']} | "
                f"{block['nMoved']} | {_fmt_ms(block['p50Ms'])} |"
            )
        add("")
        add(ttum["reachability"])
        add("")
        add(f"**Caveat.** {ttum['caveat']}")
        add("")

    add("### Python-to-ONNX parity")
    add("")
    parity = decoder["onnx_parity"]
    add(
        f"Maximum absolute probability difference between scikit-learn and ONNX "
        f"Runtime across {parity['n_samples']:,} windows: "
        f"**{parity['max_abs_diff']:.2e}** "
        f"(tolerance rtol={parity['rtol']}, atol={parity['atol']}). "
        f"{'Within tolerance.' if parity['ok'] else 'OUT OF TOLERANCE.'}"
    )
    add("")

    add("## Limitations")
    add("")
    add(
        "- **Trained on simulated data.** The simulator has no motion artefact, "
        "no skin-impedance drift, and no cross-session electrode replacement. "
        "These figures are optimistic relative to recorded data such as NinaPro; "
        "published NinaPro leave-one-subject-out results for a ten-gesture "
        "vocabulary typically fall well below what is reported here."
    )
    add(
        "- **No amputee stratum.** Every simulated subject is able-bodied by "
        "construction. The report's amputee-stratum requirement cannot be "
        "satisfied until recorded amputee data is available."
    )
    add(
        f"- **{corpus['n_subjects']} subjects.** Confidence intervals are wide and "
        "the worst-subject figure matters more than the mean."
    )
    add(
        "- **No electrode-shift robustness curve yet.** The corpus builder "
        "supports a shift override; the sweep itself is not yet run."
    )
    add("")

    add("## Reproducing")
    add("")
    add("```bash")
    add(
        "python -m neurogrip.experiments compare --out artifacts/model_comparison.json"
        + (" \
    --posteriors artifacts/posteriors" if ttum is not None else "")
    )
    if ttum is not None:
        add("npm run ttum")
    add("python -m neurogrip.experiments export  --out artifacts/ --model " + best)
    add("python -m neurogrip.model_card --artifacts artifacts --out docs/model_card.md")
    add("```")
    add("")

    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Render the model card")
    parser.add_argument("--artifacts", type=pathlib.Path, default=pathlib.Path("artifacts"))
    parser.add_argument("--out", type=pathlib.Path, default=pathlib.Path("docs/model_card.md"))
    args = parser.parse_args()

    comparison = json.loads(
        (args.artifacts / "model_comparison.json").read_text(encoding="utf-8")
    )
    decoder = json.loads((args.artifacts / "decoder.json").read_text(encoding="utf-8"))

    # Optional: the card still renders without a TTUM run, because the run
    # needs the Node toolchain and the card must not require it.
    ttum_path = args.artifacts / "ttum.json"
    ttum = json.loads(ttum_path.read_text(encoding="utf-8")) if ttum_path.exists() else None

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        render(comparison, decoder, date.today().isoformat(), ttum), encoding="utf-8"
    )
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
