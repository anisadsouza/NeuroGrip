# Model Card — NeuroGrip Gesture Decoder

**Research prototype. Not a medical device. Not validated for clinical use.**

Generated 2026-09-04 from recorded experiment artifacts.

## Intended use

Decodes a 200 ms window of multi-channel surface EMG into one of 10 hand-gesture classes with a calibrated confidence score, for driving a prosthetic-hand controller.

**Out of scope:** any use on a person, any clinical decision, and any deployment connected to real actuation. The decoder has never been evaluated on recorded human sEMG.

## Training data

| Property | Value |
| --- | --- |
| Source | Simulated sEMG (`neurogrip.simulator`) |
| Subjects | 8 |
| Repetitions per gesture | 4 |
| Windows | 13,120 |
| Features per window | 136 |
| Window / hop | 200 ms / 20 ms |
| Channels | 8 |
| Sampling rate | 2000 Hz |
| Peak simulated fatigue | 0.5 |
| Feature spec version | 1 |

## Model selection

Three families were compared under leave-one-subject-out. Selection was made on evidence, not on the architecture named in the project plan.

| Model | LOSO accuracy | SD | Worst subject | ECE |
| --- | --- | --- | --- | --- |
| `lda` | 90.5% | 0.100 | 71.9% | 0.263 |
| `linear_svm` | 90.5% | 0.109 | 71.4% | 0.070 |
| `rbf_svm` **(selected)** | 95.6% | 0.056 | 83.8% | 0.022 |

Majority-class baseline: 10.0%.

### Is the winner actually better?

Wilcoxon signed-rank over paired per-fold accuracies.

| Comparison | Mean difference | p |
| --- | --- | --- |
| rbf_svm vs lda | +0.0518 | 0.007812 |
| rbf_svm vs linear_svm | +0.0514 | 0.015625 |

## Performance

Leave-one-subject-out accuracy **95.6%** (95% CI 90.6%–100.0%), worst subject 83.8%.

### Per-gesture recall

| Gesture | Recall |
| --- | --- |
| `rest` | 100.0% |
| `fist` | 88.9% |
| `open_hand` | 100.0% |
| `pinch` | 93.5% |
| `point` | 96.8% |
| `wrist_flexion` | 99.6% |
| `wrist_extension` | 98.6% |
| `thumb_up` | 100.0% |
| `two_finger` | 86.0% |
| `spherical_grip` | 92.8% |

### Latency

Measured one window at a time on CPU, never batched: the deployed system decodes a single window every hop, so a batched throughput figure would say nothing about experienced latency.

| Stage | Mean | P50 | P95 | P99 | Max |
| --- | --- | --- | --- | --- | --- |
| Feature extraction | 0.96 ms | 0.90 ms | 1.29 ms | 1.74 ms | 2.68 ms |
| ONNX inference | 0.65 ms | 0.61 ms | 0.96 ms | 1.20 ms | 1.99 ms |
| **End to end** | 1.61 ms | 1.52 ms | 2.22 ms | 2.86 ms | 4.16 ms |

Budget 10 ms at P95 over 1,000 windows — **PASS**.

### Python-to-ONNX parity

Maximum absolute probability difference between scikit-learn and ONNX Runtime across 2,000 windows: **5.13e-07** (tolerance rtol=0.0001, atol=1e-05). Within tolerance.

## Limitations

- **Trained on simulated data.** The simulator has no motion artefact, no skin-impedance drift, and no cross-session electrode replacement. These figures are optimistic relative to recorded data such as NinaPro; published NinaPro leave-one-subject-out results for a ten-gesture vocabulary typically fall well below what is reported here.
- **No amputee stratum.** Every simulated subject is able-bodied by construction. The report's amputee-stratum requirement cannot be satisfied until recorded amputee data is available.
- **8 subjects.** Confidence intervals are wide and the worst-subject figure matters more than the mean.
- **No electrode-shift robustness curve yet.** The corpus builder supports a shift override; the sweep itself is not yet run.

## Reproducing

```bash
python -m neurogrip.experiments compare --out artifacts/model_comparison.json
python -m neurogrip.experiments export  --out artifacts/ --model rbf_svm
python -m neurogrip.model_card --artifacts artifacts --out docs/model_card.md
```

