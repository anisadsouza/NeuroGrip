# Model Card — NeuroGrip Gesture Decoder

**Research prototype. Not a medical device. Not validated for clinical use.**

Generated 2026-10-30 from recorded experiment artifacts.

## Intended use

Decodes a 200 ms window of multi-channel surface EMG into one of 10 hand-gesture classes with a calibrated confidence score, for driving a prosthetic-hand controller.

**Out of scope:** any use on a person, any clinical decision, and any deployment connected to real actuation. The simulator-trained model has not been evaluated on recorded human sEMG; the separate recorded-data development evaluation below is not clinical validation.

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
| Feature extraction | 0.89 ms | 0.86 ms | 1.21 ms | 1.40 ms | 1.99 ms |
| ONNX inference | 0.53 ms | 0.51 ms | 0.62 ms | 0.81 ms | 1.11 ms |
| **End to end** | 1.42 ms | 1.37 ms | 1.79 ms | 2.04 ms | 3.07 ms |

Budget 10 ms at P95 over 1,000 windows — **PASS**.

### Time to useful motion (TTUM)

Accuracy says whether the decoder was right. TTUM says how long the wearer pushed before the hand did anything, which is the quantity progressive actuation exists to improve and the one accuracy cannot see.

Time to Useful Motion: hops from a trial first decodable window to the first hop at which the leading gesture reaches motionOnset of its risk-weighted boundary, reported in milliseconds at the stated hop. "Correct motion" additionally requires the leader to be the trial true class. Trials that never reach the onset are censored: counted, never assigned a finite value, never dropped. No mean is reported, because under censoring a mean is either wrong or an unstated imputation. The 200 ms of window fill before the first decision is excluded and is additive: a wearer experiences roughly TTUM plus 200 ms.

Measured over 320 out-of-fold trials of 41 hops each, replayed through the deployed evidence accumulator.

| | Moved | Censored | P50 | P90 | P95 | Max |
| --- | --- | --- | --- | --- | --- | --- |
| Any motion | 320 | 0 | 40 ms | 100 ms | 100 ms | 240 ms |
| Correct motion | 318 | 2 | 40 ms | 100 ms | 140 ms | 720 ms |

Commitment latched in 309 of 320 trials at a median of 260 ms, correct in 98.7% of those; 0 fell back to rest.

Ordered by commit cost, so the risk weighting is visible as a trend rather than asserted. A costly gesture has a further boundary, so it reaches the motion onset later on the same quality of evidence.

| Gesture | Commit cost | Trials | Moved | P50 |
| --- | --- | --- | --- | --- |
| `fist` | 1 | 32 | 32 | 80 ms |
| `spherical_grip` | 1 | 32 | 32 | 60 ms |
| `pinch` | 0.6 | 32 | 32 | 80 ms |
| `two_finger` | 0.6 | 32 | 32 | 60 ms |
| `wrist_extension` | 0.3 | 32 | 32 | 40 ms |
| `wrist_flexion` | 0.3 | 32 | 32 | 60 ms |
| `point` | 0.2 | 32 | 32 | 40 ms |
| `thumb_up` | 0.2 | 32 | 32 | 40 ms |
| `open_hand` | 0.1 | 32 | 32 | 40 ms |
| `rest` | 0 | 32 | 32 | 40 ms |

Each trial is 41 hops, and the urgency timeout fires at 75, so within a trial the timeout is structurally unreachable and censoring is the only failure mode. Lengthening the repetition past about 1.5 s would activate the timeout path and change what this metric measures.

**Caveat.** Every repetition holds a constant excitation for its whole duration: the simulator applies no onset envelope. Evidence therefore accrues from an already-active steady contraction, so any time-to-motion derived from these sequences measures decoder accrual and not a wearer's reaction time. It is a lower bound on what a person would experience.

### Python-to-ONNX parity

Maximum absolute probability difference between scikit-learn and ONNX Runtime across 2,000 windows: **5.12e-07** (tolerance rtol=0.0001, atol=1e-05). Within tolerance.

## Recorded NinaPro DB2 development evaluation

This is a separate recorded-data model and must not be conflated with the simulator model above.

| Property | Value |
| --- | --- |
| Model | `lda_uniform` |
| Split | Trained on S1; tested on S10 |
| Held-out windows | 17,293 |
| Accuracy | 46.9% |
| Majority-class baseline | 36.5% |
| Expected calibration error | 0.0293 |
| Ordinary LDA reference | 45.1% accuracy; ECE 0.1644 |

**Exploratory only.** At the time this S1→S10 evaluation was generated, only S1 and S10 were available; S10 informed model diagnostics, so this is not an untouched final test. NinaPro DB2 subjects are intact participants; no amputee stratum has been evaluated.

The DB2 label mapped to `open_hand` is label 5, **abduction of all fingers**, used only as a qualified proxy rather than an exact open-hand movement.

## Additional S11 subject-held-out check

This follow-up trains on S1 and S10 and evaluates only on S11.

| Property | Value |
| --- | --- |
| Model | `lda_uniform` |
| Split | Trained on S1, S10; tested on S11 |
| Held-out windows | 20,956 |
| Accuracy | 23.1% |
| Majority-class baseline | 22.3% |
| Expected calibration error | 0.3664 |

**Poor cross-subject result.** Accuracy is only slightly above the majority baseline, and calibration is poor. S10 has informed earlier diagnostics, so this is a development check, not a fully untouched final evaluation.

## Limitations

- **Simulator-model limitation.** The simulator has no motion artefact, no skin-impedance drift, and no cross-session electrode replacement. Its figures are optimistic relative to recorded data such as NinaPro; published NinaPro leave-one-subject-out results for a ten-gesture vocabulary typically fall well below what is reported here.
- **No amputee stratum.** Every simulated subject is able-bodied by construction. The report's amputee-stratum requirement cannot be satisfied until recorded amputee data is available.
- **8 subjects.** Confidence intervals are wide and the worst-subject figure matters more than the mean.
- **Electrode-shift robustness.** The recorded result is not a human robustness study; the simulator sweep is reported separately in its artifact and does not establish robustness on people.

## Reproducing

```bash
python -m neurogrip.experiments compare --out artifacts/model_comparison.json     --posteriors artifacts/posteriors
npm run ttum
python -m neurogrip.experiments export  --out artifacts/ --model rbf_svm
python -m neurogrip.model_card --artifacts artifacts --out docs/model_card.md
```

The recorded-data section requires a separately obtained NinaPro DB2 copy; it is generated from the held-out evaluation artifact.

```bash
python -m neurogrip.heldout --data-root data/ninapro/DB2 --train-subjects 1 --test-subject 10 --model lda_uniform --out artifacts/ninapro_holdout --replay-out apps/web/public/replay/ninapro
python -m neurogrip.model_card --artifacts artifacts --out docs/model_card.md
```

