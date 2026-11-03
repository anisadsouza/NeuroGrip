# NeuroGrip technical reference

**Snapshot date:** 2026-11-04  
**System status:** Software research prototype; no deployment or physical
prosthesis control.

This document summarizes the implemented pipeline and the evidence currently
available. It deliberately separates simulator measurements, NinaPro DB2
development results, and requirements that still need new data or human
research.

## System architecture

1. **Input:** deterministic simulated sEMG or a locally staged replay from
   NinaPro DB2. Live electrodes are not connected.
2. **Worker processing:** ring buffering, windowing, signal-quality checks,
   channel-major feature extraction, ONNX inference, and 20 ms posterior
   decisions.
3. **Decision output:** the risk-weighted evidence accumulator exposes the
   leading class, commitment, reversible motion, latch, timeout, and safe
   fallback.
4. **Display:** the browser shows the signal, decision/evidence, latency, and a
   virtual hand. It sends no physical actuator command.
5. **Research workflows:** Python handles DB2 loading, model comparisons,
   held-out exports, quality audits, simulated Cartography, and exploratory
   feature attribution.

The Python and TypeScript feature implementations are a compatibility
contract. Feature set or ordering changes require coordinated version updates,
fixture regeneration, and conformance testing. See [CLAUDE.md](../CLAUDE.md).

## Recorded dataset and mapping

The local DB2 directory contains nine A1 recordings: E1-E3 for S1, S10, and
S11. These are three intact subjects, not an independent population benchmark;
each has already informed development. The dataset site describes 40 intact
DB2 subjects and requests citation.

The application has ten classes: rest plus nine DB2 movement labels. The
`open_hand` class maps to DB2 label 5, “abduction of all fingers.” It is a proxy,
not an exact general open-hand label. The full mapping is in
[ninapro.py](../services/research/neurogrip/ninapro.py).

## Recognition evidence

- The current Results candidate, uniform-prior LDA trained on S1 and tested on
  S10, reports **46.61%** on 17,293 windows, against a **36.51%** majority
  baseline; ECE is **0.026**. S10 has since informed diagnostics, so this is a
  development result.
- The separate three-subject LOSO experiment compares eight strategies. Its
  top mean accuracy is **37.46%** for empirical-prior raw LDA, against a mean
  majority baseline of **34.95%**. Results vary materially by held-out subject;
  S11 remains a severe failure case. No result establishes reliable
  cross-subject recognition.
- A new model-explanation run compares uncalibrated uniform-prior LDA with
  LinearSVC, RBF SVM, and random forest. Pooled scores are 34.80%, 29.50%,
  29.36%, and 38.04%. The random-forest result is an exploratory lead selected
  after examining the same three development subjects; it has not replaced the
  deployed decoder or been tested on an untouched subject.
- These scores use different evaluation setups and must not be presented as one
  directly comparable leaderboard. See [model_card.md](model_card.md) and
  [context.md](../context.md).

## Signal quality and S11 E2 channel 1

The window-level gate detects low channel RMS, non-finite data, extreme
amplitude/clipping patterns, repeated large sample steps, and large
half-window RMS changes. It is a heuristic—not a validated artifact
classifier. The separate rest-calibration screen issues a non-blocking warning
above an exploratory **50 µV** baseline RMS; it does not gate active windows.

In non-overlapping 200 ms rest-labeled windows from S11 E2, channel 1 exceeded
50 µV in **127/2,121 windows (6.0%)**. The median RMS is 6.31 µV, P95 is
76.87 µV, P99 is 496.28 µV, and maximum is 3,879.40 µV. Exceedances occur in
18 runs; the longest is 49 consecutive windows (9.8 seconds). The maximum
window starts at sample 1,704,000, approximately 14.2 minutes into E2. Other
channels remain much lower in those windows. The pattern is intermittent and
channel-local, but the dataset contains no artifact labels to identify its
cause. It cannot establish that the 50 µV threshold is valid.

The artificial fault probe is documented separately in
[context.md](../context.md). Artificial clipping, noise bursts, and gain jumps
are not substitutes for independently labeled real electrode faults.

## Implemented mechanisms and evidence boundaries

### Risk-weighted evidence accumulation

The accumulator integrates per-class log evidence against the strongest rival,
applies leakage and class-specific commit costs, starts reversible movement
before latch, and falls back to rest at timeout. TTUM is measured by replaying
simulator posteriors through this implementation. It has not been validated
with recorded-data posteriors, hardware, or human participants.

### Neuromotor Cartography

The research code implements simulated babble, NMF-based synergy mapping,
reachability/effort/separability estimates, prototype experiments, and a
weighted-catalogue fallback. The synthesis gate has not established that
generated prototypes outperform catalogue selection. Human babble and
teach-back are not implemented or validated.

### Error attribution

The browser attribution view demonstrates four synthetic injected cases. The
offline analysis provides LDA score-margin feature contributions and compares
three trained reference classifiers on the available subject-held-out folds.
It is not a causal explanation, not integrated into the browser router, and
not human-validated coaching.

### Clinician-oriented research views

Evaluate, Robustness, Latency, and Records are software research views.
Robustness evidence shown is simulated electrode shift; Records is study
provenance, not an electronic health record. Real-fault validation and
amputee-specific evaluation are outstanding.

## Reproducibility

Run from the repository root after installing the research package and its
dependencies:

```bash
# All Python research tests
.venv/bin/python -m pytest services/research/tests -q

# Full NinaPro DB2 comparison (may be expensive)
.venv/bin/python -m neurogrip.experiments compare \
  --source ninapro \
  --data-root data/ninapro/DB2 \
  --out artifacts/model_comparison_ninapro.json

# Three-subject recognition strategy experiment
.venv/bin/python -m neurogrip.ninapro_recognition \
  --data-root data/ninapro/DB2 \
  --out artifacts/ninapro_recognition_experiment.json

# Rest-noise percentiles and 50 µV burst analysis
.venv/bin/python -m neurogrip.ninapro_rest_calibration \
  --data-root data/ninapro/DB2 \
  --out artifacts/ninapro_rest_calibration.json

# Held-out feature attribution and reference-model comparison
.venv/bin/python -m neurogrip.ninapro_feature_attribution \
  --data-root data/ninapro/DB2 \
  --out artifacts/ninapro_feature_attribution.json

# Software checks
npm test
npm run typecheck
npm run test:conformance
npm run test:e2e
```

NinaPro-derived artifacts are git-ignored and local. The Kaggle listing
currently labels the upload CC0, but that does not verify the uploader's
authority over the original data. Do not publish derived files until the
rights chain and applicable upstream/data-use terms are verified; see
[data_and_release_review.md](data_and_release_review.md).

## Explicitly out of scope for this snapshot

- Acquiring additional licensed DB2 subjects or declaring a final untouched
  subject benchmark.
- Validating 50 µV against independent clean subjects or labeled real faults.
- Human Cartography babble/teach-back or intended-user studies.
- Integrating and validating error attributions as user coaching.
- Real-world electrode-fault robustness, amputee evaluation, hardware,
  clinical/safety validation, regulatory work, or deployment.
