# NeuroGrip

**Software-only surface-EMG gesture decoding for prosthetic-control research.**

NeuroGrip is a browser-based research prototype that converts simulated or
recorded forearm surface EMG into a gesture prediction and shows the predicted
action on a virtual prosthetic hand. It does not connect to live electrodes,
send actuator commands, or control physical hardware.

> **Research prototype — not a medical device. Not validated for clinical use.**

**Team:** Harsh Bavaskar · Anisa D'souza · Shruti Shanklesha


## What it demonstrates

- Eight-channel, 200 ms surface-EMG window processing at a 20 ms hop
- 136 handcrafted features per window, shared between Python and TypeScript
- A calibrated RBF-SVM decoder for simulated sEMG
- Risk-weighted evidence accumulation and reversible virtual-hand motion
- Subject-held-out NinaPro DB2 replay and exploratory recorded-data evaluation
- Signal-quality checks, Results, Cartography, Attribution and research-review screens
- A virtual hand with no transmitted physical actuator command

## Current evidence

### Simulator

The simulator uses a deterministic motor-unit pool model rather than shaped
noise. Eight simulated subjects were used for the selected-model experiment.
The RBF SVM achieved **95.63% LOSO accuracy** with **ECE 0.0225**. LDA and
Linear SVM were 90.45% and 90.48%, respectively.

These values are simulator-only measurements. The simulator does not include
real motion artifacts, skin-impedance drift or cross-session electrode replacement.

### Recorded NinaPro DB2

The local development dataset contains three intact subjects: S1, S10 and S11,
with nine E1–E3 `.mat` recordings. It is not an independent population or
amputee benchmark.

| Experiment | Accuracy | Majority baseline | ECE |
|---|---:|---:|---:|
| Uniform-prior LDA: train S1, test S10 | 46.61% | 36.51% | 0.026 |
| Ordinary-LDA S1 → S10 reference | 45.09% | 36.51% | 0.1644 |
| Uniform-prior LDA: train S1 + S10, test S11 | 23.07% | 22.26% | 0.3664 |

A bounded three-subject leave-one-subject-out comparison found no strategy that
reliably justified changing the decoder. S11 predicted Rest for 19,125 of
20,956 windows, and six gesture classes had zero recall. These figures are
exploratory development results, not a final cross-subject benchmark or evidence
of safe prosthetic control.

The DB2 label mapped to `open_hand` is **finger abduction (label 5)**. It is a
qualitative proxy for open hand, not an exact general open-hand movement.

### Browser latency

The browser worker test measures 1,000 windows one at a time and requires
threaded WASM and cross-origin isolation. The latest measured simulated replay
P95 was **1.905 ms**. The recorded S10 replay measured **1.185 ms** P95.
Both are below the 10 ms benchmark and exclude reaction time, postMessage and
UI-thread scheduling.

## System overview

```text
Simulated or recorded sEMG
           ↓
200 ms window / 20 ms hop
           ↓
Signal-quality check
           ↓
136 channel-major features
           ↓
ONNX classifier in browser worker
           ↓
Risk-weighted evidence accumulator
           ↓
Virtual-hand preview / latch / rest fallback
```

- **Python**: simulator, NinaPro loader, feature extraction, model evaluation,
  ONNX export, reports and experiments.
- **TypeScript/core**: mirrored feature implementation, evidence accumulation,
  TTUM/CWER and hand kinematics.
- **React/Vite browser app**: Live, Results, Cartography, Attribution and
  clinician-oriented research views.
- **ONNX Runtime Web**: threaded browser inference.

No raw EMG, credentials or prediction history are uploaded by the application.
Local NinaPro files and generated replay/model assets are git-ignored.

## Quick start

Requirements:

- Python 3.12+
- Node 24+
- npm

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e "./services/research[dev,ml]"
npm install
npm run dev:web
```

The default application uses the included simulator and requires no NinaPro
download or account. Open the local address printed by Vite, typically
`http://localhost:5173`.

To regenerate the simulator model and replay assets:

```bash
npm run build:assets
```

## Documentation

- [Using NeuroGrip](docs/walkthrough.md) — interface and demo workflow
- [Technical reference](docs/technical_reference.md) — architecture, methods and limitations
- [Model card](docs/model_card.md) — model, measurements and evidence boundaries
- [Data and release review](docs/data_and_release_review.md) — NinaPro/Kaggle provenance status
- [Repository status and remaining work](context.md) — implementation and research summary

## Repository layout

| Path | Purpose |
|---|---|
| `services/research/neurogrip/` | Simulator, NinaPro loader, DSP, features, training and experiments |
| `packages/core/` | Mirrored DSP/features, evidence accumulator, TTUM/CWER and hand kinematics |
| `packages/design/` | Design tokens and icon system |
| `apps/web/` | React browser application, worker and browser tests |
| `fixtures/conformance/` | Python/TypeScript golden feature vectors |
| `artifacts/` | Reproducible experiment and model artifacts |
| `docs/` | Technical, data-review and walkthrough documentation |

## Testing

Python tests:

```bash
source .venv/bin/activate
python -m pytest services/research/tests -q
```

TypeScript, DOM and unit tests:

```bash
npm test
npm run typecheck
```

Python/TypeScript feature conformance:

```bash
npm run fixtures
npm run test:conformance
```

Production-browser tests:

```bash
npm run test:e2e
```

GitHub Actions runs Python tests, TypeScript/type checks, fixture freshness,
and Playwright browser tests. See [playwright.config.ts](playwright.config.ts)
for the browser configuration.

## Limitations

- Recorded NinaPro recognition is weak across subjects and is not a clinical
  or amputee benchmark.
- The quality gate is a heuristic, with only controlled/artificial fault probes.
- No real electrode, hardware actuator or physical prosthesis is connected.
- Cartography teach-back, error attribution/coaching and robustness figures are
  simulated or exploratory unless explicitly marked otherwise.
- Real-world motion artifacts and amputee validation remain unperformed.
- NinaPro-derived data and models must remain private until redistribution rights
  are confirmed.

## NinaPro citation

NeuroGrip's recorded-data experiments use NinaPro DB2:

- Atzori et al., “Electromyography data for non-invasive naturally-controlled
  robotic hand prostheses,” *Scientific Data* 1, 140053 (2014).
  [doi:10.1038/sdata.2014.53](https://doi.org/10.1038/sdata.2014.53)
- Atzori et al., “Characterization of a benchmark database for myoelectric
  movement classification,” *IEEE Transactions on Neural Systems and
  Rehabilitation Engineering* 23(1), 73–83 (2015).
  [doi:10.1109/TNSRE.2014.2328495](https://doi.org/10.1109/TNSRE.2014.2328495)

The Kaggle listing was used by the project owner, but the Kaggle CC0 label alone
does not establish authority to redistribute NinaPro data or derived assets.
