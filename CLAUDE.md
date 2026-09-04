# NeuroGrip

Real-time sEMG hand-gesture decoding for prosthetic control.
Harsh Bavaskar · Anisa D'Souza · Shruti Shanklesha — BTECH CAP 501.

**Research prototype. Not a medical device. Not validated for clinical use.**

---

## What this is

A dual-mode application (prosthesis wearer + clinician) built around four novel
mechanisms. `docs/superpowers/specs/2026-09-04-neurogrip-design.md` holds the
full design and the prior-art analysis behind each claim.

1. **Neuromotor Cartography** — map a user's reachable muscle-activation
   manifold, synthesise gesture prototypes optimised for separability,
   repeatability and effort, and teach them back. *Not built yet (Phase 2).*
2. **Risk-weighted evidence accumulation** — the hand begins moving reversibly
   as evidence accrues and latches only at full commitment; high-cost gestures
   must clear a higher evidence bar than cheap ones. **Built.**
3. **Error-attribution router** — decompose each error into physiological,
   behavioural, model-capacity and drift components and route each to a
   different remedy. *Not built yet (Phase 3).*
4. **Anatomical attribution** — back-project feature attributions onto forearm
   anatomy and convert them into plain-language coaching. *Not built yet
   (Phase 3).*

**How to use the application: `docs/walkthrough.md`.**

---

## Status

| Phase | Scope | State |
| --- | --- | --- |
| 0 | Simulator, feature pipeline in both languages, conformance gate, CI | Complete |
| 1a | Corpus, LOSO evaluation, model comparison, calibration, ONNX export, latency | Complete |
| 1b | Evidence accumulator, design system, browser app, Live screen | Vertical slice running |
| 2 | Neuromotor Cartography | Not started |
| 3 | Error-attribution router, anatomical coaching | Not started |
| 4 | Clinician mode, amputee stratum, robustness | Not started |
| 5 | Azure deployment, patent disclosures | Not started |

**248 tests** — 126 Python, 122 TypeScript.

Not yet built: the virtual hand, Cartography, the attribution router, clinician
mode, and deployment.

---

## Layout

| Path | Holds |
| --- | --- |
| `services/research/` | Python: simulator, features, training, evaluation, export |
| `packages/core/` | TypeScript: the same DSP and features, plus the evidence accumulator |
| `packages/design/` | Design tokens and the icon set |
| `apps/web/` | The browser application |
| `fixtures/conformance/` | Golden vectors pinning the two feature implementations together |
| `fixtures/simulator_signature.json` | Pins simulator behaviour against silent drift |
| `artifacts/` | Trained model, experiment results, latency measurements |
| `docs/` | Walkthrough, model card, design reference, specs and plans |

---

## The rule that matters most

`services/research/neurogrip/features.py` and `packages/core/src/features.ts`
must compute identical values. Python trains the model; TypeScript feeds it at
inference. If they diverge, accuracy degrades silently with no error anywhere.

`packages/core/test/conformance.test.ts` enforces this against committed golden
vectors. **After changing anything in either feature pipeline:**

```bash
npm run fixtures          # regenerate golden vectors from Python
npm run test:conformance  # prove TypeScript still agrees
```

Bump `FEATURE_SPEC_VERSION` in **both** `services/research/neurogrip/__init__.py`
and `packages/core/src/spec.ts` whenever the feature set or its ordering changes.

Never widen the conformance tolerance to make a failure pass. The tolerance is
the contract.

### Debugging a conformance failure

The failure message names the offending feature and prints both values. Check in
this order — these are the divergences this port is actually likely to hit:

1. `*_psd*`, `*_mdf`, `*_mnf` — `nextPow2` padding, the Hann window (symmetric,
   not periodic), the `1 .. bins-2` doubling range, `array_split` band sizing.
2. `*_ar*` — the Levinson-Durbin reversal index. Python's `previous[::-1]`
   corresponds to `previous[m - 1 - i]` in TypeScript. Getting this backwards is
   the classic port bug and it only shows at order 3 or above.
3. `*_zc`, `*_ssc` — both sides must use the strict product tests, not sign
   comparison.
4. `*_rms`, `*_mav`, `*_wl` — a mismatch means float32 versus float64
   accumulation. Python casts input to float32 in `_as_channel_matrix`, then
   computes in float64.

### The second gate

The conformance fixtures store their own input waveforms, so they are blind to
simulator changes. `fixtures/simulator_signature.json` closes that gap: it pins
per-channel RMS per gesture at a fixed seed and catches a 0.01 change to a
single muscle's excitation.

```bash
python -m neurogrip.signature --out fixtures/simulator_signature.json
```

If that test fails and the change was intended, regenerate and **review the
diff** — every accuracy number in the repo was produced by the old simulator.

---

## The feature set

Seventeen features per channel, in this fixed order:

`rms · mav · wl · zc · ssc · ar1-ar4 · psd1-psd6 · mdf · mnf`

Ordering is **channel-major and never sorted**. `sorted()` places `ch10_rms`
before `ch2_rms`; NinaPro DB2 has twelve channels, so lexicographic ordering
would silently permute the vector between an 8-channel and a 12-channel run.

---

## The simulator

`services/research/neurogrip/simulator.py` is a Fuglevand-style motor-unit pool
model, not shaped noise. Each muscle holds motor units with exponentially
distributed recruitment thresholds; units above threshold fire at
excitation-dependent rates; each firing convolves a Hermite-Rodriguez MUAP whose
duration scales with unit size. Muscle sources mix into electrode channels
through the volume-conduction matrix in `anatomy.py`.

Fatigue widens MUAPs, which compresses the spectrum and lowers median frequency.
That emerges from the mechanism rather than being hard-coded, which is what makes
it usable as a test bed for the drift and fatigue estimators.

Output is calibrated to real voltages: about 15 µV at rest and 158–354 µV under
contraction, matching published surface-EMG ranges.

**Gestures must differ in pattern, never by a uniform amplitude offset.**
Per-subject gain spans 0.7–1.4×, so two classes separated only by overall level
are mathematically indistinguishable under LOSO. `point` and `two_finger` once
differed by a uniform 0.05 and confused at 32% in both directions.
`test_no_two_gestures_are_gain_degenerate` enforces pairwise cosine < 0.98.

Everything is seeded. Nothing in the simulator may touch global RNG state.

**This is why no dataset download is needed.** NinaPro registration is optional,
not blocking.

---

## The decoder

Selected on evidence, not on the architecture named in the project report.

| Model | LOSO accuracy | SD | Worst subject | ECE |
| --- | --- | --- | --- | --- |
| **rbf_svm (selected)** | **95.63%** | 0.056 | 83.8% | **0.022** |
| linear_svm | 90.48% | 0.109 | 71.4% | 0.070 |
| lda | 90.45% | 0.100 | 71.9% | 0.263 |

RBF SVM beats both significantly (Wilcoxon p = 0.008 vs LDA, p = 0.016 vs linear
SVM). Calibration is the decisive margin: LDA's ECE of 0.263 fails the project's
NFR-5 requirement of ≤ 0.10 by more than 2.5×, and a decoder that abstains below
a confidence threshold is unusable if its confidences mean nothing.

**Calibration uses `ensemble=False`.** The ensemble form fits one classifier per
calibration fold and averages them, which for an RBF SVM multiplies the stored
support vectors by the fold count. Measured on the full corpus the single form
was better on every axis at once — accuracy, ECE, 2.4× smaller, 2.3× faster —
so there is no trade-off to weigh.

Latency, measured one window at a time on CPU, never batched:

| Stage | P95 |
| --- | --- |
| Feature extraction | 1.29 ms |
| ONNX inference | 0.96 ms |
| **End to end** | **2.22 ms** (budget 10 ms) |

These come from the most recent `npm run build:assets` and are re-measured every
time it runs, so they move by a few tenths of a millisecond between runs on a
busy machine. The authoritative copy is always `artifacts/decoder.json`; the
table above is a snapshot of it. Observed range across runs: 1.9-2.2 ms.

In the browser the same path measured 2.4-3.2 ms P95 across several runs with
WASM threads enabled. The spread is real: it is a shared machine, and the Live
screen reports what it actually measured rather than a best case.

**These figures come from simulated data.** The simulator has no motion
artefact, no skin-impedance drift, and no cross-session electrode replacement.
Published NinaPro LOSO results for a ten-gesture vocabulary sit well below this.
Report them as simulator results, and expect a drop on recorded data.

---

## Risk-weighted evidence accumulation

`packages/core/src/evidence.ts`. A leaky competing race: each gesture accrues
the log ratio of its probability to its strongest rival, decays toward zero, and
is floored at zero so a class that fell behind recovers promptly.

Two properties distinguish it from a confidence threshold:

- Each gesture carries its own boundary, `θ_k = base + weight × risk_k`. A
  closing power grip must clear a higher bar than an opening hand, because the
  mistakes are not equally recoverable.
- The actuator command is continuous. Motion begins at a fraction of the
  boundary and retracts if the evidence turns.

Measured behaviour at a 20 ms hop:

| Confidence | Gesture | Risk | Motion | Latch | Reversible window |
| --- | --- | --- | --- | --- | --- |
| 0.95 | fist | 1.0 | 40 ms | 180 ms | 140 ms |
| 0.95 | open_hand | 0.1 | 20 ms | 120 ms | 100 ms |
| 0.60 | fist | 1.0 | 60 ms | 460 ms | 400 ms |
| 0.40 | fist | 1.0 | 100 ms | 740 ms | 640 ms |

Boundaries collapse with elapsed time so the system cannot hang. If nothing
commits before `maxHops`, it falls back to `rest` — a refusal to guess, because
an unintended grip is a physical event and an unintended rest is not.

`baseThreshold` is 24. An earlier value of 6 latched in two hops, leaving no
reversible window and making "progressive" actuation indistinguishable from a
threshold.

Commit costs live in `apps/web/src/screens/Live.tsx` (`COMMIT_COST`). They are a
safety judgement about how hard each mistake is to undo, not a tuning parameter.

---

## Design system

`packages/design/`. Two rules, both enforced by tests:

1. **The graticule means measurement.** A grid appears only behind data with a
   real scale — signal lanes, the commitment bar. Never as chrome.
2. **Carmine means muscle.** The one chromatic hue encodes activation intensity,
   following anatomical illustration convention. It is never used to emphasise a
   heading or a control.

Type is Atkinson Hyperlegible for interface text — designed by the Braille
Institute for maximum character distinction at low vision, which matters because
the intended users are people with disabilities — and IBM Plex Mono for numerals
only, where tabular alignment is functional.

No gradients, no glows, no shadows beyond a hairline. Corners are rounded on a
scale: `sm 3px` for small marks, `6px` default, `lg 10px` for outermost
surfaces. A nested surface always takes a smaller radius than its container.

`packages/design/test/contrast.test.ts` parses the real token file and computes
WCAG ratios for every foreground/background pair in both themes. A palette
change that breaks legibility fails the build. It also asserts the two dark
declarations (media query and `[data-theme]`) stay identical, because CSS cannot
share them and duplication drifts.

Icons are constructed on a 24-pixel grid at 1.5 stroke with butt caps and miter
joins. Every icon must carry a note justifying its form.
`packages/design/test/icons.test.ts` enforces the grid, rejects rounded caps and
hard-coded colour, and holds fill to an allowlist with a reason per entry.

Regenerate the visual reference sheet with:

```bash
npx vite-node packages/design/scripts/reference.ts   # -> docs/design/reference.html
```

---

## The browser application

`apps/web/`. Everything expensive runs in `src/worker/inference.worker.ts`:
ring buffer, windowing, signal-quality gate, feature extraction, ONNX inference,
and evidence accumulation. The UI thread receives a small decision every 20 ms
and does nothing but draw.

The signal source is `src/sources/replaySource.ts`, which streams a recorded
bundle of **simulated** sEMG at the sampling rate. The simulator is Python and
cannot run in the browser; porting it would mean a second implementation owing a
second conformance gate, for a component that only produces test input. When
real electrodes arrive they replace this source and nothing downstream changes.

**On privacy.** The project requirement (ER-3 / NFR-6) is that raw sEMG is never
persisted and never leaves the device. Samples *do* cross the worker→UI thread
boundary, because the oscilloscope has to draw the wearer's own signal on their
own screen. Nothing is written to storage or sent over the network anywhere in
this application.

The dev server sets `Cross-Origin-Opener-Policy` and
`Cross-Origin-Embedder-Policy`. Without them `SharedArrayBuffer` is unavailable,
ONNX Runtime silently falls back to single-threaded WASM, and inference roughly
doubles. The Live screen reports which one it got rather than publishing an
optimistic number.

---

## Commands

```bash
# Setup
pip install -e "./services/research[dev,ml]"
npm install

# Tests
python -m pytest services/research/tests -q
npm test
npm run typecheck

# The gates
npm run test:conformance     # Python vs TypeScript features
npm run fixtures             # regenerate golden vectors

# Run the app
npm run dev:web              # http://localhost:5173

# Regenerate model + replay assets the app consumes
npm run build:assets

# Experiments (minutes, not milliseconds)
python -m neurogrip.experiments compare --out artifacts/model_comparison.json
python -m neurogrip.experiments export  --out artifacts --model rbf_svm
python -m neurogrip.model_card --artifacts artifacts --out docs/model_card.md
```

---

## Deviations from the project report, and why

- **20 ms hop, not 100 ms.** Progressive actuation needs evidence between
  decisions; at a 10 Hz decision rate there is none. Costs 5× the inference
  calls, each under 2 ms, so the 10 ms P95 budget still holds.
- **No `scipy.signal.welch`.** Its detrending, segment averaging and scaling
  conventions are a porting hazard for the browser runtime and buy nothing at a
  single 200 ms window. The periodogram is hand-implemented in both languages.
- **Feature ordering is channel-major, never sorted.** See above.
- **Two thresholds recalibrated to real voltages.** Baseline noise moved from
  300 µV to 15 µV (the old value buried the motor-unit signal), and the
  electrode-detachment gate from 100 µV to 5 µV so it sits below baseline noise
  rather than above it — at the old value a genuine rest window was rejected as
  a detached electrode.
- **Model chosen by measurement, not by plan.** The report names an RBF SVM. It
  won, but LDA and a linear SVM were evaluated first and the comparison is
  recorded in `artifacts/model_comparison.json`.
- **A TCN was not built.** The report anticipates an LSTM that may fail the
  latency gate. The handcrafted-feature SVM meets it with 5× margin, so a
  sequence model has not been needed yet.

---

## Working conventions

- Tests first. Every gate in this repo was observed failing before it was
  trusted — the conformance gate, the signature gate, the contrast tests.
- Never weaken an assertion to make a failure pass. Fix the thing, or fix the
  test's premise and say which.
- Every number in the docs traces to a JSON artifact written by a runnable
  command. `docs/model_card.md` is generated, never hand-edited.
- Nothing here has been evaluated on a human recording. Say so wherever a number
  appears.
