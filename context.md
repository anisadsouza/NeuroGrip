# NeuroGrip: current state and next steps

Updated 2026-10-28 after the NinaPro DB2 held-out workflow, Results screen,
mobile layout, dataset audit, and final browser/regression checks. Read
`CLAUDE.md` for the project's engineering and safety rules.

**Research prototype. Not a medical device. Not validated for clinical use.**
The project now has a real-recording evaluation path, but its only recorded
comparison uses two intact subjects (S1 and S10). The S1-to-S10 result is
exploratory and must not be presented as population-level or clinical evidence.
Simulator results and recorded results remain separate.

---

## 1. The project in one paragraph

NeuroGrip decodes eight channels of forearm surface EMG (sEMG) into a ten-class
gesture vocabulary. Python (`services/research/`) simulates data, ingests
NinaPro DB2 recordings, extracts features, evaluates models, and exports ONNX.
The browser (`apps/web/`) runs features and inference in a worker every 20 ms.
Live can replay either simulated signal or a locally generated held-out DB2
recording, with source/model provenance. Results keeps simulated comparisons,
recorded comparisons, held-out confusion/timeline, and browser/CPU latency
separate. No live electrode input is implemented.

## 2. Setup

```bash
python -m venv .venv
source .venv/bin/activate              # macOS/Linux
# Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -e "./services/research[dev,ml]"
npm install

python -m pytest services/research/tests -q   # Python tests
npm test                                       # TypeScript tests
npm run dev:web                                # app at http://localhost:5173
```

## 3. Rules that must not be broken

These are the ones an AI assistant is most likely to break. The full list is in
`CLAUDE.md`.

1. **Features must match in both languages.** `services/research/neurogrip/features.py`
   (training) and `packages/core/src/features.ts` (browser) must compute
   identical values. After touching either, run `npm run fixtures` and then
   `npm run test:conformance`. Never loosen the tolerance to make it pass.
2. **Feature order is channel-major and never sorted.** It runs `ch1_rms, ch1_mav, …,
   ch1_mnf, ch2_rms, …`, never through `sorted()`.
3. **Never commit NinaPro data.** `data/` is already gitignored; keep raw
   recordings there. Redistribution is governed by NinaPro's terms, so read
   them before putting real recordings into a public deployment.
4. **Label every number with its source.** Simulated and recorded results must
   never be mixed in one table without a label saying which is which.
5. **Tests first, and never weaken an assertion to make a failure pass.**
6. **Design rules are enforced by tests** (`packages/design/test/`). Carmine is
   used only for muscle activation, never for headings or buttons. Grid lines
   appear only behind real measured data. No gradients, glows or shadows.

## 4. Where things stand

| Phase | What | State |
| --- | --- | --- |
| 0–1b | Simulator, Python/TypeScript feature conformance, decoder, ONNX, browser Live/virtual hand | Done |
| Recorded DB2 path | Six local S1/S10 sessions, loader and mapping, comparison, held-out export/replay, Results views | Implemented and tested; exploratory, two subjects |
| 2 | Neuromotor Cartography (Pillar 1) | Partly done; see §6 |
| 3 | Error-attribution router, anatomical coaching | Not started |
| 4 | Robustness | Electrode-shift sweep complete: passes through 0.2 rad, fails at 0.3/0.4 rad; simulated only |
| 5 | Azure deploy, patent disclosures | Not started |

The app has Live and Results screens. Additional independent recorded subjects,
clinician mode, error attribution, anatomical coaching, amputee evaluation,
and deployment remain future work.

---

## 5. NinaPro DB2: completed work and verified facts

### Dataset ingestion and quality audit

The local dataset has six A1 MATLAB recordings: S1 and S10, each with E1, E2,
and E3. The audit was run against these actual files (not only synthetic test
fixtures):

| Check | Observed |
| --- | --- |
| Subjects / files | S1 and S10; 6 files total (E1-E3 per subject) |
| EMG dimensions | Samples × 12 channels, float32; the loader retains channels 1-8 |
| Sample rate | 2 kHz, matching the feature pipeline and DB2 documentation |
| Label arrays | E1 has 0-17, E2 has 0 and 18-40, E3 has 0 and 41-49; each exercise's expected IDs were present |
| Repetitions | 0-6 in all six sessions; 0 marks rest |
| Finite values | Every raw EMG array and extracted feature vector is finite |
| Unequal tails | S1 E3 has 1 extra EMG sample and S10 E3 has 2; the loader trims only these unmatched tails |
| Unit multiplier | Loader heuristic selected 1.0 (no conversion) for all six files; values were treated as volts |
| Rest level | Median per-channel rest RMS across all session/channel pairs is 8.09 µV; maximum is 66.44 µV (S10 E1 channel 1) |
| Extracted corpus | 35,758 windows × 136 features, from two subjects; mapped windows are finite |

Rest RMS is not uniform: some channel/session outliers exceed the project's
nominal baseline. The unit scale is a loader heuristic supported by the
observed magnitudes, not a fresh independent instrument calibration. Treat
this as a local dataset audit, not a guarantee of electrode quality or
human-device safety. The loader uses `restimulus`, keeps only 200 ms windows
whose label and repetition are constant, samples every eighth rest window,
and marks fatigue unknown (zero) rather than inventing a fatigue label.

The full DB2 release has 40 intact subjects; only S1 and S10 are present
locally. This evaluation cannot establish generalization across that
population.

### Movement mapping

Keeping the app vocabulary means the virtual hand, commit costs and UI need no
changes. The label assignments below were checked against the
[official DB2 guide](https://ninapro.hevs.ch/instructions/DB2.html) and
Atzori et al. (2014), [Table 1 and Figure 2](https://doi.org/10.1038/sdata.2014.53).
`restimulus` is the post-hoc realigned performed-movement label. DB2 numbers
movements locally within each exercise; the global IDs below follow the
documented sequential exercise ranges.

| App gesture | NinaPro movement | Exercise / label |
| --- | --- | --- |
| rest | rest | 0 |
| thumb_up | Thumb up | E1 / 1 |
| two_finger | Extension of index and middle, flexion of the others | E1 / 2 |
| open_hand | Abduction of all fingers (proxy, not an exact general open-hand label) | E1 / 5 |
| fist | Fingers flexed together in fist | E1 / 6 |
| point | Pointing index | E1 / 7 |
| wrist_flexion | Wrist flexion | E1 / 13 |
| wrist_extension | Wrist extension | E1 / 14 |
| spherical_grip | Power sphere grasp | E2 / 27 |
| pinch | Tip pinch grasp | E2 / 32 |

The DB2 movement descriptions are the labels used in recorded-data Results and
replay. In particular, label 5 is finger abduction, displayed as an explicitly
qualified proxy for the app's `open_hand` class; it must not be reported as a
generic open-hand movement. `pinch` means tip pinch and `spherical_grip` means
power sphere grasp, not every possible pinch or spherical grip.

### Recorded comparison and held-out export

The earlier `TypeError: 'int' object is not iterable` came from passing
`calibration_folds` as the third positional argument to
`evaluate_repetition_split`; that position is `train_repetitions`. The caller
now passes it by keyword, so repetition metadata remains a tuple when the
summary serializes it. The comparison also reports subject count/IDs from the
actual corpus rather than simulator configuration.

The DB2 comparison runs on the actual 35,758-window corpus. Two-subject LOSO
results are in `artifacts/model_comparison_ninapro.json`:

| Model | Accuracy (mean ± SD) | ECE |
| --- | --- | --- |
| LDA | 49.17% ± 4.65 percentage points | 0.1720 |
| Linear SVM | 45.59% ± 8.27 percentage points | 0.1593 |
| RBF SVM | 38.57% ± 11.01 percentage points | 0.3775 |

These are exploratory two-subject comparison results, not final population
estimates. Keep the split and rest sampling explicit when reusing the artifact.

The separate `neurogrip.heldout` workflow trains LDA on S1, tests S10, exports a
NinaPro-only ONNX model, and generates a held-out S10 replay. It enforces
disjoint training and test subjects. Current metrics in
`artifacts/ninapro_holdout/evaluation.json`:

- 18,465 training windows; 17,293 held-out windows.
- Accuracy 45.09%, majority-class baseline 36.51%, ECE 0.1644.
- Confusion matrix total equals all 17,293 test windows; the Results screen
  renders all matrix cells and 54 performed-versus-decoded timeline events.
- ONNX/sklearn probability parity passed on 2,000 held-out windows; maximum
  absolute difference was `3.816e-7`.
- CPU ONNX P95 is 0.0322 ms for 1,000 individual pre-extracted feature rows.
  It excludes feature extraction, browser worker messaging, and UI time.

The browser bench is a separate worker measurement. The recorded replay
manifest calibrates each channel's minimum RMS to 25% of that subject's
recorded-rest RMS, capped at the existing 5 µV floor. With that replay-specific
calibration, the latest run processed 1,000 windows: all 1,000 passed the
quality gate and were decoded, with threaded WASM enabled and P95 1.325 ms
against the 10 ms budget. Exact latency varies between runs. P95 covers worker
feature extraction plus ONNX inference, excludes message/UI overhead, and
pushes windows in lockstep rather than real time. It is not a physiological
reaction-time measure. This calibration is for replay demonstration only; it
has not been validated as a live-electrode safety gate. The simulator still
uses the original fixed 5 µV threshold.

### Held-out replay and Live screen (implemented)

The held-out workflow writes an int16 replay bundle with a JSON manifest that
names the source, test subject, and training subjects. It rejects a test subject
that overlaps training. Replay data stays under the git-ignored
`apps/web/public/replay/ninapro/` directory. Live selects the recorded bundle
and its separate model only when both are staged, displays provenance, and
shows the dataset movement descriptions. No NinaPro samples should be
committed or redistributed without permission.

### Results screen and latest browser data (implemented and verified)

Results has separate sections for simulated comparison, recorded NinaPro
comparison, held-out accuracy/confusion/timeline, and simulator versus recorded
browser latency. The confusion matrix scrolls horizontally on mobile without
causing page-level overflow.

The browser latency test writes JSON after Playwright's production preview has
already been built. Therefore, after a fresh `npm run test:e2e`, run
`npm run build --workspace @neurogrip/web` to put that exact run's JSON into
`apps/web/dist/results/`. The latest staged files were compared to their
artifact/public sources; they are identical. A focused Results browser test
after the build verifies the displayed held-out accuracy, calibration, every
confusion cell, timeline event count, latest worker P95, and accepted/rejected
counts.

## 6. Remaining work, in priority order

1. **Broaden the recorded-data evaluation.** Obtain additional independently
   licensed DB2 subjects, include them in the data-quality audit, rerun the
   model comparison, and report uncertainty across a meaningful subject set.
   The present S1-to-S10 split is a single exploratory holdout.
2. **Review scientific validity before making claims.** Label 5 is finger
   abduction used as an `open_hand` proxy; rest RMS varies by channel/session;
   the model's ECE is 0.1644. Keep these limitations visible and assess whether
   mapping, signal quality, calibration, and class balance are acceptable.
3. **Continue unfinished project phases** below: Cartography gate/assignment,
   error attribution, anatomical coaching, clinician mode, amputee evaluation,
   and deployment.
4. **Do not optimize `build:assets` yet.** The app's production Vite build was
   about 5 seconds and the full E2E suite about 33 seconds in this environment.
   `build:assets` is an explicit simulator asset regeneration pipeline, not
   part of routine E2E or normal app startup. Its own duration was not
   benchmarked in this pass; optimize only if it becomes a measured practical
   contributor workflow problem.

---

## 7. Phase 2 (Cartography): status and an honest note

Built in `services/research/neurogrip/cartography.py`, with 19 tests:

- **Babble:** a simulated exploratory walk over muscle synergies.
- **Synergy map:** NMF synergy extraction, with a 2-D map for display.
- **Fields:** reachability, effort and separability.
- **Synthesis:** places prototypes from the user's own babble poses.
- **Repeat probe:** the user re-attempts shortlisted poses so repeatability can
  be measured.
- **Weighted catalogue selection:** picks catalogue gestures, weighing
  separability against effort and fatigue.

Run it with `python -m neurogrip.experiments cartography`, which writes
`artifacts/cartography_gate.json`.

**Result:** synthesised gestures did **not** beat catalogue selection in the
simulator (0 of 5 subjects in both groups). Effort- and fatigue-weighted
catalogue selection traded 11–21% of worst-pair separability for 7–15% less
effort.

**Caveat:** that test was stricter than the spec's actual gate. It compared
whole vocabularies against the best possible catalogue choice, while the spec
asks whether *a* synthesised gesture is learnable and beats *a* catalogue
gesture. Several choices also departed from the design spec
(`docs/superpowers/specs/2026-09-04-neurogrip-design.md`, Pillar 1); the module
docstring lists them. Before more Phase 2 work:

1. Write a short Phase 2 plan in `docs/superpowers/plans/` that records these
   decisions and gets the team to agree on the gate definition.
2. Re-run the comparison under the spec's per-gesture gate.
3. Then build gesture-to-function assignment (the Hungarian algorithm,
   weighted by `apps/web/src/decode/commitCost.ts`) and the Cartography screen.

Cartography remains unfinished. Before extending it, expand the recorded
evaluation first; babble on real EMG would require a separately designed
recording session and safety review.

## 8. Later phases (summary only)

- **Phase 3:** attribute each error to physiology, user behaviour, model
  capacity or drift. Validate with injected synthetic faults of each type.
  Then add forearm-anatomy coaching.
- **Phase 4:** clinician screens (Evaluate, Robustness, Latency, Records). The
  amputee stratum needs NinaPro DB3 (amputees, 2 kHz, also registration) and
  must be reported separately with sample-size caveats.
- **Phase 5:** Azure deployment and patent-disclosure drafts. Per the design
  spec, keep the repo private and draft the disclosures before anything goes
  public.

## 9. Working with an AI assistant on this

- Paste this file, `CLAUDE.md`, and the specific files the task names. The
  assistant can't see the repo, so give it the real file contents rather than
  a description of them.
- Ask for one task at a time, tests first, and run the tests yourself before
  accepting the code.
- If it suggests sorting feature names, loosening the conformance tolerance,
  committing data, or quoting a simulated number as a real one, reject it.
  Those are the four mistakes this repo is built to prevent.
