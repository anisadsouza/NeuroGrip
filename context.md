# NeuroGrip: current state and next steps

Updated 2026-11-02 after the S11 subject-held-out and signal-quality audits,
current browser latency measurement, and app command-readout verification. Read
`CLAUDE.md` for the project's engineering and safety rules.

**Research prototype. Not a medical device. Not validated for clinical use.**
The project now has a real-recording evaluation path with three locally
available intact subjects (S1, S10, and S11). S10 informed candidate
diagnostics. The S1-to-S10 score is a development result; the quick S1+S10-to-
S11 check is only slightly above its majority baseline and has poor calibration.
Neither establishes population-level performance. Simulator results and
recorded results remain separate.

---

## 1. The project in one paragraph

NeuroGrip decodes eight channels of forearm surface EMG (sEMG) into a ten-class
gesture vocabulary. Python (`services/research/`) simulates data, ingests
NinaPro DB2 recordings, extracts features, evaluates models, and exports ONNX.
The browser (`apps/web/`) runs features and inference in a worker every 20 ms.
Live can replay either simulated signal or a locally generated held-out DB2
recording, with source/model provenance. Results keeps simulated comparisons,
recorded comparisons, held-out confusion/timeline, quality-gate and accuracy
metrics, and browser/CPU latency separate. No live electrode input is
implemented.

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
| Recorded DB2 path | Nine local S1/S10/S11 sessions, loader and mapping, subject-held-out export/replay, Results views, S11 quality audit | Implemented; exploratory, three subjects; full multi-model comparison remains |
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

The local dataset has nine A1 MATLAB recordings: S1, S10, and S11, each
with E1, E2, and E3. The loader, exercise labels, rest proportions, and
first-eight-channel rest RMS were checked for all nine files.

| Check | Observed |
| --- | --- |
| Subjects / files | S1, S10, and S11; 9 files total (E1-E3 per subject) |
| EMG dimensions | Samples × 12 channels, float32; the loader retains channels 1-8 |
| Sample rate | 2 kHz, matching the feature pipeline and DB2 documentation |
| Label arrays | All three subjects have rest label 0 and every expected global movement ID: E1 1-17, E2 18-40, E3 41-49 |
| Repetitions | 0-6 in all nine files; 0 marks rest |
| Finite values | All nine files load; movement label ranges match across subjects |
| Unequal tails | S1 E3 has 1 extra EMG sample and S10 E3 has 2; the loader trims only these unmatched tails |
| Unit multiplier | Loader heuristic selected 1.0 (no conversion) for all nine files; values were treated as volts |
| Rest sample fraction | S1 E1/E2/E3: 58.87% / 56.59% / 37.83%; S10: 54.79% / 36.07% / 26.34%; S11: 26.61% / 35.45% / 25.98% |
| Rest RMS | S11 E2 channel 1 is 136.70 µV, a conspicuous session/channel outlier; S10 E1 channel 1 is 66.44 µV |
| Extracted corpus | Original S1/S10 corpus: 35,758 windows × 136 features; current three-subject corpus: 56,714 windows × 136 features |

Rest RMS is not uniform: S11 E2 channel 1 is especially high, while several
other channels have low single-digit microvolt RMS. The cause of the outlier
cannot be inferred from these files; there are no independent artifact labels.
The unit scale is a loader heuristic, not a fresh independent instrument
calibration. Treat this as a local dataset audit, not a guarantee of electrode
quality or human-device safety. The loader uses `restimulus`, keeps only 200 ms windows
whose label and repetition are constant, samples every eighth rest window,
and marks fatigue unknown (zero) rather than inventing a fatigue label.

Movement IDs match across S1, S10, and S11 with no subject-specific offset:
E1 contains labels 1–17, E2 18–40, and E3 41–49, plus rest label 0. For S11,
rest RMS in µV across channels 1–8 is E1 `[7.27, 3.62, 3.38, 4.42, 2.58,
2.66, 4.94, 9.70]`, E2 `[136.70, 3.29, 3.17, 3.06, 2.69, 3.10, 5.93, 9.89]`,
and E3 `[21.28, 10.74, 8.98, 4.10, 2.97, 3.43, 10.81, 19.17]`. The unusual
S11 E2 channel-1 level needs investigation; the files do not identify its cause.

The full DB2 release has 40 intact subjects; only S1, S10, and S11 are present
locally. A quick LDA-uniform model trained on S1+S10 and tested on S11 scored
23.07% across 20,956 windows, versus a 22.26% majority-class baseline, with ECE
0.3664. This is poor cross-subject performance. The comprehensive comparison
across model families was stopped because its SVM evaluations were taking
substantially longer; it remains to be run. This three-subject check cannot
establish generalization across the full population.

The low S11 score is a cross-subject failure: despite rest being about 22% of
the held-out windows, the model predicted rest for about 91%. Possible
contributors include subject-specific signal scale/electrode placement,
the S11 E2 channel-1 rest RMS outlier, calibration and feature-distribution
shift, and the limited two-subject training set. The current evidence does not
isolate a single cause. The open-hand label is also a finger-abduction proxy.
Reaching 80% cannot be promised by changing a threshold or normalizing one
channel; it needs experiments on training subjects and evaluation on a new,
untouched subject.

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

The ordinary-LDA S1→S10 reference is 45.09% on 17,293 windows, with a 36.51%
majority baseline. S10 informed diagnostics, so this is exploratory rather
than an untouched final test. The additional uniform-prior check trained on
S1+S10 and tested on S11 scored 23.07% on 20,956 windows, against a 22.26%
majority baseline (ECE 0.3664). This is poor cross-subject performance. The
full multi-model comparison was stopped because its SVM evaluations were
taking substantially longer and remains open.

The S11 confusion matrix predicts `rest` for 19,125 / 20,956 windows. Recall
is 99.08% for `rest`, 5.11% for `thumb_up`, 4.92% for `two_finger`, and 1.63%
for `spherical_grip`; `fist`, `open_hand`, `pinch`, `point`, `wrist_flexion`,
and `wrist_extension` all have 0% recall. For every movement class, its most
frequent error is prediction as rest. This failure pattern and the S11 signal
audit are included in the model card.

### Recorded quality gate and correctness (validated against the local corpus)

The replay manifest calibrates each channel's minimum RMS to 25% of that
subject's selected recorded-rest segment, capped at the shared 5 uV floor.
Running the same gate over all 17,293 S10 held-out windows accepted 16,691
(96.52%) and rejected 602 (3.48%). **All 602 rejected windows are labeled rest;
no mapped movement window was rejected.** Acceptance and rejection counts are
quality-gate observations, not accuracy or correctness measurements.

Controlled negative controls reject a zero signal, a detached channel, and a
sample window attenuated to half amplitude. This shows the code responds to
obvious low-signal probes; it does not establish performance on actual
electrode detachment, motion artifact, or skin-impedance changes. The S10
recording has no independent artifact labels. Thresholds are replay-specific
and unvalidated for live electrodes; the simulator still uses its fixed 5 uV
minimum.

The current browser bench scores top-1 model predictions only after a 200 ms
transition warm-up, runs an all-zero rejection control, and separately reports
quality rejects and worker latency. The 1,000-window latency run is not an
accuracy estimate of the full DB2 corpus. Latency covers feature extraction
plus ONNX inference in the worker, excludes message/UI overhead, and pushes
windows in lockstep rather than in real time. It is not a physiological
reaction-time measure.

The latest recorded S10 browser latency artifact, measured 2026-11-02 on a
MacBook Air with Apple M3, reports worker P95 **1.93 ms**. An earlier browser
run reported P95 2.70 ms; timings vary between runs. They cover feature
extraction plus ONNX inference, exclude message/UI overhead, and are not total
user reaction time. The zero-signal control is rejected. The separate
simulated bench has its own accuracy and latency results; do not mix them with
recorded-data metrics.

The quality gate is a minimum-RMS gate plus a broad 5 V absolute-amplitude
sanity check. It has no useful per-channel RMS upper limit for detecting
excessive baseline noise. S11 E2 channel 1's 136.70 µV rest RMS is therefore
not rejected as noisy; high-RMS channel faults remain an unvalidated failure
mode.

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
comparison, held-out confusion/timeline, quality-gate acceptance, and simulator
versus recorded browser latency. The Live Decoder panel now shows the
current-window posterior, a named virtual action, its candidate/reversible/
latched/fallback state, and latency. It sends no physical motor command. The
confusion matrix scrolls horizontally on mobile without causing page-level
overflow. The virtual hand now uses the same class as the named output: it
previews the leader while reversible, shows the latched class after commitment,
and shows rest after timeout. Rest is not reported as actuator travel.
This confirms software wiring only: risk-weighted commitment was evaluated on
simulator posteriors and is not validated on recorded NinaPro data. S11's
recorded-model ECE of 0.3664 makes its probabilities too poorly calibrated for
a claim that the commitment behaviour works reliably on that recording.

The browser latency test writes JSON after Playwright's production preview has
already been built. Therefore, after a fresh `npm run test:e2e`, run
`npm run build --workspace @neurogrip/web` to put that exact run's JSON into
`apps/web/dist/results/`. The latest staged files were compared to their
artifact/public sources; they are identical. A focused Results browser test
after the build verifies the displayed held-out accuracy, calibration, every
confusion cell, timeline event count, worker P95 and stable prediction score,
quality-gate counts, zero-signal control, and Live virtual action.

## 6. Remaining work, in priority order

1. **Broaden and complete the recorded-data evaluation.** Run the full
   multi-model comparison, investigate S11 E2 channel 1's high rest RMS, and
   evaluate more independently licensed DB2 subjects if available. Report
   uncertainty across a meaningful subject set. Current S1-to-S10 and
   S1+S10-to-S11 results are exploratory checks, not population-level evidence.
2. **Review scientific validity before making claims.** Label 5 is finger
   abduction used as an `open_hand` proxy; rest RMS varies by channel/session;
   the S11 model is poorly calibrated and misses six gesture classes. Keep
   these limitations visible and assess mapping, signal quality, calibration,
   and class balance with additional subjects.
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
