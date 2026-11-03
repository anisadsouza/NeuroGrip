# NeuroGrip: current state and next steps

Updated 2026-11-04 after the Cartography gate, simulator-only attribution pilot,
and clinician-oriented research review screens. Read
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
metrics, and browser/CPU latency separate. Cartography and Attribution are
simulator-only research demonstrations. No live electrode input is implemented.

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
| Recorded DB2 path | Nine local S1/S10/S11 sessions, loader and mapping, subject-held-out export/replay, Results views, S11 quality audit | Implemented; exploratory, three subjects; independent benchmark and full-size SVM sweep remain |
| 2 | Neuromotor Cartography (Pillar 1) | Simulated map, corrected per-prototype gate, illustrative assignment and screen implemented; human validation remains |
| 3 | Error-attribution router, anatomical coaching | P/B/M/D simulator pilot and channel-localized coaching screen implemented; offline held-out LDA score-margin attribution and three reference-model comparisons added; real-data generalization and human coaching validation remain |
| 4 | Clinician views and robustness | Evaluate/Robustness/Latency/Records views built; the electrode-shift view is simulated only. Recorded-fault validation and amputee evaluation remain |
| 5 | Azure deploy, patent disclosures | Deployment not started; internal draft prepared locally; legal/inventorship review and permission clearance remain |

The app has Live, Results, Cartography, Attribution and clinician-oriented
Research review screens. Additional independent recorded subjects, validated
error attribution/coaching and robustness, amputee evaluation, and deployment
remain future work.

The technical summary is in `docs/technical_reference.md`; the separate
`docs/data_and_release_review.md` records that redistribution terms are
unverified. Internal disclosure drafts are local working material and must not
be committed while the GitHub repository is public. None is a legal clearance
or patentability opinion.

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
substantially longer. A separate bounded comparison across all three local
subjects has now tested LDA variants, rest balancing, calibration, and sampled
SVMs. A full-size SVM comparison remains unrun. These three subjects cannot
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
majority baseline. The current Results export is a separate uniform-prior LDA
artifact on the same named split: 46.61% accuracy and ECE 0.026. An archived
`ninapro_lda_uniform/evaluation.json` reports 46.85% for that split, but its
run provenance has not been reconciled; use the current `ninapro_holdout`
artifact when describing the app. S10 informed diagnostics, so these are
exploratory rather than untouched final tests. The additional uniform-prior
check trained on S1+S10 and tested on S11 scored 23.07% on 20,956 windows,
against a 22.26% majority baseline (ECE 0.3664). This is poor cross-subject
performance. The full-corpus multi-model comparison was not completed; a
bounded, three-subject comparison is documented below; the full-size SVM sweep
remains unrun.

The S11 confusion matrix predicts `rest` for 19,125 / 20,956 windows. Recall
is 99.08% for `rest`, 5.11% for `thumb_up`, 4.92% for `two_finger`, and 1.63%
for `spherical_grip`; `fist`, `open_hand`, `pinch`, `point`, `wrist_flexion`,
and `wrist_extension` all have 0% recall. For every movement class, its most
frequent error is prediction as rest. This failure pattern and the S11 signal
audit are included in the model card.

### Exploratory attempts to improve cross-subject accuracy

These are development experiments on the three local subjects, not a new
headline result. Nothing from these experiments replaced the recorded decoder.
The checked-in runner
`python -m neurogrip.ninapro_recognition --data-root data/ninapro/DB2
--out artifacts/ninapro_recognition_experiment.json` evaluates the same
56,714-window corpus with leave-one-subject-out folds. Its sigmoid calibrators
use grouped internal folds by subject and repetition, keeping overlapping
windows from the same repetition together. SVMs are capped at 250 examples per
class per training subject to bound runtime. The resulting local artifact is
gitignored with other NinaPro-derived experiment outputs.

| Strategy | Mean LOSO accuracy | Mean majority baseline | Mean macro movement recall | Mean predicted-rest fraction | OOF ECE |
| --- | ---: | ---: | ---: | ---: | ---: |
| LDA empirical, raw | 37.46% | 34.95% | 18.05% | 53.98% | 0.4807 |
| LDA empirical, sigmoid | 37.41% | 34.95% | 14.64% | 61.38% | 0.2168 |
| LDA uniform, raw | 35.08% | 34.95% | 20.60% | 44.21% | 0.4654 |
| LDA uniform, sigmoid | 37.27% | 34.95% | 15.84% | 58.60% | 0.1676 |
| LDA uniform, rest downsampled + sigmoid | 29.70% | 34.95% | 23.04% | 26.81% | 0.1421 |
| Linear SVM, sampled + sigmoid | 29.23% | 34.95% | 18.91% | 35.30% | 0.1708 |
| Linear SVM, sampled + class-weighted + sigmoid | 29.22% | 34.95% | 18.91% | 35.31% | 0.1713 |
| RBF SVM, sampled + class-weighted + sigmoid | 27.96% | 34.95% | 19.86% | 36.32% | 0.2667 |

These are equal-weight means over the held-out S1, S10, and S11 folds;
calibration error is calculated over all out-of-fold predictions. The
leave-one-subject-out analysis includes subjects already inspected during
development, so it is exploratory, not an untouched final test. Rest
downsampling reduced the rest-prediction tendency and raised average movement
recall, but lowered mean accuracy below the average majority baseline.
Calibration reduced ECE for the LDA variants but did not solve cross-subject
recognition. The weighted and unweighted sampled linear SVM results were
effectively identical. No tested candidate was adopted.

The separate `ninapro_feature_attribution` runner computes an exact
per-feature decomposition of the predicted-versus-intended score margin for
each uncalibrated uniform-prior LDA error. It also compares three trained
references in the same subject-held-out folds (balanced LinearSVC, balanced
RBF SVM, and balanced random forest), capped at 250 examples per class per
training subject. Pooled scores were 34.80% for LDA, 29.50% for LinearSVC,
29.36% for RBF SVM, and 38.04% for random forest. The random forest was right
on 19.7% of LDA errors. Because these three subjects informed model and
analysis choices, the random-forest result is an exploratory lead, not a
reliable benchmark or reason to change the deployed decoder. The max numeric
residual in the LDA score decomposition was 1.21e-13.

The grouped-calibration uniform-LDA S11 fold scored 24.17% with ECE 0.4126,
compared with 23.07% / 0.3664 in the existing staged S11 artifact. Both use
the same 20,956 held-out windows, but the prior artifact uses the existing
stratified calibration path while this runner groups internal calibration
folds by subject/repetition. Keep the staged artifact as the app's reported
result; these are distinct calibration runs and must not be conflated.

- Changing LDA from empirical to uniform class priors barely moved the
  three-subject mean (36.15% vs 36.23%). With uniform priors, S11 remained at
  23.06%, predicted rest for 91.25% of windows, and had 1.30% macro movement
  recall.
- Per-subject channel scaling from recorded rest RMS was tested as a
  transductive calibration experiment. It used the held-out subject's rest
  windows to estimate scale, so it is not an untouched-test result. It lowered
  mean LOSO accuracy to 33.09%; S11 fell to 15.06%. Dropping channel 1 also
  failed to help (35.98% mean LOSO; 22.65% on S11). The noisy S11 E2 channel is
  therefore not, by itself, a demonstrated explanation for the failure.
- Downsampling training rest to the median movement-class count raised S11 to
  25.65%, with 17.33% macro movement recall and 26.37% predicted-rest windows,
  but the three-subject mean fell to 16.44%. A nested choice of rest ratio from
  training-subject validation selected 4× for the S11 fold and gave 27.58% on
  S11, but only 7.72% macro movement recall; its three-subject mean was 25.47%.
  These variants trade a rest-only prediction for poor performance elsewhere
  and are not a general solution.
- Earlier inline SVM probes used a different calibration procedure and are
  retained as development notes, not as the canonical repeatable comparison.
  The checked-in grouped-calibration runner above is the reproducible
  comparison for class balancing and calibration.

The limited gain on S11 does not support an 80% claim. Rest balancing and
rest-based normalization need a larger, independent subject set and a
pre-specified evaluation before adoption. S11 has already been inspected and
is development data, not an untouched final test subject.

### Recorded quality gate and fault probes (local development audit)

Step 3 is implementation-complete; its scientific validation remains
incomplete.

The replay manifest calibrates each channel's minimum RMS to 25% of that
subject's selected recorded-rest segment, capped at the shared 5 uV floor.
After the waveform heuristics were added, the gate was rerun on all 17,293 S10
held-out windows: 16,572 (95.83%) accepted and 721 (4.17%) rejected. Rejections
include movement windows as well as rest, so they must not be interpreted as
correctness. Acceptance and rejection counts are quality-gate observations,
not accuracy measurements.

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

The window-level gate additionally checks for repeated clipping plateaus,
repeated large sample steps, and abrupt half-window RMS changes. A separate
rest-calibration diagnostic now computes per-channel RMS across all
rest-labeled samples in a subject's available recordings. Its exploratory
50 µV threshold raises a warning but does not stop replay or apply a ceiling to
active windows:

| Subject | Rest samples | Channel-1 RMS | Screen |
| --- | ---: | ---: | --- |
| S1 | 2,841,213 | 14.14 µV | Clear |
| S10 | 2,138,968 | 56.82 µV | Flag channel 1 |
| S11 | 1,609,557 | 102.74 µV | Flag channel 1 |

The exercise-specific S11 E2 rest RMS remains 136.70 µV. The screen identifies
an unusual baseline level, not its cause. Its cutoff was selected on these
development subjects, and the recordings have no independent artifact labels;
this is not threshold validation or real-world fault validation.

The reproducible audit also summarizes non-overlapping 200 ms rest-labeled
windows. For S11 E2 channel 1, 2,121 windows have median RMS 6.31 µV, P95
76.87 µV, P99 496.28 µV, and maximum 3,879.40 µV. This strongly skewed
distribution is consistent with intermittent high-amplitude windows rather
than a uniformly elevated baseline; 127/2,121 windows (6.0%) exceed the
exploratory 50 µV advisory level. The dataset does not identify their cause or
provide artifact labels for validating that threshold. Those windows occur in
18 separate runs; the longest is 49 consecutive windows (9.8 seconds). The
maximum-RMS window begins at sample 1,704,000 (about 14.2 minutes into E2).
The accompanying channels 2-8 remain far lower in the same 200 ms windows.
This channel-local, intermittent pattern narrows the symptom but cannot
distinguish electrode contact, movement, or another acquisition artifact.

A repeatable fault probe samples 100 stable 200 ms S10 windows from each of
the ten mapped classes (1,000 total, seed 20261103) and applies one fault at a
time using the replay's recorded per-channel floors. The revised clean sample
rejected 17/1,000 (1.7%): 14 failed the pre-existing per-channel RMS floors,
one triggered the clipping heuristic, and two triggered the gain-shift
heuristic. A flat channel in channel 1 rejected 1,000/1,000 (100%). Clipping
all channels at ±50 µV rejected 966/1,000 (96.6%). A 1 mV alternating noise
burst lasting 20 samples and a 10× amplitude jump halfway through the window
each rejected 1,000/1,000. These are controlled signal manipulations, not
measured fault detector sensitivity or clinical validation. Reproduce the
rest-baseline screen with:

```bash
python -m neurogrip.ninapro_rest_calibration \
  --data-root data/ninapro/DB2 \
  --out artifacts/ninapro_rest_calibration.json
```

### Recorded-data feature attribution

`python -m neurogrip.ninapro_feature_attribution` fits a uniform-prior LDA and
three reference models (balanced LinearSVC, RBF SVM, and random forest) in
each leave-one-subject-out fold. The LDA's predicted-versus-intended score
margin is exactly decomposed into standardized per-feature contributions for
every held-out LDA error. On the current three-subject corpus, pooled
uncalibrated LDA accuracy was 34.80%; LinearSVC, RBF SVM, and random forest
scored 29.50%, 29.36%, and 38.04%. The random forest was correct on 19.7% of
LDA errors. These research models are not the deployed ONNX decoder; pooled
scores are not directly comparable to equal-subject means in the recognition
table. The maximum numerical reconstruction residual was below 1.3e-13. This
is a recorded-data model explanation, not an attribution accuracy estimate,
causal physiology, or human coaching validation; all three subjects have
informed development.

Reproduce the analysis with:

```bash
python -m neurogrip.ninapro_feature_attribution \
  --data-root data/ninapro/DB2 \
  --out artifacts/ninapro_feature_attribution.json
```

The separate screen warns about S10 and S11 channel 1 but does not identify
motion artifacts or electrode-contact faults. Reproduce the window-level
injection probe with:

```bash
python services/research/scripts/ninapro_quality_fault_probe.py
```

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
   multi-model comparison (only a bounded subsampled SVM comparison has been
   completed), investigate S11 E2 channel 1's high rest RMS, and
   evaluate more independently licensed DB2 subjects if available. Report
   uncertainty across a meaningful subject set. Current S1-to-S10 and
   S1+S10-to-S11 results are exploratory checks, not population-level evidence.
2. **Review scientific validity before making claims.** Label 5 is finger
   abduction used as an `open_hand` proxy; rest RMS varies by channel/session;
   the S11 model is poorly calibrated and misses six gesture classes. Keep
   these limitations visible and assess mapping, signal quality, calibration,
   and class balance with additional subjects.
3. **Continue unfinished project phases** below: Cartography teach-back;
   validate offline feature attribution on additional untouched subjects,
   integrate it with error routing, and validate coaching with human
   participants; clinician mode, amputee evaluation, and deployment.
4. **Verify NinaPro/Kaggle provenance and redistribution terms.** The owner
   identifies Kaggle listing `quddusikashaf/ninapro-db2`; Kaggle currently
   displays CC0 and names Quddusi Kashaf as owner/creator. Verify that the
   uploader was authorized to license the upstream NinaPro data and confirm
   derived-file rights before publishing raw or derived files.
5. **Do not optimize `build:assets` yet.** The app's production Vite build was
   about 5 seconds and the full E2E suite about 33 seconds in this environment.
   `build:assets` is an explicit simulator asset regeneration pipeline, not
   part of routine E2E or normal app startup. Its own duration was not
   benchmarked in this pass; optimize only if it becomes a measured practical
   contributor workflow problem.

---

## 7. Phase 2 (Cartography): status and an honest note

The simulator pipeline and browser demonstration are implemented. The Python
Cartography tests cover the map, synthesis, evaluation and matching; the
assignment helper has focused TypeScript tests, and the screen has a browser
test.

- **Babble:** a simulated exploratory walk over muscle synergies.
- **Synergy map:** NMF synergy extraction, with a 2-D map for display.
- **Fields:** reachability, effort and separability.
- **Synthesis:** places prototypes from the user's own babble poses.
- **Repeat probe:** the user re-attempts shortlisted poses so repeatability can
  be measured.
- **Weighted catalogue selection:** picks catalogue gestures, weighing
  separability against effort and fatigue.

Run `python -m neurogrip.experiments cartography --out
artifacts/cartography_gate.json`, then `npm run stage:cartography` to produce
the browser's simulator-only demo report. The evaluation fits scaling inside
each repetition-held-out fold and compares synthesized targets one by one to
distinct matched catalogue functions. The agreed pass criterion is at least
80% held-out repetition recall and better nearest-rival Fisher separation.

**Result:** in the regenerated eight-subject cohorts, at least one prototype
passes for 2/8 intact subjects and 5/8 finger-atrophy simulated subjects.
Across the cohorts, 3/40 and 6/40 prototypes pass respectively. Weighted
catalogue selection retains 97.22% of max-min separation while reducing
excitation by 2.27 percentage points and fatigue by 1.33 points for intact
subjects; for finger atrophy it retains 90.77% while reducing excitation by
4.61 points and fatigue by 2.65 points. In this simulator, fatigue is largely
proportional to excitation, so those savings are not independent.

The Cartography screen displays the first intact simulated subject's map,
prototype gate, illustrative Hungarian function assignment, and cohort counts.
This completes the software demonstration milestone, not the research claim:
there is no recorded human babble, teach-back, or wearer validation, and the
function assignment is not a clinical recommendation. Real-user mapping and
teach-back remain future work.

## 8. Phase 3 (error attribution and coaching): simulator pilot

`services/research/neurogrip/attribution.py` implements four evidence channels:

- **P:** diagonal-Gaussian Bhattacharyya overlap of the two cleanest 80% class
  repeat clusters.
- **B:** maximum channel-wise standardized deviation from the intended class's
  repeat cluster, relative to the historical 95th-percentile spread.
- **M:** whether a stronger reference prediction is correct while the deployed
  prediction is wrong.
- **D:** centroid CUSUM and median-frequency shift, each relative to an
  exploratory threshold.

The scores are normalized, but the router requires minimum raw evidence and a
lead margin; weak or conflicting evidence is reported as inconclusive. The
generated demo has one designed injection per cause and routes 4/4 to the
expected action. This is a logic check on known synthetic cases, not a measured
attribution accuracy. In the demo, the reference-model result is supplied as
evidence; it is not a newly trained k-NN result.

The Attribution screen visualizes channel-level residuals on the electrode
ring and provides bounded practice text for the behavioral case. This is not
SHAP/integrated-gradients model attribution and does not identify a specific
muscle or finger. Thresholds, routes, and coaching are not validated on
recorded errors or human users. No clinical or safety claim is supported.

Reproduce and stage the demonstration:

```bash
python -m neurogrip.experiments attribution --out artifacts/attribution_demo.json
npm run stage:attribution
```

### Remaining phases

- **Phase 3 validation:** test the preliminary attribution and reference-model
  comparison on additional untouched subjects, then validate whether coaching
  is useful and understandable with approved human-participant research.
- **Phase 4:** clinician screens (Evaluate, Robustness, Latency, Records). The
  amputee stratum needs NinaPro DB3 (amputees, 2 kHz, also registration) and
  must be reported separately with sample-size caveats.
- **Phase 5:** Deployment remains unstarted. Internal patent-disclosure
  working drafts exist, but inventorship, ownership, prior art, disclosure
  timing, and filing strategy need review with the team/institution and patent
  counsel before anything is made public.

## 9. Working with an AI assistant on this

- Paste this file, `CLAUDE.md`, and the specific files the task names. The
  assistant can't see the repo, so give it the real file contents rather than
  a description of them.
- Ask for one task at a time, tests first, and run the tests yourself before
  accepting the code.
- If it suggests sorting feature names, loosening the conformance tolerance,
  committing data, or quoting a simulated number as a real one, reject it.
  Those are the four mistakes this repo is built to prevent.
