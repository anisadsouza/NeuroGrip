# NeuroGrip: context and next steps

A handoff for whoever picks this up next, written so it can be pasted into
ChatGPT or Claude along with the files a task names. Read `CLAUDE.md` alongside
it: that file holds the rules, the measured numbers and the reasoning behind
both. This file holds only what to do next.

**Research prototype. Not a medical device.** Every number in the repo so far
comes from simulated sEMG. Nothing has been evaluated on a human recording yet,
and Task 1 changes that.

---

## 1. The project in one paragraph

NeuroGrip decodes hand gestures from forearm surface EMG (sEMG) for prosthetic
control. A Python research service (`services/research/`) simulates sEMG,
extracts 17 features per channel, trains an RBF-SVM and exports it to ONNX. A
browser app (`apps/web/`) runs the same features and the ONNX model inside a web
worker every 20 ms. It shows the signal (oscilloscope), a commitment bar
(evidence building towards a decision) and a virtual hand that moves as the
evidence accrues. Today the app replays **simulated** signal. The next job is
to make it replay and decode **real recorded EMG from NinaPro**.

## 2. Setup

```bash
python -m venv .venv
source .venv/Scripts/activate          # macOS/Linux: source .venv/bin/activate
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
| 0–1b | Simulator, features in Python and TypeScript, conformance gate, decoder, ONNX, browser app, Live screen, virtual hand | Done |
| 2 | Neuromotor Cartography (Pillar 1) | Partly done; see §6 |
| 3 | Error-attribution router, anatomical coaching | Not started |
| 4 | Clinician screens, amputee stratum | Electrode-shift robustness eval exists (`neurogrip/robustness.py`) |
| 5 | Azure deploy, patent disclosures | Not started |

Only one screen exists: `apps/web/src/screens/Live.tsx`.

---

## 5. Next work, in priority order

### Task 1: Load NinaPro DB2 and train on real EMG (Python)

**Why DB2:** it is recorded at **2 kHz**, the rate the whole pipeline already
assumes, so no DSP constant changes. It has 40 intact subjects, and its first 8
channels form the forearm electrode ring the app's 8-channel layout models.

**Get the data:** register at the NinaPro site (ninapro.hevs.ch), download DB2
and unzip it to `data/ninapro/DB2/`. Each subject has files like `S1_E1_A1.mat`
(exercise 1), `S1_E2_A1.mat` and `S1_E3_A1.mat`. The variables needed are:

- `emg`: samples × 12 channels
- `restimulus`: per-sample movement label, re-aligned to the actual movement.
  Use this, not `stimulus`.
- `rerepetition`: per-sample repetition number, 1–6, with 0 meaning rest

**Verify these before writing code.** They are from the DB2 documentation as
best remembered, not checked against the files:

- Print `np.unique(restimulus)` for one file of each exercise. DB2 is believed
  to number labels globally: E1 = 1–17, E2 = 18–40, E3 = 41–49.
- Check the units of `emg` by printing per-channel RMS during rest. The
  pipeline's thresholds are in volts (about 15 µV at rest, a 5 µV detached
  electrode gate). If the data isn't in volts, convert it in the loader and
  record the conversion.

**Map NinaPro movements onto the app's 10 gestures.** Keeping the vocabulary
means the virtual hand, the commit costs and the UI need no changes. Tentative
mapping (verify every row against the DB2 movement list):

| App gesture | NinaPro movement | Exercise / label (verify) |
| --- | --- | --- |
| rest | rest | 0 |
| thumb_up | Thumb up | E1 / 1 |
| two_finger | Extension of index and middle, flexion of the others | E1 / 2 |
| open_hand | Abduction of all fingers | E1 / 5 |
| fist | Fingers flexed together in fist | E1 / 6 |
| point | Pointing index | E1 / 7 |
| wrist_flexion | Wrist flexion | E1 / 13 |
| wrist_extension | Wrist extension | E1 / 14 |
| spherical_grip | Power sphere | E2 / ~27 |
| pinch | Tip pinch grasp | E2 / ~32 |

**Build:**

1. `services/research/neurogrip/ninapro.py`:
   - `load_session(path) -> (emg float32 [channels, samples], labels, reps)`.
     Keep channels 1–8 only, and say why in a comment.
   - `build_ninapro_corpus(root, subjects, mapping) -> Corpus`. Return the same
     `Corpus` dataclass as `neurogrip/datasets.py`, so evaluation, training
     and export work unchanged. Use 200 ms windows with a 20 ms hop, and drop
     any window that straddles a label change.
   - Subsample `rest`, which otherwise dominates. Put the fatigue field to zeros
     and say in a comment that it's unknown for real data.
2. Add `--source {simulator,ninapro}` and `--data-root` to `compare` and `export`
   in `neurogrip/experiments.py`. Write `"trained_on": "ninapro_db2"` into
   `decoder.json`.
3. Evaluate two ways and report both: leave-one-subject-out, and the NinaPro
   convention of training on repetitions 1, 3, 4, 6 and testing on 2 and 5.
   Expect accuracy well below the simulator's 95.6%; that drop is the finding,
   not a bug. Write the results to `artifacts/model_comparison_ninapro.json`.
4. Tests in `services/research/tests/test_ninapro.py`. Build a tiny fake `.mat`
   file inside the test with `scipy.io.savemat` (scipy is already a
   dependency). Never use real data in tests or in the repo.

**Done when:** the loader tests pass, `compare --source ninapro` writes a JSON
artifact with both evaluations, and `export --source ninapro` produces a
`decoder.onnx` with matching `decoder.json`.

### Task 2: Replay real recordings in the Live screen (visual)

1. Add `--source ninapro --subject S<n>` to `neurogrip/replay.py`. It should
   write the same bundle format (int16 samples plus a JSON manifest, one
   segment per gesture), cut from one subject's real recording. That subject
   must be **held out** of training.
   - Add `"source"` and `"subject"` to the manifest.
   - Check that `VOLTS_PER_COUNT` doesn't clip real peaks, and change it if it
     does.
   - The manifest's `muscleGroups` weights come from the simulator's anatomy
     model and are wrong for real electrodes. Mark them as such, or omit them
     and have the UI handle their absence.
2. **Live screen** (`apps/web/src/screens/Live.tsx`, `apps/web/src/sources/replaySource.ts`):
   - A source badge reading "Recorded · NinaPro DB2 · subject S12" or "Simulated".
   - A strip under the oscilloscope showing what was **actually performed**
     (from the recording's labels) next to what was **decoded**, with a running
     correct/incorrect timeline. Real data has ground truth; showing it is the
     point.
3. Keep `npm test` green. `packages/core/test/artifacts.test.ts` checks
   `decoder.json` and the replay manifest against `FEATURE_SPEC_VERSION`.

**Done when:** `npm run dev:web` plays a real held-out recording, the badge says
so, and the performed-versus-decoded strip updates live.

### Task 3: A Results screen charting the real artifacts (visual)

Add a second screen reading the JSON files in `artifacts/`, copied into
`apps/web/public/results/` by `scripts/build-assets.mjs`:

- accuracy per model, **simulated and NinaPro side by side**, with LOSO spread
- a confusion matrix per gesture (NinaPro)
- latency (`artifacts/decoder.json`, `artifacts/browser_latency.json`)
- time to useful motion (`artifacts/ttum.json`)

Add simple tab navigation in `apps/web/src/App.tsx`, which currently renders
Live only. Follow the design rules in §3. Charts need text alternatives and
must work at phone width.

---

## 6. Phase 2 (Cartography): status and an honest note

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

Cartography should come after Tasks 1–3. Real data matters more for the
coursework, and babble on real data would need its own recording session.

## 7. Later phases (summary only)

- **Phase 3:** attribute each error to physiology, user behaviour, model
  capacity or drift. Validate with injected synthetic faults of each type.
  Then add forearm-anatomy coaching.
- **Phase 4:** clinician screens (Evaluate, Robustness, Latency, Records). The
  amputee stratum needs NinaPro DB3 (amputees, 2 kHz, also registration) and
  must be reported separately with sample-size caveats.
- **Phase 5:** Azure deployment and patent-disclosure drafts. Per the design
  spec, keep the repo private and draft the disclosures before anything goes
  public.

## 8. Working with an AI assistant on this

- Paste this file, `CLAUDE.md`, and the specific files the task names. The
  assistant can't see the repo, so give it the real file contents rather than
  a description of them.
- Ask for one task at a time, tests first, and run the tests yourself before
  accepting the code.
- If it suggests sorting feature names, loosening the conformance tolerance,
  committing data, or quoting a simulated number as a real one, reject it.
  Those are the four mistakes this repo is built to prevent.
