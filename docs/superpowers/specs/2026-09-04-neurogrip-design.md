# NeuroGrip — Design & Implementation Plan

**Project:** Real-Time Hand Gesture Control Interface for Prosthetic Limb Actuation
**Team:** Harsh Bavaskar · Anisa D'Souza · Shruti Shanklesha
**Course:** BTECH CAP 501 — Project Life Cycle Management, ATLAS SkillTech / uGDX
**Date:** 2026-09-04

---

## 1. Context

### Why this change is being made

`d:\NeuroGrip` currently holds ~250 lines of Python that stand in for a system that does not exist yet. The audit:

| File | State | Verdict |
|---|---|---|
| `src/pclm/features.py` | Real DSP — RMS, MAV, WL, ZC, SSC, Welch PSD | **Keep as seed.** Two defects: `_slope_sign_changes` compares an amplitude-unit threshold against a *product* of two differences (amplitude squared), so the same threshold gates differently at different signal scales; `_zero_crossings` compares `np.sign` values and `np.sign(0) == 0`, so a sample resting exactly at zero counts as crossing against both neighbours. AR coefficients are specified in the report (§5.3) but absent. |
| `src/pclm/signal_quality.py` | RMS noise-floor + amplitude + non-finite checks | **Keep.** Clean and correct. |
| `src/pclm/api.py` | 3 FastAPI endpoints | **Rewrite.** `/predict` measures latency into `_elapsed_ms` then discards it (line 47) — the report's headline NFR-2 metric is computed and thrown away. |
| `src/pclm/model.py` | `PlaceholderGestureModel` | **Delete.** `int(abs(np.sum(vector) * 1000)) % 9` is a hash function, not a classifier. |
| `src/pclm/schemas.py` | Pydantic request/response | Rewrite against the new contract. |
| — | No windowing (report specifies 200 ms / 100 ms hop) | Missing |
| — | No dataset loader, no training, no LOSO, no ONNX, no CI, no UI | Missing |

So: no UI, no ML, no data. The report (`NeuroGrip.pdf`) is a strong plan for a *coursework classifier*. The goal here is larger — a consumer-oriented product with a genuinely novel, patentable technical core.

### Intended outcome

A dual-mode application (prosthesis wearer + clinician) whose ML core is novel enough to support **3–4 independent utility-patent claims** and **4 design-patent GUI figures**, running on-device in the browser, hosted on Azure, fully demoable with zero dataset downloads, and documented well enough that a stranger can operate it from `CLAUDE.md` alone.

### Decisions locked with the user

| Axis | Decision |
|---|---|
| Data | Biophysical sEMG **simulator** + auto-downloadable public dataset; NinaPro `.mat` loader ships ready as a drop-in |
| Inference | **On-device** (ONNX Runtime Web, Web Worker); Azure hosts the bundle; Python research API on Azure for slow-path jobs |
| Novelty | **All four pillars** |
| Users | **Dual-mode** wearer + clinician |
| Visual spine | **Clinical Instrument, light-first**; real dark counterpart, not an inversion |

---

## 2. Prior art — what is already taken

Researched before designing, so the novelty claims are grounded rather than assumed.

| Idea | Status |
|---|---|
| Confidence-threshold abstention | Prior art, well established |
| Showing users an LDA feature-space cloud | **Prior art** — arXiv 2505.09819 (May 2025), "The Reviewer": 3D LDA space + live cursor, 12 subjects, 10 sessions |
| Gesture selection by confusion-matrix ranking | Prior art — BioMed Eng OnLine 2015 (PNM index) |
| Separability Index / Repeatability Index | Prior art |
| Cross-user zero-shot decoding | Prior art — Meta, *Nature* 2025; Frontiers 2024 (93 %, 306 unseen users) |
| Error-augmentation user training | Prior art — arXiv 2309.07289 |
| **Risk-weighted evidence-accumulation progressive actuation** | **No prior art found** |
| **Error-attribution routing to distinct remedies** | **No prior art found** |
| **Gesture synthesis from an individual's reachable manifold** | **No prior art found** |

The seam: PMC7870859 finds **unpredictability**, not average accuracy, drives prosthesis abandonment. And arXiv 2505.09819's own stated future work is *"autonomous self-adaptation… transitioning users toward an overseer role."* That is the open gap this project fills.

**Thesis:** every prior system optimizes the *decoder*. NeuroGrip optimizes the **joint human–decoder policy**, treating the user's motor repertoire as a designable variable.

---

## 3. The four pillars

### Pillar 1 — Neuromotor Cartography & Gesture Synthesis (NMC)

> *Method for synthesizing a personalized gesture vocabulary from an individual's measured neuromuscular reachable set.*

1. **Babble protocol** — ~3 min of *unstructured* forearm exploration (not cued gestures), plus a few cued anchors.
2. **Synergy embedding** — NMF muscle-synergy decomposition (physiologically meaningful, and it feeds Pillar 4), then a 2-D embedding for display.
3. **Three fields fitted over the manifold:**
   - Reachability ρ(x) — kernel density + local covariance of repeats
   - Effort/fatigue E(x) — amplitude, time-to-reach, and median-frequency drop during sustained holds (the standard fatigue marker)
   - Separability potential S(x) — expected margin to already-placed prototypes
4. **Synthesis** — place *k* prototypes maximizing `Σ [w_s·S(xᵢ) + w_r·ρ(xᵢ) − w_e·E(xᵢ)]` subject to pairwise distance ≥ δ. Farthest-point init + projected gradient refinement.
5. **Teach-back** — a synthesized prototype is a coordinate, not a gesture the user knows. The Cartograph renders it as a target and the user learns to reach it under closed-loop feedback, then names it.
6. **Risk-weighted assignment** — Hungarian algorithm maps prototypes → prosthetic functions, weighted by confusion risk × functional criticality.

**Why novel:** prior art *selects* from a fixed catalogue. This *invents* coordinates in the user's own manifold that no catalogue contains, under effort and fatigue constraints, then teaches them back.

### Pillar 2 — Risk-Weighted Evidence Accumulation & Progressive Actuation (REAP)

> *Method for graded prosthetic actuation driven by accumulated per-class log-evidence against asymmetric, risk-derived decision boundaries.*

- Every 20 ms hop the decoder emits posteriors. Accumulate a multi-hypothesis race:
  `A_k(t) = Σ_τ [ log p_k(τ) − log max_{j≠k} p_j(τ) ]`
- Per-class boundary `θ_k = θ₀ + λ·R_k`, where `R_k` comes from a **functional risk matrix** — falsely committing to a power grip costs more than falsely opening a hand.
- **Progressive actuation:** commitment `c(t) = clamp(A_k*/θ_k*, 0, 1)`. Motion *begins* at c > 0.2, stays **reversible** until c = 1, latches at c ≥ 1, retracts if evidence flips.
- **Urgency:** collapsing bounds so the system never hangs — it falls back to `rest` (safe state), never a forced guess.
- **New metrics this produces:** *Time-to-Useful-Motion* (TTUM) and *Cost-Weighted Error Rate* (CWER), reported alongside accuracy and P95 latency.

**Technical effect:** useful motion starts tens of ms before a threshold classifier would fire, *while high-cost errors are required to clear a higher bar.* Borrows the drift-diffusion decision model from neuroscience and inverts it — the machine decides the way a brain does.

### Pillar 3 — Error-Attribution Router (EAR) — *strongest single claim*

> *Method for allocating corrective action in a myoelectric interface by decomposing decoding error into physiological, behavioral, model-capacity, and drift components and routing each to a distinct remedy.*

Per misclassification, four scores → normalized to a simplex:

| Component | Estimator | Routed remedy |
|---|---|---|
| **P** physiological | Bhattacharyya overlap of the two classes' *tightest-repeat* clusters — if even the user's cleanest repeats overlap, it's anatomy | Vocabulary surgery (re-run Pillar 1 for that prototype) |
| **B** behavioral | Mahalanobis distance from the user's own prototype centroid, normalized by their historical within-class spread | Targeted drill, with Pillar 4 supplying the instruction |
| **M** model capacity | Does a stronger reference model (k-NN over all data / the LSTM) get it right when the deployed model does not? | Model adaptation / architecture swap |
| **D** drift | CUSUM on centroid displacement + median-frequency shift (fatigue) | Recalibration / adaptive re-centering |

This is the "overseer role" made concrete: the system diagnoses *whose fault* an error was and acts accordingly, instead of blindly retraining.

### Pillar 4 — Anatomical Attribution & Prescriptive Coaching (AAC)

> *Back-projection of feature attributions onto an anatomical channel–time map, converted into an executable motor instruction.*

- SHAP / integrated gradients → attributions per (channel × feature × time sub-window)
- Back-project channel index → forearm anatomical position via electrode-ring geometry → an animated **spatiotemporal attribution field** over a forearm cross-section
- A rule grammar converts it to plain language: *"start your index-finger press about 60 ms earlier, before you tighten your wrist"* — not *"SHAP value 0.31."*

This is what makes the product **consumer-grade** rather than a research notebook.

### Design-patent targets (ornamental GUI)

Four distinctive visual objects, drafted as design-patent figures:

1. **The Cartograph** — manifold plate: reachability contours, effort shown by *hatching density* (never gradient), synthesized prototype markers
2. **The Commitment Bar** — segmented evidence race with asymmetric per-class boundary marks
3. **The Attribution Simplex** — quaternary plot placing each error among P / B / M / D
4. **The Forearm Attribution Ring** — anatomical cross-section carrying the spatiotemporal field

---

## 4. Architecture

```
neurogrip/
  apps/web/              React 19 + TS + Vite        → Azure Static Web Apps
  packages/core/         TS: DSP, features, evidence accumulator, attribution (pure, tested)
  packages/viz/          TS: the four signature visualizations + primitives
  services/research/     Python: datasets, training, LOSO, ONNX export, SHAP, benchmarks
                                                     → Azure Container Apps
  models/                exported .onnx + metadata
  docs/                  CLAUDE.md, walkthrough, model card, data card, patent disclosures
```

### The single highest-risk constraint

**Python and TypeScript must compute byte-comparable features**, or the ONNX model sees a different input distribution at inference than at training and silently degrades.

Mitigation, built **first**, before any model: a **golden-vector conformance suite**. Python emits fixture JSON (input window → expected feature vector); a TS test asserts agreement at `rtol=1e-5`. Any drift fails CI. This is non-negotiable and is the reason Phase 0 exists.

### Real-time loop (browser)

```
EmgSource (simulator | dataset replay | future WebSerial/WebSocket device)
  → ring buffer  [Web Worker]
  → window 200 ms, hop 20 ms
  → signal-quality gate
  → feature extraction
  → ONNX Runtime Web (WASM + SIMD + threads, INT8)
  → posteriors
  → postMessage (small structs only — raw EMG never leaves the worker)  ≤ 50 Hz
  → [main] evidence accumulator → progressive actuation → render
```

**Documented deviation from the report:** hop is **20 ms, not 100 ms**. Progressive actuation is meaningless at 10 Hz decision rate. Cost is 5× the inference calls, but each is < 2 ms so the 10 ms P95 budget still holds. This deviation gets written up as an engineering finding, not hidden.

**Second documented constraint:** NinaPro sampling rates differ sharply — DB1 (Ottobock) 100 Hz, DB2 (Delsys Trigno) 2 kHz, DB5 (double Myo) 200 Hz. The pipeline resamples to a common 200 Hz for cross-database work, which makes a 200 ms window only 40 samples and coarsens PSD. The simulator runs at 2 kHz so the full-fidelity path is still exercised and benchmarked.

### Rendering

- `webgl-plot` for multi-channel oscilloscope lanes (hundreds of updates/sec)
- Custom Canvas2D + D3 scales for the four signature visualizations (20–60 Hz — comfortably within Canvas2D)
- `motion` for micro-interactions; every animation gated on `prefers-reduced-motion`

### Models (Python)

Majority-class baseline · **SVM-RBF** on handcrafted features (Harsh) · **LSTM** on windowed raw (Anisa) · **TCN** added as the likely deployment candidate — parallel, not sequential, so far better latency than the LSTM; the report already anticipates the LSTM failing the gate. Optuna tuning, MLflow tracking, LOSO CV, ONNX export, INT8 quantization.

### Evaluation (Shruti)

LOSO · amputee stratum **reported separately as its own headline block** (ER-2) · electrode-shift degradation curve · SHAP · ECE + reliability diagram · plus the two new metrics **TTUM** and **CWER**.

---

## 5. Design system & UI

Nothing gets built before tokens, type scale, spacing scale and the icon construction sheet exist.

- **Type** — one text face + one tabular-numeral monospace. Every number tabular, always.
- **Color** — paper `#FAF9F6`, ink `#16161A`, rule `#E2E0DA`, one signal accent, semantic ok/warn/stop. All ≥ 4.5:1 on paper. Dark counterpart designed as its own palette. Tokens on bare `:root`, redefined under `@media (prefers-color-scheme: dark) :root:not([data-theme="light"])` and again under `:root[data-theme="dark"]`.
- **No gradients, no glows, no shadows** beyond a 1 px hairline. Depth from rules and spacing only.
- **Icons** — ~30, hand-drawn on a 24 px grid, 1.5 px stroke, square caps, geometric construction, documented on a construction sheet. No icon library.
- **Density** — comfortable (wearer) / compact (clinician) via one token swap.
- **Responsive** — every visualization declares a minimum viable form and a defined degradation path (e.g. the Cartograph drops contour labels below 480 px). The body never scrolls horizontally; wide tables get their own `overflow-x:auto` container.
- **Motion** — 120–200 ms, one easing family, only where it carries information (evidence accumulating, a prototype landing, a value changing). Nothing decorative.
- **A11y** — full keyboard nav, visible focus, ARIA on every visualization plus a text-equivalent table behind a toggle, reduced-motion, 4.5:1 floor.

### Screens

**Wearer mode**
1. **Today** — is my hand working well right now? Grip streak, drift warning, one prescribed drill.
2. **Cartography** — babble session; watch your manifold get drawn; accept/reject synthesized gestures.
3. **Live** — oscilloscope + Commitment Bar + virtual hand. The core real-time experience.
4. **Train** — router-prescribed drills with the Attribution Ring giving the coaching.
5. **My Hand** — gesture ↔ function assignment editor with the risk matrix made tangible.

**Clinician mode**
6. **Evaluate** — LOSO, per-gesture breakdown, amputee stratum as its own block, confusion matrix, reliability diagram, ECE.
7. **Attribution** — the simplex, error browser, SHAP.
8. **Robustness** — electrode-shift curve, drift monitor, fatigue.
9. **Latency** — TTUM and P95 distributions, on-device vs. reference.
10. **Records** — model card, data card, ethics review, data lineage.

Persistent **safety banner**: *"Research prototype — not a medical device"* (ER-4).

---

## 6. Phases

| Phase | Deliverable | Gate |
|---|---|---|
| **0 — Foundations** | Monorepo, design tokens, icon sheet, biophysical sEMG simulator (MUAP summation, fatigue, electrode shift, crosstalk), **golden-vector conformance harness** | TS features match Python at rtol 1e-5 |
| **1 — Vertical slice** | simulator → features → trained SVM → ONNX → browser inference → **Live** screen with Commitment Bar | **First demoable artifact.** P95 < 10 ms measured on-device |
| **2 — Cartography** | Pillar 1 end to end: babble, manifold, synthesis, teach-back, assignment | A synthesized gesture is learnable and beats a catalogue gesture on separability |
| **3 — Router & coaching** | Pillars 3 + 4: attribution simplex, forearm ring, prescription grammar, drills | Router routes correctly on injected synthetic faults of each type |
| **4 — Clinician mode** | Screens 6–10, LOSO, amputee stratum, SHAP, robustness, records | Report-grade numbers reproducible from one command |
| **5 — Ship** | Azure deploy, `CLAUDE.md`, walkthrough, patent-disclosure drafts | Stranger can operate the app from docs alone |

---

## 7. Files

**Delete:** `src/pclm/model.py` (a hash, not a model)
**Port + fix:** `src/pclm/features.py` (fix `_slope_sign_changes`; add AR coefficients; add windowing) → becomes both `services/research/neurogrip/features.py` and `packages/core/src/features.ts`
**Port as-is:** `src/pclm/signal_quality.py`
**Rewrite:** `src/pclm/api.py`, `src/pclm/schemas.py` → `services/research/` (training/eval API + future hardware bridge; *not* the real-time path)
**Keep:** all three test files as the seed of the Python suite

Everything else is new.

---

## 8. Skills to invoke during implementation

Per the request to find tooling that raises quality and pace:

| Skill | Where |
|---|---|
| `superpowers:writing-plans` | Immediately after this design is approved |
| `superpowers:test-driven-development` | Every core algorithm — the four pillars are all testable in pure functions |
| `frontend-design` + `taste-skill:minimalist-skill` | The design system. `minimalist-skill` matches the brief exactly: flat, no gradients, no heavy shadows |
| `dataviz` | **Before** the first line of any chart code |
| `artifact-diagramming` | Architecture and pipeline diagrams for the docs |
| `modern-web-guidance` | Current web-platform practice for the worker/WASM/OffscreenCanvas path |
| `huggingface-skills:hf-cli`, `huggingface-datasets` | Locating and pulling the auto-downloadable EMG dataset |
| `azure:azure-deploy`, `azure:azure-app-onboard` | Static Web Apps + Container Apps deployment on the student plan |
| `superpowers:verification-before-completion` | Before any "done" claim |

---

## 9. Verification

**Per phase, before it is called complete:**

- `pytest` green in `services/research/`; `vitest` green in `packages/`
- **Conformance:** `pnpm test:conformance` — TS features vs Python golden vectors, rtol 1e-5. A failure here blocks everything downstream.
- **Latency:** `pnpm bench:latency` runs 1 000 windows through the real browser ONNX path via Playwright and asserts P95 ≤ 10 ms; also reports TTUM
- **Router correctness:** inject synthetic faults of each of the four types (forced class overlap, jittered user execution, deliberately under-capacity model, injected drift) and assert the simplex points at the right corner
- **Visual:** Playwright screenshots at 360 / 768 / 1280 / 1920 px in both themes; assert zero horizontal body overflow and 4.5:1 contrast via axe
- **A11y:** axe-core clean; full keyboard traversal of every screen
- **End-to-end demo:** `pnpm dev` → open app → run a babble session → accept a synthesized gesture → drive the virtual hand → trigger a deliberate misclassification → confirm the router prescribes the matching drill

**Final:** deployed Azure URL loads, runs inference on-device with the network tab showing no EMG leaving the browser, and `CLAUDE.md` walks a newcomer through all of it.

---

## 10. Honest risks

1. **Scope.** This is a large build. Phasing is real: Phase 1 alone is a complete, defensible, demoable deliverable that satisfies the report's committed Phase-1 scope. Everything after it compounds value but nothing after it is load-bearing for the coursework.
2. **Patentability is a claim, not a fact.** My searches found no prior art for three of the four pillars, but a professional prior-art search and a patent attorney are required before filing. I will draft disclosures, not applications. Note also that novelty is destroyed by public disclosure in many jurisdictions — **the repo should stay private and the report should stay unpublished until a provisional is filed.** Flagging this now because the timing matters more than the drafting.
3. **Synthesized gestures may be unlearnable.** Pillar 1 assumes a user can learn to hit a coordinate they have no name for. Phase 2's gate tests exactly this; if it fails, Pillar 1 degrades gracefully to catalogue selection with effort/fatigue weighting — still novel, less so.
4. **Amputee data is thin.** DB5 is 10 subjects. Every amputee-stratum number ships with confidence intervals and an explicit sample-size caveat, per the report's own ER-2.
