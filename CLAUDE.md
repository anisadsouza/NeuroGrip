# NeuroGrip

Real-time sEMG hand-gesture decoding for prosthetic control.
Harsh Bavaskar · Anisa D'Souza · Shruti Shanklesha — BTECH CAP 501.

**Research prototype. Not a medical device. Not validated for clinical use.**

## What this is

A dual-mode application (prosthesis wearer + clinician) built around four novel
mechanisms. See `docs/superpowers/specs/2026-09-04-neurogrip-design.md` for the
full design and the prior-art analysis behind each claim.

1. **Neuromotor Cartography** — map a user's reachable muscle-activation
   manifold, then synthesise gesture prototypes optimised for separability,
   repeatability and effort, and teach them back.
2. **Risk-weighted evidence accumulation** — the hand begins moving reversibly
   as evidence accrues and latches only at full commitment; high-cost gestures
   must clear a higher evidence bar than cheap ones.
3. **Error-attribution router** — decompose each error into physiological,
   behavioural, model-capacity and drift components, and route each to a
   different remedy.
4. **Anatomical attribution** — back-project feature attributions onto forearm
   anatomy and convert them into plain-language motor coaching.

## Layout

| Path | Holds |
| --- | --- |
| `services/research/` | Python: simulator, features, training, evaluation |
| `packages/core/` | TypeScript: the same DSP and features, for the browser |
| `fixtures/conformance/` | Golden vectors pinning the two implementations together |
| `docs/superpowers/specs/` | Design documents |
| `docs/superpowers/plans/` | Implementation plans |

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
4. `*_rms`, `*_mav`, `*_wl` — a mismatch here means float32 versus float64
   accumulation. Python casts input to float32 in `_as_channel_matrix`, then
   computes in float64.

## The feature set

Seventeen features per channel, in this fixed order:

`rms · mav · wl · zc · ssc · ar1-ar4 · psd1-psd6 · mdf · mnf`

Ordering is **channel-major and never sorted**. `sorted()` places `ch10_rms`
before `ch2_rms`; NinaPro DB2 has twelve channels, so lexicographic ordering
would silently permute the vector between an 8-channel and a 12-channel run.

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

Output is calibrated to real voltages: about 15 µV at rest (baseline noise) and
158–354 µV under contraction, matching published surface-EMG ranges.

Everything is seeded. Nothing in the simulator may touch global RNG state, or
tests stop being reproducible.

**This is why no dataset download is needed.** NinaPro registration is optional,
not blocking.

## Commands

```bash
# Python
pip install -e "./services/research[dev]"
python -m pytest services/research/tests -q

# TypeScript
npm install
npm test
npm run typecheck

# The gate
npm run test:conformance
```

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

## Status

**Phase 0 (Foundations) complete.** Simulator, feature pipeline in both
languages, conformance gate, CI.

Not yet built: trained models, ONNX export, the browser application, and all
four pillars' algorithms. Those are Phases 1–5, each with its own plan in
`docs/superpowers/plans/`.
