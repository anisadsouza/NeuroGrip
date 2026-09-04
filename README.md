# NeuroGrip

Real-time sEMG hand-gesture decoding for prosthetic control.

**Research prototype. Not a medical device. Not validated for clinical use.**

Harsh Bavaskar · Anisa D'Souza · Shruti Shanklesha
BTECH CAP 501, Project Life Cycle Management — ATLAS SkillTech / uGDX

## Status

Phase 0 (Foundations) complete: a biophysical sEMG simulator, a feature pipeline
implemented in both Python and TypeScript, and a conformance gate that pins them
together.

No trained model or user interface yet — those are Phase 1.

The project runs entirely on simulated data. **No dataset download or
registration is required** to build, test, or demo it.

## Quick start

```bash
# Python
pip install -e "./services/research[dev]"
python -m pytest services/research/tests -q

# TypeScript
npm install
npm test
```

## Layout

| Path | Holds |
| --- | --- |
| `services/research/` | Python: simulator, features, training, evaluation |
| `packages/core/` | TypeScript: the same DSP and features, for the browser |
| `fixtures/conformance/` | Golden vectors pinning the two implementations together |
| `docs/superpowers/` | Design specs and implementation plans |

See [CLAUDE.md](CLAUDE.md) for architecture, the conformance rule, and the full
command reference.
