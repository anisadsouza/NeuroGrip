# NeuroGrip Phase 0 — Foundations Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the deterministic signal-processing foundation for NeuroGrip — a biophysical sEMG simulator, a fully specified feature pipeline in Python, and a TypeScript port proven byte-comparable against Python golden vectors.

**Architecture:** An npm-workspaces monorepo. Python (`services/research`) owns training-time computation and generates golden fixture vectors. TypeScript (`packages/core`) owns inference-time computation in the browser and is tested against those fixtures at `rtol=1e-5`. Nothing downstream — no model, no UI — is built until that conformance gate is green, because a Python/TS feature mismatch silently corrupts every model that follows.

**Tech Stack:** Python 3.14 · numpy 2.5 · scipy 1.18 · pytest 9 · Node 24 · TypeScript 5 · Vitest · npm workspaces

**Spec:** `docs/superpowers/specs/2026-09-04-neurogrip-design.md`

## Global Constraints

- **Python** 3.14.2, existing venv at `.venv/` (`.venv/Scripts/python.exe` on Windows).
- **Node** 24.11.1, **npm** 11.13.0. Use npm workspaces — `pnpm` is not installed and must not be introduced.
- **No scipy in the conformance-critical path.** PSD, autocorrelation and AR estimation are hand-implemented in both languages against an explicit written algorithm. `scipy.signal.welch` is removed from the feature path entirely — porting its exact windowing, detrending and scaling to TS is a divergence risk with no upside.
- **Determinism is mandatory.** Every stochastic component takes an explicit integer seed. No calls to unseeded global RNG anywhere in `services/research/neurogrip/` or `packages/core/src/`.
- **Feature vector ordering is channel-major with a fixed intra-channel order**, never `sorted()`. (`sorted()` places `ch10_rms` before `ch2_rms`, silently permuting the vector at ≥10 channels — this is a live bug in the current `features.py`.)
- **`FEATURE_SPEC_VERSION = 1`** is embedded in every fixture file and asserted by the TS conformance test. Any change to the feature set or its ordering requires bumping it.
- **Conformance tolerance:** `rtol=1e-5`, `atol=1e-8`.
- **Window geometry:** 200 ms window, 20 ms hop (not the report's 100 ms — see spec §4).
- **Licence/safety copy:** any user-facing surface carries "Research prototype — not a medical device."
- **Commit style:** Conventional Commits. Every commit message ends with:
  `Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>`

---

## File Structure

```
d:\NeuroGrip\
  package.json                          npm workspaces root
  tsconfig.base.json
  vitest.workspace.ts
  packages/
    core/                               @neurogrip/core — pure TS, no DOM
      package.json  tsconfig.json
      src/
        spec.ts                         FEATURE_SPEC_VERSION, feature name table
        windowing.ts                    RingBuffer, frameWindows
        dsp.ts                          hann, fft, periodogram, autocorrelate, levinson
        features.ts                     extractFeatures, featureVector
        signalQuality.ts                checkSignalQuality
        index.ts
      test/
        dsp.test.ts  windowing.test.ts  signalQuality.test.ts
        conformance.test.ts             <-- THE GATE
  services/
    research/
      pyproject.toml
      neurogrip/
        __init__.py                     FEATURE_SPEC_VERSION
        spec.py                         feature name table (single source of truth)
        windowing.py
        dsp.py                          hann, rfft-based periodogram, autocorrelate, levinson
        features.py
        signal_quality.py
        simulator.py                    biophysical sEMG generator
        anatomy.py                      forearm geometry, muscles, electrode ring
        fixtures.py                     golden-vector generator CLI
      tests/
        test_dsp.py  test_windowing.py  test_features.py
        test_signal_quality.py  test_simulator.py  test_spec.py
  fixtures/conformance/                 generated JSON, committed to git
  docs/superpowers/{specs,plans}/
```

**Responsibility boundaries.** `spec.py`/`spec.ts` are the only place feature names and order are declared; `features` modules consume them and cannot reorder. `dsp` holds pure numeric primitives with no EMG semantics. `simulator.py` has no Python-side twin in TS and never needs one — it is training-time only.

**Deleted:** `src/pclm/model.py` (a hash function masquerading as a classifier), `src/pclm/api.py`, `src/pclm/schemas.py`. The API is rebuilt in Phase 1 against a different contract.

---

## Tasks

### Task 1: Repo restructure — move `pclm` into `services/research/neurogrip`

**Files:**
- Create: `services/research/pyproject.toml`, `services/research/neurogrip/__init__.py`
- Move: `src/pclm/features.py` → `services/research/neurogrip/features.py`
- Move: `src/pclm/signal_quality.py` → `services/research/neurogrip/signal_quality.py`
- Move: `tests/test_features.py`, `tests/test_signal_quality.py` → `services/research/tests/`
- Delete: `src/pclm/model.py`, `src/pclm/api.py`, `src/pclm/schemas.py`, `tests/test_api.py`, `src/pclm.egg-info/`, root `pyproject.toml`

**Interfaces:**
- Consumes: nothing (first task)
- Produces: importable package `neurogrip` exposing `neurogrip.features.extract_features` and `neurogrip.signal_quality.check_signal_quality` with unchanged behaviour; `neurogrip.__version__ = "0.2.0"`; `neurogrip.FEATURE_SPEC_VERSION = 1`

- [x] **Step 1: Confirm the suite is green before touching anything**

```bash
cd d:/NeuroGrip && ./.venv/Scripts/python.exe -m pytest -q
```
Expected: PASS. Record the test count — it must not drop except for the deliberately deleted `test_api.py`.

- [x] **Step 2: Move the keepers with `git mv` so history follows**

```bash
cd d:/NeuroGrip
mkdir -p services/research/neurogrip services/research/tests
git mv src/pclm/features.py services/research/neurogrip/features.py
git mv src/pclm/signal_quality.py services/research/neurogrip/signal_quality.py
git mv tests/test_features.py services/research/tests/test_features.py
git mv tests/test_signal_quality.py services/research/tests/test_signal_quality.py
git rm -q src/pclm/model.py src/pclm/api.py src/pclm/schemas.py src/pclm/__init__.py tests/test_api.py pyproject.toml
rm -rf src/pclm.egg-info src tests
```

- [x] **Step 3: Write `services/research/neurogrip/__init__.py`**

```python
"""NeuroGrip research toolkit: sEMG simulation, features, training, evaluation."""

__all__ = ["__version__", "FEATURE_SPEC_VERSION"]

__version__ = "0.2.0"

# Bump whenever the feature set, its ordering, or any DSP constant changes.
# packages/core/src/spec.ts must be bumped in lockstep and fixtures regenerated.
FEATURE_SPEC_VERSION = 1
```

- [x] **Step 4: Write `services/research/pyproject.toml`**

```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "neurogrip"
version = "0.2.0"
description = "NeuroGrip research toolkit: sEMG simulation, features, training, evaluation"
requires-python = ">=3.12"
dependencies = ["numpy>=2.0", "scipy>=1.13"]

[project.optional-dependencies]
dev = ["pytest>=8.0"]
ml = ["scikit-learn>=1.5", "onnxruntime>=1.20", "skl2onnx>=1.17", "onnx>=1.16"]

[tool.setuptools.packages.find]
where = ["."]
include = ["neurogrip*"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

- [x] **Step 5: Repoint imports in the moved tests**

Both moved test files import from `pclm.*`:

```bash
cd d:/NeuroGrip
sed -i 's/from pclm\./from neurogrip./g; s/import pclm\./import neurogrip./g' services/research/tests/*.py
grep -rn "pclm" services/research/ || echo "no pclm references remain"
```

- [x] **Step 6: Reinstall under the new name and run the suite**

```bash
cd d:/NeuroGrip
./.venv/Scripts/python.exe -m pip uninstall -y pclm
./.venv/Scripts/python.exe -m pip install -e "./services/research[dev]"
cd services/research && ../../.venv/Scripts/python.exe -m pytest -q
```
Expected: PASS at the Step 1 count minus `test_api.py`'s tests.

- [x] **Step 7: Commit**

Write the message to a file to avoid shell quoting problems, then use `-F`:

```bash
cd d:/NeuroGrip
printf '%s\n' \
  'refactor: restructure pclm into neurogrip research package' '' \
  'Removes the placeholder model (a hash function, not a classifier) plus the' \
  'API and schemas, which are rebuilt in Phase 1 against a different contract.' \
  'Keeps features.py and signal_quality.py, which are sound.' '' \
  'Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>' > /tmp/ng-msg.txt
git add -A && git commit -F /tmp/ng-msg.txt
```

---

### Task 2: `spec.py` — single source of truth for feature names and order

**Files:**
- Create: `services/research/neurogrip/spec.py`
- Test: `services/research/tests/test_spec.py`

**Interfaces:**
- Consumes: nothing
- Produces:
  - `PER_CHANNEL_FEATURES: tuple[str, ...]` — 17 names in canonical order
  - `AR_ORDER: int = 4`, `PSD_BINS: int = 6`
  - `feature_names(n_channels: int) -> tuple[str, ...]`
  - `feature_count(n_channels: int) -> int`

Why a fixed table: the existing `feature_vector()` sorts names lexicographically, placing `ch10_rms` before `ch2_rms`. At ten or more channels — NinaPro DB2 has twelve — the vector silently permutes between an 8-channel and a 12-channel run. An explicit table removes the failure mode entirely.

- [x] **Step 1: Write the failing test**

```python
# services/research/tests/test_spec.py
import pytest

from neurogrip.spec import (
    AR_ORDER,
    PER_CHANNEL_FEATURES,
    PSD_BINS,
    feature_count,
    feature_names,
)


def test_per_channel_table_is_the_documented_seventeen():
    assert PER_CHANNEL_FEATURES == (
        "rms", "mav", "wl", "zc", "ssc",
        "ar1", "ar2", "ar3", "ar4",
        "psd1", "psd2", "psd3", "psd4", "psd5", "psd6",
        "mdf", "mnf",
    )
    assert len(PER_CHANNEL_FEATURES) == 5 + AR_ORDER + PSD_BINS + 2


def test_feature_names_are_channel_major():
    names = feature_names(2)
    assert names[0] == "ch1_rms"
    assert names[16] == "ch1_mnf"
    assert names[17] == "ch2_rms"
    assert len(names) == 34


def test_ordering_is_numeric_not_lexicographic_at_ten_channels():
    """sorted() places ch10_rms before ch2_rms. This is the bug being fixed."""
    names = feature_names(12)
    assert names.index("ch2_rms") < names.index("ch10_rms")
    assert names != tuple(sorted(names))


def test_feature_count_matches_names():
    for n in (1, 2, 8, 12, 16):
        assert feature_count(n) == len(feature_names(n))


def test_rejects_non_positive_channel_counts():
    with pytest.raises(ValueError):
        feature_names(0)
    with pytest.raises(ValueError):
        feature_count(0)
```

- [x] **Step 2: Run it and watch it fail**

```bash
cd d:/NeuroGrip/services/research && ../../.venv/Scripts/python.exe -m pytest tests/test_spec.py -q
```
Expected: FAIL — `ModuleNotFoundError: No module named 'neurogrip.spec'`

- [x] **Step 3: Implement `spec.py`**

```python
"""Canonical feature specification.

The ONLY place the feature set and its ordering are declared.
`packages/core/src/spec.ts` mirrors this exactly and
`packages/core/test/conformance.test.ts` asserts they agree. Any change here
requires bumping `neurogrip.FEATURE_SPEC_VERSION` and regenerating fixtures.
"""

from __future__ import annotations

AR_ORDER = 4
PSD_BINS = 6

PER_CHANNEL_FEATURES: tuple[str, ...] = (
    # Time domain
    "rms",   # root mean square
    "mav",   # mean absolute value
    "wl",    # waveform length
    "zc",    # zero crossings
    "ssc",   # slope sign changes
    # Autoregressive coefficients (Levinson-Durbin over biased autocorrelation)
    *(f"ar{i + 1}" for i in range(AR_ORDER)),
    # Power spectral density, linearly spaced bands over the one-sided spectrum
    *(f"psd{i + 1}" for i in range(PSD_BINS)),
    # Spectral shape. Consumed downstream by the fatigue and drift estimators:
    # median frequency falls as a muscle fatigues.
    "mdf",
    "mnf",
)


def feature_names(n_channels: int) -> tuple[str, ...]:
    """Full ordering: channel-major, fixed intra-channel order.

    Deliberately NOT lexicographic. sorted() emits ch10_* before ch2_*, which
    silently permutes the vector for any recording with ten or more channels.
    """
    if n_channels < 1:
        raise ValueError("n_channels must be at least 1")
    return tuple(
        f"ch{channel}_{feature}"
        for channel in range(1, n_channels + 1)
        for feature in PER_CHANNEL_FEATURES
    )


def feature_count(n_channels: int) -> int:
    if n_channels < 1:
        raise ValueError("n_channels must be at least 1")
    return n_channels * len(PER_CHANNEL_FEATURES)
```

- [x] **Step 4: Run and confirm green**

```bash
cd d:/NeuroGrip/services/research && ../../.venv/Scripts/python.exe -m pytest tests/test_spec.py -q
```
Expected: 5 passed

- [x] **Step 5: Commit**

```bash
cd d:/NeuroGrip
printf '%s\n' \
  'feat: add canonical feature spec with channel-major ordering' '' \
  'Replaces lexicographic sorting, which permutes the feature vector at ten or' \
  'more channels (ch10_rms sorts before ch2_rms). NinaPro DB2 has twelve.' '' \
  'Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>' > /tmp/ng-msg.txt
git add services/research/neurogrip/spec.py services/research/tests/test_spec.py
git commit -F /tmp/ng-msg.txt
```

---

### Task 3: `dsp.py` — numeric primitives with no EMG semantics

**Files:**
- Create: `services/research/neurogrip/dsp.py`
- Test: `services/research/tests/test_dsp.py`

**Interfaces:**
- Consumes: nothing
- Produces:
  - `next_pow2(n: int) -> int`
  - `hann(n: int) -> np.ndarray` — symmetric
  - `periodogram(x, fs) -> tuple[np.ndarray, np.ndarray]` — `(freqs, psd)`, one-sided
  - `band_means(values: np.ndarray, n_bands: int) -> np.ndarray`
  - `autocorrelate(x: np.ndarray, max_lag: int) -> np.ndarray` — biased, length `max_lag + 1`
  - `levinson_durbin(r: np.ndarray, order: int) -> np.ndarray` — length `order`
  - `median_frequency(freqs, psd) -> float`, `mean_frequency(freqs, psd) -> float`

**These algorithms are the conformance contract.** They are specified exactly here because `packages/core/src/dsp.ts` must reproduce them. `scipy.signal.welch` is deliberately NOT used — its internal detrending, segment averaging and scaling are a porting hazard with no benefit at a single 200 ms window.

Definitions, normative:

- `next_pow2(n)` — smallest power of two ≥ `n`; `next_pow2(1) == 1`.
- `hann(N)` — symmetric: `w[i] = 0.5 - 0.5*cos(2*pi*i/(N-1))`, and `hann(1) == [1.0]`.
- `periodogram(x, fs)`:
  - `N = len(x)`, `w = hann(N)`, `xw = x * w`, `nfft = next_pow2(N)`
  - `X = rfft(xw, n=nfft)`, `U = sum(w**2)`
  - `psd = |X|**2 / (fs * U)`; multiply bins `1 .. nfft//2 - 1` by 2 (Nyquist bin and DC are not doubled)
  - `freqs = arange(nfft//2 + 1) * fs / nfft`
- `band_means(values, k)` — split `values` into `k` contiguous groups using numpy `array_split` semantics (the first `len % k` groups get `len//k + 1` elements, the rest `len//k`), return the mean of each. Empty groups yield `0.0`.
- `autocorrelate(x, max_lag)` — biased: `r[k] = sum(x[n] * x[n+k] for n in 0..N-1-k) / N`.
- `levinson_durbin(r, p)` — returns `a[1..p]` for the model `x[n] = -sum(a[i]*x[n-i]) + e[n]`. If `r[0] <= 0` or the prediction error reaches `<= 0`, return zeros of length `p`.
- `median_frequency` — the smallest `freqs[i]` where `cumsum(psd)[i] >= 0.5 * sum(psd)`; `0.0` if `sum(psd) <= 0`.
- `mean_frequency` — `sum(freqs * psd) / sum(psd)`; `0.0` if `sum(psd) <= 0`.

- [x] **Step 1: Write the failing test**

```python
# services/research/tests/test_dsp.py
import numpy as np
import pytest

from neurogrip.dsp import (
    autocorrelate,
    band_means,
    hann,
    levinson_durbin,
    mean_frequency,
    median_frequency,
    next_pow2,
    periodogram,
)


def test_next_pow2():
    assert next_pow2(1) == 1
    assert next_pow2(2) == 2
    assert next_pow2(3) == 4
    assert next_pow2(400) == 512


def test_hann_is_symmetric_and_zero_at_the_ends():
    w = hann(9)
    assert w[0] == pytest.approx(0.0, abs=1e-12)
    assert w[-1] == pytest.approx(0.0, abs=1e-12)
    assert w[4] == pytest.approx(1.0)
    assert np.allclose(w, w[::-1])
    assert np.array_equal(hann(1), np.array([1.0]))


def test_periodogram_locates_a_pure_tone():
    fs = 1000.0
    t = np.arange(512) / fs
    x = np.sin(2 * np.pi * 100.0 * t)
    freqs, psd = periodogram(x, fs)
    assert freqs[int(np.argmax(psd))] == pytest.approx(100.0, abs=fs / 512)
    assert len(freqs) == len(psd) == 512 // 2 + 1


def test_band_means_uses_array_split_semantics():
    # 7 items into 3 bands -> sizes 3, 2, 2
    values = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
    assert np.allclose(band_means(values, 3), [2.0, 4.5, 6.5])


def test_band_means_pads_with_zero_when_shorter_than_bands():
    assert np.allclose(band_means(np.array([5.0]), 3), [5.0, 0.0, 0.0])


def test_autocorrelate_is_biased_and_peaks_at_zero_lag():
    x = np.array([1.0, 2.0, 3.0, 4.0])
    r = autocorrelate(x, 2)
    assert len(r) == 3
    assert r[0] == pytest.approx(np.sum(x**2) / 4)
    assert r[0] >= r[1] >= r[2]


def test_levinson_recovers_a_known_ar2_process():
    rng = np.random.default_rng(7)
    n = 4000
    a1, a2 = -1.2, 0.5  # x[n] = 1.2 x[n-1] - 0.5 x[n-2] + e
    x = np.zeros(n)
    e = rng.standard_normal(n) * 0.1
    for i in range(2, n):
        x[i] = -a1 * x[i - 1] - a2 * x[i - 2] + e[i]
    coeffs = levinson_durbin(autocorrelate(x, 2), 2)
    assert coeffs[0] == pytest.approx(a1, abs=0.05)
    assert coeffs[1] == pytest.approx(a2, abs=0.05)


def test_levinson_returns_zeros_for_a_silent_signal():
    assert np.allclose(levinson_durbin(np.zeros(5), 4), np.zeros(4))


def test_frequency_moments_on_a_flat_spectrum():
    freqs = np.array([0.0, 10.0, 20.0, 30.0, 40.0])
    psd = np.ones(5)
    assert mean_frequency(freqs, psd) == pytest.approx(20.0)
    assert median_frequency(freqs, psd) == pytest.approx(20.0)


def test_frequency_moments_are_zero_for_no_power():
    freqs = np.array([0.0, 10.0, 20.0])
    psd = np.zeros(3)
    assert mean_frequency(freqs, psd) == 0.0
    assert median_frequency(freqs, psd) == 0.0
```

- [x] **Step 2: Run and watch it fail**

```bash
cd d:/NeuroGrip/services/research && ../../.venv/Scripts/python.exe -m pytest tests/test_dsp.py -q
```
Expected: FAIL — `ModuleNotFoundError: No module named 'neurogrip.dsp'`

- [x] **Step 3: Implement `dsp.py`**

```python
"""Numeric primitives for the feature pipeline.

Every function here is mirrored bit-for-bit in packages/core/src/dsp.ts.
scipy.signal.welch is deliberately avoided: its detrending, segment averaging
and scaling conventions are a porting hazard and buy nothing at a single
200 ms window. The algorithms below are the conformance contract.
"""

from __future__ import annotations

import numpy as np


def next_pow2(n: int) -> int:
    if n < 1:
        raise ValueError("n must be positive")
    power = 1
    while power < n:
        power *= 2
    return power


def hann(n: int) -> np.ndarray:
    """Symmetric Hann window. hann(1) is [1.0]."""
    if n < 1:
        raise ValueError("n must be positive")
    if n == 1:
        return np.ones(1, dtype=np.float64)
    i = np.arange(n, dtype=np.float64)
    return 0.5 - 0.5 * np.cos(2.0 * np.pi * i / (n - 1))


def periodogram(x: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]:
    """One-sided Hann-windowed periodogram, zero-padded to a power of two."""
    signal = np.asarray(x, dtype=np.float64)
    n = signal.size
    if n < 1:
        raise ValueError("signal must not be empty")

    window = hann(n)
    nfft = next_pow2(n)
    spectrum = np.fft.rfft(signal * window, n=nfft)
    scale = fs * float(np.sum(window**2))

    psd = np.abs(spectrum) ** 2 / scale if scale > 0 else np.zeros(spectrum.size)
    # Fold negative-frequency power into the positive bins, excluding DC and Nyquist.
    if psd.size > 2:
        psd[1:-1] *= 2.0

    freqs = np.arange(nfft // 2 + 1, dtype=np.float64) * fs / nfft
    return freqs, psd


def band_means(values: np.ndarray, n_bands: int) -> np.ndarray:
    """Mean of each of n_bands contiguous groups, numpy array_split semantics."""
    if n_bands < 1:
        raise ValueError("n_bands must be positive")
    data = np.asarray(values, dtype=np.float64)
    out = np.zeros(n_bands, dtype=np.float64)
    for index, chunk in enumerate(np.array_split(data, n_bands)):
        out[index] = float(np.mean(chunk)) if chunk.size else 0.0
    return out


def autocorrelate(x: np.ndarray, max_lag: int) -> np.ndarray:
    """Biased autocorrelation r[k] = sum(x[n] * x[n+k]) / N, k = 0..max_lag."""
    if max_lag < 0:
        raise ValueError("max_lag must not be negative")
    signal = np.asarray(x, dtype=np.float64)
    n = signal.size
    out = np.zeros(max_lag + 1, dtype=np.float64)
    for lag in range(min(max_lag, n - 1) + 1):
        out[lag] = float(np.dot(signal[: n - lag], signal[lag:])) / n
    return out


def levinson_durbin(r: np.ndarray, order: int) -> np.ndarray:
    """AR coefficients a[1..order] for x[n] = -sum(a[i] * x[n-i]) + e[n].

    Returns zeros when the signal carries no power or the recursion becomes
    numerically degenerate, which is the correct answer for a silent channel.
    """
    if order < 1:
        raise ValueError("order must be positive")
    acf = np.asarray(r, dtype=np.float64)
    if acf.size < order + 1 or acf[0] <= 0.0:
        return np.zeros(order, dtype=np.float64)

    a = np.zeros(order + 1, dtype=np.float64)
    a[0] = 1.0
    error = float(acf[0])

    for m in range(1, order + 1):
        acc = acf[m] + float(np.dot(a[1:m], acf[m - 1 : 0 : -1]))
        reflection = -acc / error
        previous = a[1:m].copy()
        a[m] = reflection
        if m > 1:
            a[1:m] = previous + reflection * previous[::-1]
        error *= 1.0 - reflection * reflection
        if error <= 0.0:
            return np.zeros(order, dtype=np.float64)

    return a[1:]


def median_frequency(freqs: np.ndarray, psd: np.ndarray) -> float:
    """Frequency at which cumulative power first reaches half the total."""
    power = np.asarray(psd, dtype=np.float64)
    total = float(np.sum(power))
    if total <= 0.0:
        return 0.0
    cumulative = np.cumsum(power)
    index = int(np.searchsorted(cumulative, 0.5 * total, side="left"))
    index = min(index, power.size - 1)
    return float(np.asarray(freqs, dtype=np.float64)[index])


def mean_frequency(freqs: np.ndarray, psd: np.ndarray) -> float:
    """Power-weighted mean frequency."""
    power = np.asarray(psd, dtype=np.float64)
    total = float(np.sum(power))
    if total <= 0.0:
        return 0.0
    return float(np.dot(np.asarray(freqs, dtype=np.float64), power) / total)
```

- [x] **Step 4: Run and confirm green**

```bash
cd d:/NeuroGrip/services/research && ../../.venv/Scripts/python.exe -m pytest tests/test_dsp.py -q
```
Expected: 9 passed

- [x] **Step 5: Commit**

```bash
cd d:/NeuroGrip
printf '%s\n' \
  'feat: add DSP primitives as the Python/TypeScript conformance contract' '' \
  'Hand-implements the periodogram rather than using scipy.signal.welch, whose' \
  'detrending and scaling conventions would be a porting hazard for the browser' \
  'runtime with no benefit at a single 200 ms window.' '' \
  'Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>' > /tmp/ng-msg.txt
git add services/research/neurogrip/dsp.py services/research/tests/test_dsp.py
git commit -F /tmp/ng-msg.txt
```

---

### Task 4: `windowing.py` — frame a stream into overlapping windows

**Files:**
- Create: `services/research/neurogrip/windowing.py`
- Test: `services/research/tests/test_windowing.py`

**Interfaces:**
- Consumes: nothing
- Produces:
  - `WindowConfig` dataclass — `window_ms: int = 200`, `hop_ms: int = 20`, `sampling_rate_hz: int = 2000`
  - `WindowConfig.window_samples -> int`, `WindowConfig.hop_samples -> int`
  - `frame_windows(channels: np.ndarray, config: WindowConfig) -> np.ndarray` — shape `(n_windows, n_channels, window_samples)`
  - `window_start_indices(n_samples: int, config: WindowConfig) -> list[int]`

Windows are non-padded: a trailing partial window is dropped, never zero-filled, because a zero-filled tail would produce features that no real contraction could generate and would pollute training.

- [x] **Step 1: Write the failing test**

```python
# services/research/tests/test_windowing.py
import numpy as np
import pytest

from neurogrip.windowing import WindowConfig, frame_windows, window_start_indices


def test_sample_counts_derive_from_milliseconds_and_rate():
    config = WindowConfig(window_ms=200, hop_ms=20, sampling_rate_hz=2000)
    assert config.window_samples == 400
    assert config.hop_samples == 40


def test_start_indices_are_hop_spaced_and_drop_the_partial_tail():
    config = WindowConfig(window_ms=100, hop_ms=50, sampling_rate_hz=100)
    # window = 10 samples, hop = 5 samples
    assert window_start_indices(22, config) == [0, 5, 10]
    assert window_start_indices(9, config) == []


def test_frame_windows_shape_and_content():
    config = WindowConfig(window_ms=100, hop_ms=50, sampling_rate_hz=100)
    channels = np.tile(np.arange(20, dtype=np.float32), (2, 1))
    windows = frame_windows(channels, config)
    assert windows.shape == (3, 2, 10)
    assert np.allclose(windows[0, 0], np.arange(10))
    assert np.allclose(windows[1, 0], np.arange(5, 15))


def test_frame_windows_returns_empty_when_signal_is_too_short():
    config = WindowConfig(window_ms=200, hop_ms=20, sampling_rate_hz=2000)
    windows = frame_windows(np.zeros((2, 100), dtype=np.float32), config)
    assert windows.shape == (0, 2, 400)


def test_rejects_non_2d_input():
    config = WindowConfig()
    with pytest.raises(ValueError):
        frame_windows(np.zeros(400, dtype=np.float32), config)


def test_rejects_non_positive_geometry():
    with pytest.raises(ValueError):
        WindowConfig(window_ms=0).window_samples
    with pytest.raises(ValueError):
        WindowConfig(hop_ms=0).hop_samples
```

- [x] **Step 2: Run and watch it fail**

```bash
cd d:/NeuroGrip/services/research && ../../.venv/Scripts/python.exe -m pytest tests/test_windowing.py -q
```
Expected: FAIL — `ModuleNotFoundError: No module named 'neurogrip.windowing'`

- [x] **Step 3: Implement `windowing.py`**

```python
"""Framing a continuous multi-channel stream into overlapping analysis windows.

Mirrored in packages/core/src/windowing.ts. The 20 ms hop (rather than the
100 ms in the project report) is what makes progressive actuation possible:
at a 10 Hz decision rate there is no evidence to accumulate between decisions.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class WindowConfig:
    window_ms: int = 200
    hop_ms: int = 20
    sampling_rate_hz: int = 2000

    @property
    def window_samples(self) -> int:
        if self.window_ms <= 0:
            raise ValueError("window_ms must be positive")
        return int(self.window_ms * self.sampling_rate_hz / 1000)

    @property
    def hop_samples(self) -> int:
        if self.hop_ms <= 0:
            raise ValueError("hop_ms must be positive")
        return int(self.hop_ms * self.sampling_rate_hz / 1000)


def window_start_indices(n_samples: int, config: WindowConfig) -> list[int]:
    """Start offsets of every complete window. A partial tail is dropped."""
    width = config.window_samples
    hop = config.hop_samples
    if n_samples < width:
        return []
    return list(range(0, n_samples - width + 1, hop))


def frame_windows(channels: np.ndarray, config: WindowConfig) -> np.ndarray:
    """Frame (n_channels, n_samples) into (n_windows, n_channels, window_samples).

    A trailing partial window is dropped rather than zero-padded: a zero-filled
    tail yields features no real contraction could produce and would pollute
    both training data and the drift statistics.
    """
    signal = np.asarray(channels, dtype=np.float32)
    if signal.ndim != 2:
        raise ValueError("EMG input must have shape (n_channels, n_samples)")

    n_channels, n_samples = signal.shape
    width = config.window_samples
    starts = window_start_indices(n_samples, config)
    if not starts:
        return np.zeros((0, n_channels, width), dtype=np.float32)

    return np.stack([signal[:, start : start + width] for start in starts])
```

- [x] **Step 4: Run and confirm green**

```bash
cd d:/NeuroGrip/services/research && ../../.venv/Scripts/python.exe -m pytest tests/test_windowing.py -q
```
Expected: 6 passed

- [x] **Step 5: Commit**

```bash
cd d:/NeuroGrip
printf '%s\n' \
  'feat: add windowing with a 20 ms hop for progressive actuation' '' \
  'Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>' > /tmp/ng-msg.txt
git add services/research/neurogrip/windowing.py services/research/tests/test_windowing.py
git commit -F /tmp/ng-msg.txt
```

---

### Task 5: Rewrite `features.py` against the spec, fixing the ZC and SSC defects

**Files:**
- Modify: `services/research/neurogrip/features.py` (full rewrite)
- Modify: `services/research/tests/test_features.py` (rewrite — the feature set changed)

**Interfaces:**
- Consumes: `neurogrip.spec.{PER_CHANNEL_FEATURES, AR_ORDER, PSD_BINS, feature_names}`, `neurogrip.dsp.*`
- Produces:
  - `FeatureConfig` dataclass — `sampling_rate_hz: int = 2000`, `zero_crossing_threshold: float = 1e-5`, `slope_sign_threshold: float = 1e-5`
  - `extract_features(channels: np.ndarray, config: FeatureConfig) -> dict[str, float]`
  - `feature_vector(features: dict[str, float], n_channels: int) -> np.ndarray` — float32, spec order

**Two defects in the current implementation are fixed here.**

*Zero crossings.* The current code tests `np.sign(previous) != np.sign(current)`. Because `np.sign(0.0) == 0`, a sample sitting exactly at zero registers a crossing against either neighbour, so a signal that merely touches zero is counted as crossing it — twice. The fix tests the product against zero: `x[n] * x[n+1] < 0` is true only for a genuine sign change.

*Slope sign changes.* The current code compares `previous_diff * next_diff >= threshold`. The product has units of amplitude squared while `threshold` is expressed in amplitude units, so the same threshold behaves completely differently at different signal scales — a 10 µV threshold gates almost nothing at 1 mV amplitudes and gates almost everything at 10 µV amplitudes. The fix separates the two concerns: the *sign* test runs on the product (`> 0`, dimensionless), and the *magnitude* test runs on the differences themselves (amplitude units, directly comparable to the threshold).

- [x] **Step 1: Write the failing test**

```python
# services/research/tests/test_features.py
import numpy as np
import pytest

from neurogrip.features import FeatureConfig, extract_features, feature_vector
from neurogrip.spec import feature_count, feature_names


def _config(**kwargs) -> FeatureConfig:
    return FeatureConfig(**kwargs)


def test_extracts_every_spec_feature_for_every_channel():
    channels = np.random.default_rng(0).standard_normal((3, 400)).astype(np.float32) * 0.01
    features = extract_features(channels, _config())
    assert set(features) == set(feature_names(3))
    assert len(features) == feature_count(3)


def test_time_domain_values_are_correct():
    channels = np.array([[3.0, -4.0, 3.0, -4.0]], dtype=np.float32)
    features = extract_features(channels, _config(zero_crossing_threshold=0.0))
    assert features["ch1_rms"] == pytest.approx(np.sqrt((9 + 16 + 9 + 16) / 4))
    assert features["ch1_mav"] == pytest.approx(3.5)
    assert features["ch1_wl"] == pytest.approx(7.0 + 7.0 + 7.0)


def test_a_sample_resting_exactly_at_zero_is_not_a_crossing():
    """np.sign(0) == 0 made the old implementation count this as two crossings."""
    channels = np.array([[1.0, 0.0, 1.0]], dtype=np.float32)
    features = extract_features(channels, _config(zero_crossing_threshold=0.0))
    assert features["ch1_zc"] == 0.0


def test_a_genuine_sign_change_is_counted_once():
    channels = np.array([[1.0, -1.0, 1.0]], dtype=np.float32)
    features = extract_features(channels, _config(zero_crossing_threshold=0.0))
    assert features["ch1_zc"] == 2.0


def test_zero_crossing_threshold_suppresses_tiny_wiggles():
    channels = np.array([[1e-9, -1e-9, 1e-9, -1e-9]], dtype=np.float32)
    features = extract_features(channels, _config(zero_crossing_threshold=1e-5))
    assert features["ch1_zc"] == 0.0


def test_slope_sign_threshold_is_scale_consistent():
    """The threshold is in amplitude units, so scaling the signal by k and the
    threshold by k must leave the SSC count unchanged. The old implementation
    compared the threshold against a squared product and failed this."""
    base = np.array([[0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0]], dtype=np.float32)
    small = extract_features(base * 1e-3, _config(slope_sign_threshold=1e-3 * 0.5))
    large = extract_features(base * 1e3, _config(slope_sign_threshold=1e3 * 0.5))
    assert small["ch1_ssc"] == large["ch1_ssc"]
    assert small["ch1_ssc"] > 0


def test_silent_channel_yields_zeroed_ar_and_spectral_features():
    channels = np.zeros((1, 400), dtype=np.float32)
    features = extract_features(channels, _config())
    for name in ("ch1_ar1", "ch1_ar2", "ch1_ar3", "ch1_ar4", "ch1_mdf", "ch1_mnf"):
        assert features[name] == 0.0


def test_feature_vector_follows_spec_order_and_is_float32():
    channels = np.random.default_rng(1).standard_normal((2, 400)).astype(np.float32) * 0.01
    features = extract_features(channels, _config())
    vector = feature_vector(features, 2)
    assert vector.dtype == np.float32
    assert vector.shape == (feature_count(2),)
    assert vector[0] == pytest.approx(features["ch1_rms"], rel=1e-6)
    assert vector[17] == pytest.approx(features["ch2_rms"], rel=1e-6)


def test_rejects_malformed_input():
    with pytest.raises(ValueError):
        extract_features(np.zeros(400, dtype=np.float32), _config())
    with pytest.raises(ValueError):
        extract_features(np.zeros((1, 2), dtype=np.float32), _config())
```

- [x] **Step 2: Run and watch it fail**

```bash
cd d:/NeuroGrip/services/research && ../../.venv/Scripts/python.exe -m pytest tests/test_features.py -q
```
Expected: FAIL — the new spec features (`ar*`, `mdf`, `mnf`) do not exist yet and `feature_vector` has the wrong signature.

- [x] **Step 3: Rewrite `features.py`**

```python
"""Handcrafted sEMG feature extraction.

Mirrored bit-for-bit in packages/core/src/features.ts and pinned by the golden
vectors in fixtures/conformance/. Feature names and ordering come from
neurogrip.spec and must never be re-derived here.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from neurogrip import dsp
from neurogrip.spec import AR_ORDER, PSD_BINS, feature_names


@dataclass(frozen=True)
class FeatureConfig:
    sampling_rate_hz: int = 2000
    zero_crossing_threshold: float = 1e-5
    slope_sign_threshold: float = 1e-5


def extract_features(channels: np.ndarray, config: FeatureConfig) -> dict[str, float]:
    """Extract the 17 spec features per channel from one analysis window.

    Expected input shape is (n_channels, n_samples).
    """
    signal = _as_channel_matrix(channels)
    features: dict[str, float] = {}

    for index, values in enumerate(signal):
        prefix = f"ch{index + 1}"
        window = np.asarray(values, dtype=np.float64)

        features[f"{prefix}_rms"] = float(np.sqrt(np.mean(np.square(window))))
        features[f"{prefix}_mav"] = float(np.mean(np.abs(window)))
        features[f"{prefix}_wl"] = float(np.sum(np.abs(np.diff(window))))
        features[f"{prefix}_zc"] = float(
            _zero_crossings(window, config.zero_crossing_threshold)
        )
        features[f"{prefix}_ssc"] = float(
            _slope_sign_changes(window, config.slope_sign_threshold)
        )

        coefficients = dsp.levinson_durbin(dsp.autocorrelate(window, AR_ORDER), AR_ORDER)
        for order, coefficient in enumerate(coefficients, start=1):
            features[f"{prefix}_ar{order}"] = float(coefficient)

        freqs, psd = dsp.periodogram(window, float(config.sampling_rate_hz))
        for band, power in enumerate(dsp.band_means(psd, PSD_BINS), start=1):
            features[f"{prefix}_psd{band}"] = float(power)

        features[f"{prefix}_mdf"] = dsp.median_frequency(freqs, psd)
        features[f"{prefix}_mnf"] = dsp.mean_frequency(freqs, psd)

    return features


def feature_vector(features: dict[str, float], n_channels: int) -> np.ndarray:
    """Numeric vector in canonical spec order.

    Never sorts. sorted() places ch10_* before ch2_*, silently permuting the
    vector for any recording with ten or more channels.
    """
    return np.array(
        [features[name] for name in feature_names(n_channels)], dtype=np.float32
    )


def _as_channel_matrix(channels: np.ndarray) -> np.ndarray:
    signal = np.asarray(channels, dtype=np.float32)
    if signal.ndim != 2:
        raise ValueError("EMG input must have shape (n_channels, n_samples).")
    if signal.shape[0] < 1 or signal.shape[1] < 3:
        raise ValueError("EMG input needs at least one channel and three samples.")
    return signal


def _zero_crossings(values: np.ndarray, threshold: float) -> int:
    """Count genuine sign changes exceeding an amplitude threshold.

    Uses a strict product-negative test rather than comparing np.sign values:
    np.sign(0.0) is 0, so a sample resting exactly at zero would otherwise be
    counted as crossing against both of its neighbours.
    """
    previous = values[:-1]
    current = values[1:]
    crossed = (previous * current) < 0.0
    large_enough = np.abs(current - previous) >= threshold
    return int(np.count_nonzero(crossed & large_enough))


def _slope_sign_changes(values: np.ndarray, threshold: float) -> int:
    """Count direction reversals whose limbs exceed an amplitude threshold.

    The sign test runs on the product of the two differences (dimensionless);
    the magnitude test runs on the differences themselves, which share units
    with the threshold. Comparing the threshold directly against the product,
    as the previous implementation did, is dimensionally inconsistent: the
    product carries amplitude-squared units, so the same threshold gates almost
    nothing at millivolt amplitudes and almost everything at microvolt ones.
    """
    if values.size < 3:
        return 0
    previous_diff = values[1:-1] - values[:-2]
    next_diff = values[1:-1] - values[2:]
    reversed_direction = (previous_diff * next_diff) > 0.0
    large_enough = (
        np.maximum(np.abs(previous_diff), np.abs(next_diff)) >= threshold
    )
    return int(np.count_nonzero(reversed_direction & large_enough))
```

- [x] **Step 4: Run and confirm green**

```bash
cd d:/NeuroGrip/services/research && ../../.venv/Scripts/python.exe -m pytest tests/test_features.py -q
```
Expected: 9 passed

- [x] **Step 5: Commit**

```bash
cd d:/NeuroGrip
printf '%s\n' \
  'fix: correct zero-crossing and slope-sign-change definitions' '' \
  'ZC tested np.sign inequality, and np.sign(0) == 0, so a sample resting at' \
  'exactly zero counted as crossing against both neighbours. SSC compared an' \
  'amplitude-unit threshold against an amplitude-squared product, making the' \
  'threshold behave differently at different signal scales.' '' \
  'Also adds AR coefficients (specified in the project report but absent) and' \
  'median/mean frequency, which the fatigue and drift estimators consume.' '' \
  'Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>' > /tmp/ng-msg.txt
git add services/research/neurogrip/features.py services/research/tests/test_features.py
git commit -F /tmp/ng-msg.txt
```

---

### Task 6: `anatomy.py` — forearm geometry and volume conduction

**Files:**
- Create: `services/research/neurogrip/anatomy.py`
- Test: `services/research/tests/test_anatomy.py`

**Interfaces:**
- Consumes: nothing
- Produces:
  - `MUSCLES: tuple[Muscle, ...]` — six named forearm muscles with fixed angles
  - `Muscle` dataclass — `name: str`, `angle_rad: float`, `label: str`
  - `ForearmGeometry` dataclass — `skin_radius_mm: float = 40.0`, `muscle_radius_mm: float = 25.0`, `conduction_lambda_mm: float = 15.0`
  - `electrode_positions(n_electrodes, shift_rad, geometry) -> np.ndarray` shape `(n_electrodes, 2)`
  - `muscle_positions(geometry) -> np.ndarray` shape `(6, 2)`
  - `attenuation_matrix(n_electrodes, shift_rad, geometry) -> np.ndarray` shape `(n_electrodes, 6)`

This module carries the anatomical model that Pillar 4 later back-projects attributions onto, so the geometry is defined once, here, and reused by both the simulator and the forearm attribution ring.

- [x] **Step 1: Write the failing test**

```python
# services/research/tests/test_anatomy.py
import numpy as np
import pytest

from neurogrip.anatomy import (
    MUSCLES,
    ForearmGeometry,
    attenuation_matrix,
    electrode_positions,
    muscle_positions,
)


def test_six_named_muscles_at_distinct_angles():
    assert len(MUSCLES) == 6
    angles = [m.angle_rad for m in MUSCLES]
    assert len(set(angles)) == 6
    assert all(0.0 <= a < 2 * np.pi for a in angles)


def test_electrodes_sit_on_the_skin_circle():
    geometry = ForearmGeometry()
    positions = electrode_positions(8, 0.0, geometry)
    assert positions.shape == (8, 2)
    radii = np.linalg.norm(positions, axis=1)
    assert np.allclose(radii, geometry.skin_radius_mm)


def test_electrode_shift_rotates_the_ring():
    geometry = ForearmGeometry()
    unshifted = electrode_positions(8, 0.0, geometry)
    shifted = electrode_positions(8, np.pi / 4, geometry)
    # An eighth turn on an eight-electrode ring maps electrode 0 onto electrode 1.
    assert np.allclose(shifted[0], unshifted[1], atol=1e-9)


def test_muscles_sit_inside_the_skin_circle():
    geometry = ForearmGeometry()
    radii = np.linalg.norm(muscle_positions(geometry), axis=1)
    assert np.allclose(radii, geometry.muscle_radius_mm)
    assert geometry.muscle_radius_mm < geometry.skin_radius_mm


def test_attenuation_falls_off_with_distance_and_is_bounded():
    geometry = ForearmGeometry()
    matrix = attenuation_matrix(8, 0.0, geometry)
    assert matrix.shape == (8, 6)
    assert np.all(matrix > 0.0)
    assert np.all(matrix <= 1.0)


def test_nearest_electrode_sees_each_muscle_most_strongly():
    """This is what produces realistic crosstalk: every electrode sees every
    muscle, but the closest one dominates."""
    geometry = ForearmGeometry()
    matrix = attenuation_matrix(16, 0.0, geometry)
    electrodes = electrode_positions(16, 0.0, geometry)
    muscles = muscle_positions(geometry)
    for muscle_index in range(len(MUSCLES)):
        distances = np.linalg.norm(electrodes - muscles[muscle_index], axis=1)
        assert int(np.argmax(matrix[:, muscle_index])) == int(np.argmin(distances))


def test_rejects_bad_electrode_counts():
    with pytest.raises(ValueError):
        electrode_positions(0, 0.0, ForearmGeometry())
```

- [x] **Step 2: Run and watch it fail**

```bash
cd d:/NeuroGrip/services/research && ../../.venv/Scripts/python.exe -m pytest tests/test_anatomy.py -q
```
Expected: FAIL — `ModuleNotFoundError: No module named 'neurogrip.anatomy'`

- [x] **Step 3: Implement `anatomy.py`**

```python
"""Forearm cross-section geometry: muscles, electrode ring, volume conduction.

Defined once and shared by the simulator (which uses it to mix muscle sources
into electrode channels) and by the Pillar 4 attribution ring (which uses it to
back-project per-channel attributions onto anatomy). Angles are measured
counter-clockwise from the radial (thumb) side, viewed distally.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Muscle:
    name: str
    angle_rad: float
    label: str


# Six superficial forearm muscles at representative cross-sectional angles.
MUSCLES: tuple[Muscle, ...] = (
    Muscle("fcr", math.radians(30.0), "Flexor carpi radialis"),
    Muscle("fds", math.radians(80.0), "Flexor digitorum superficialis"),
    Muscle("fcu", math.radians(140.0), "Flexor carpi ulnaris"),
    Muscle("ecu", math.radians(210.0), "Extensor carpi ulnaris"),
    Muscle("edc", math.radians(265.0), "Extensor digitorum communis"),
    Muscle("ecr", math.radians(320.0), "Extensor carpi radialis"),
)


@dataclass(frozen=True)
class ForearmGeometry:
    skin_radius_mm: float = 40.0
    muscle_radius_mm: float = 25.0
    conduction_lambda_mm: float = 15.0


def _ring(radius_mm: float, angles: np.ndarray) -> np.ndarray:
    return np.stack([radius_mm * np.cos(angles), radius_mm * np.sin(angles)], axis=1)


def electrode_positions(
    n_electrodes: int, shift_rad: float, geometry: ForearmGeometry
) -> np.ndarray:
    """Evenly spaced electrodes on the skin circle, rotated by shift_rad.

    shift_rad models the donning variation that makes electrode shift the
    dominant real-world failure mode for myoelectric pattern recognition.
    """
    if n_electrodes < 1:
        raise ValueError("n_electrodes must be at least 1")
    angles = np.arange(n_electrodes, dtype=np.float64) * (2.0 * np.pi / n_electrodes)
    return _ring(geometry.skin_radius_mm, angles + shift_rad)


def muscle_positions(geometry: ForearmGeometry) -> np.ndarray:
    angles = np.array([m.angle_rad for m in MUSCLES], dtype=np.float64)
    return _ring(geometry.muscle_radius_mm, angles)


def attenuation_matrix(
    n_electrodes: int, shift_rad: float, geometry: ForearmGeometry
) -> np.ndarray:
    """Volume-conduction gain from each muscle to each electrode.

    Exponential decay with distance, exp(-d / lambda). Every electrode sees
    every muscle to some degree, which is precisely the crosstalk that makes
    single-channel thresholding inadequate and pattern recognition necessary.
    """
    electrodes = electrode_positions(n_electrodes, shift_rad, geometry)
    muscles = muscle_positions(geometry)
    distances = np.linalg.norm(
        electrodes[:, None, :] - muscles[None, :, :], axis=2
    )
    return np.exp(-distances / geometry.conduction_lambda_mm)
```

- [x] **Step 4: Run and confirm green**

```bash
cd d:/NeuroGrip/services/research && ../../.venv/Scripts/python.exe -m pytest tests/test_anatomy.py -q
```
Expected: 7 passed

- [x] **Step 5: Commit**

```bash
cd d:/NeuroGrip
printf '%s\n' \
  'feat: add forearm anatomy and volume-conduction model' '' \
  'Shared by the simulator and the Pillar 4 attribution ring so the geometry' \
  'is defined exactly once.' '' \
  'Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>' > /tmp/ng-msg.txt
git add services/research/neurogrip/anatomy.py services/research/tests/test_anatomy.py
git commit -F /tmp/ng-msg.txt
```

---

### Task 7: `simulator.py` — biophysical sEMG generator

**Files:**
- Create: `services/research/neurogrip/simulator.py`
- Test: `services/research/tests/test_simulator.py`

**Interfaces:**
- Consumes: `neurogrip.anatomy.{MUSCLES, ForearmGeometry, attenuation_matrix}`
- Produces:
  - `GESTURES: tuple[str, ...]` — the ten-gesture vocabulary
  - `GESTURE_SYNERGIES: dict[str, tuple[float, ...]]` — per-gesture excitation of the six muscles
  - `SimulatorConfig` dataclass
  - `SubjectProfile` dataclass, `make_subject(subject_id: str, seed: int) -> SubjectProfile`
  - `simulate(gesture, duration_s, config, subject, seed, fatigue=0.0) -> np.ndarray` shape `(n_electrodes, n_samples)`

This is a Fuglevand-style motor-unit pool model, not noise shaped to look like EMG. Each muscle carries a pool of motor units with exponentially distributed recruitment thresholds; units recruited above their threshold fire at excitation-dependent rates with inter-spike-interval jitter; each firing convolves a Hermite-Rodriguez MUAP whose duration grows with unit size. Muscle sources are mixed into electrode channels through the volume-conduction matrix from Task 6, which is what produces realistic crosstalk.

Fatigue widens MUAPs, which compresses the spectrum and lowers median frequency — the standard physiological fatigue marker. Getting that behaviour out of the mechanism rather than hard-coding it is what makes the simulator usable for testing the Pillar 3 drift estimator.

Two gesture pairs are deliberately close in synergy space — `fist`/`spherical_grip` and `pinch`/`two_finger`. Realistic confusability is a requirement, not a defect: Pillars 1 and 3 have nothing to demonstrate on a trivially separable vocabulary.

`SubjectProfile` jitters synergies, electrode shift, amplitude and conduction length per subject. Without it, leave-one-subject-out evaluation is meaningless because every simulated subject would be identical.

- [x] **Step 1: Write the failing test**

```python
# services/research/tests/test_simulator.py
import numpy as np
import pytest

from neurogrip import dsp
from neurogrip.simulator import (
    GESTURE_SYNERGIES,
    GESTURES,
    SimulatorConfig,
    make_subject,
    simulate,
)


@pytest.fixture
def subject():
    return make_subject("s01", seed=1)


def test_vocabulary_is_ten_gestures_with_synergies_for_each():
    assert len(GESTURES) == 10
    assert "rest" in GESTURES
    for gesture in GESTURES:
        assert len(GESTURE_SYNERGIES[gesture]) == 6
        assert all(0.0 <= v <= 1.0 for v in GESTURE_SYNERGIES[gesture])


def test_output_shape_matches_electrodes_and_duration(subject):
    config = SimulatorConfig(sampling_rate_hz=2000, n_electrodes=8)
    signal = simulate("fist", 0.5, config, subject, seed=0)
    assert signal.shape == (8, 1000)
    assert signal.dtype == np.float32


def test_identical_seeds_produce_identical_signals(subject):
    config = SimulatorConfig()
    a = simulate("pinch", 0.2, config, subject, seed=42)
    b = simulate("pinch", 0.2, config, subject, seed=42)
    assert np.array_equal(a, b)


def test_different_seeds_produce_different_signals(subject):
    config = SimulatorConfig()
    a = simulate("pinch", 0.2, config, subject, seed=1)
    b = simulate("pinch", 0.2, config, subject, seed=2)
    assert not np.array_equal(a, b)


def test_rest_is_much_quieter_than_a_fist(subject):
    config = SimulatorConfig()
    rest = float(np.sqrt(np.mean(simulate("rest", 0.5, config, subject, seed=3) ** 2)))
    fist = float(np.sqrt(np.mean(simulate("fist", 0.5, config, subject, seed=3) ** 2)))
    assert fist > rest * 3.0


def test_fatigue_lowers_median_frequency(subject):
    """The standard physiological fatigue marker. It must emerge from the
    MUAP-widening mechanism, not be hard-coded."""
    config = SimulatorConfig()
    fresh = simulate("fist", 1.0, config, subject, seed=5, fatigue=0.0)
    tired = simulate("fist", 1.0, config, subject, seed=5, fatigue=1.0)

    def mdf(signal: np.ndarray) -> float:
        freqs, psd = dsp.periodogram(signal[0], float(config.sampling_rate_hz))
        return dsp.median_frequency(freqs, psd)

    assert mdf(tired) < mdf(fresh)


def test_electrode_shift_changes_the_channel_amplitude_profile(subject):
    config = SimulatorConfig()
    shifted = make_subject("s01", seed=1)
    object.__setattr__(shifted, "electrode_shift_rad", np.pi / 6)
    a = simulate("wrist_flexion", 0.5, config, subject, seed=7)
    b = simulate("wrist_flexion", 0.5, config, shifted, seed=7)
    profile_a = np.sqrt(np.mean(a**2, axis=1))
    profile_b = np.sqrt(np.mean(b**2, axis=1))
    assert not np.allclose(profile_a, profile_b, rtol=0.05)


def test_subjects_differ_from_one_another():
    config = SimulatorConfig()
    a = simulate("fist", 0.3, config, make_subject("s01", seed=1), seed=9)
    b = simulate("fist", 0.3, config, make_subject("s02", seed=2), seed=9)
    assert not np.allclose(
        np.sqrt(np.mean(a**2, axis=1)), np.sqrt(np.mean(b**2, axis=1)), rtol=0.05
    )


def test_flexion_and_extension_load_opposite_sides_of_the_ring(subject):
    """Sanity check that anatomy is actually driving the mixing."""
    config = SimulatorConfig(n_electrodes=8)
    flex = simulate("wrist_flexion", 0.5, config, subject, seed=11)
    extend = simulate("wrist_extension", 0.5, config, subject, seed=11)
    assert int(np.argmax(np.sqrt(np.mean(flex**2, axis=1)))) != int(
        np.argmax(np.sqrt(np.mean(extend**2, axis=1)))
    )


def test_rejects_unknown_gesture(subject):
    with pytest.raises(KeyError):
        simulate("moonwalk", 0.1, SimulatorConfig(), subject, seed=0)
```

- [x] **Step 2: Run and watch it fail**

```bash
cd d:/NeuroGrip/services/research && ../../.venv/Scripts/python.exe -m pytest tests/test_simulator.py -q
```
Expected: FAIL — `ModuleNotFoundError: No module named 'neurogrip.simulator'`

- [x] **Step 3: Implement `simulator.py`**

```python
"""Biophysical surface-EMG simulator.

A Fuglevand-style motor-unit pool model, not shaped noise. Each muscle holds a
pool of motor units with exponentially distributed recruitment thresholds and
sizes; units above threshold fire at excitation-dependent rates with
inter-spike-interval jitter; each firing convolves a Hermite-Rodriguez motor
unit action potential whose duration scales with unit size. Muscle sources mix
into electrode channels through the volume-conduction matrix in anatomy.py.

Fatigue widens MUAPs, compressing the spectrum and lowering median frequency.
That emerges from the mechanism rather than being asserted, which is what makes
the simulator usable as a test bed for the drift and fatigue estimators.

Everything is seeded. Nothing here may touch global RNG state.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from neurogrip.anatomy import MUSCLES, ForearmGeometry, attenuation_matrix

GESTURES: tuple[str, ...] = (
    "rest",
    "fist",
    "open_hand",
    "pinch",
    "point",
    "wrist_flexion",
    "wrist_extension",
    "thumb_up",
    "two_finger",
    "spherical_grip",
)

# Excitation of (fcr, fds, fcu, ecu, edc, ecr) in [0, 1] for each gesture.
# fist/spherical_grip and pinch/two_finger are deliberately close: a trivially
# separable vocabulary would give Pillars 1 and 3 nothing to work on.
GESTURE_SYNERGIES: dict[str, tuple[float, ...]] = {
    "rest":            (0.02, 0.02, 0.02, 0.02, 0.02, 0.02),
    "fist":            (0.50, 0.90, 0.60, 0.20, 0.10, 0.15),
    "open_hand":       (0.10, 0.10, 0.10, 0.60, 0.90, 0.60),
    "pinch":           (0.30, 0.55, 0.20, 0.15, 0.25, 0.20),
    "point":           (0.25, 0.45, 0.25, 0.30, 0.55, 0.30),
    "wrist_flexion":   (0.85, 0.35, 0.80, 0.10, 0.10, 0.10),
    "wrist_extension": (0.10, 0.10, 0.10, 0.80, 0.35, 0.85),
    "thumb_up":        (0.50, 0.20, 0.15, 0.20, 0.20, 0.55),
    "two_finger":      (0.20, 0.50, 0.20, 0.25, 0.50, 0.25),
    "spherical_grip":  (0.55, 0.70, 0.55, 0.40, 0.35, 0.45),
}


@dataclass(frozen=True)
class SimulatorConfig:
    sampling_rate_hz: int = 2000
    n_electrodes: int = 8
    motor_units_per_muscle: int = 30
    recruitment_range: float = 30.0
    amplitude_range: float = 25.0
    min_firing_rate_hz: float = 8.0
    max_firing_rate_hz: float = 35.0
    firing_rate_gain: float = 30.0
    isi_cv: float = 0.2
    muap_tau_s: float = 0.0025
    fatigue_tau_widening: float = 0.8
    noise_std: float = 3e-4
    powerline_hz: float = 50.0
    powerline_amplitude: float = 0.0
    output_scale: float = 2e-3


@dataclass
class SubjectProfile:
    """Per-subject variation. Without it, LOSO evaluation is meaningless."""

    subject_id: str
    electrode_shift_rad: float
    amplitude_scale: float
    geometry: ForearmGeometry
    synergy_gain: np.ndarray = field(repr=False)  # shape (6,), multiplicative


def make_subject(subject_id: str, seed: int) -> SubjectProfile:
    rng = np.random.default_rng(seed)
    return SubjectProfile(
        subject_id=subject_id,
        electrode_shift_rad=float(rng.uniform(-math.pi / 8, math.pi / 8)),
        amplitude_scale=float(rng.uniform(0.7, 1.4)),
        geometry=ForearmGeometry(
            conduction_lambda_mm=float(rng.uniform(12.0, 19.0)),
        ),
        synergy_gain=rng.uniform(0.75, 1.25, size=len(MUSCLES)),
    )


def simulate(
    gesture: str,
    duration_s: float,
    config: SimulatorConfig,
    subject: SubjectProfile,
    seed: int,
    fatigue: float = 0.0,
) -> np.ndarray:
    """Generate (n_electrodes, n_samples) of simulated sEMG.

    `fatigue` in [0, 1] widens motor unit action potentials, which lowers
    median frequency exactly as a fatiguing muscle does.
    """
    if gesture not in GESTURE_SYNERGIES:
        raise KeyError(f"unknown gesture: {gesture}")
    if duration_s <= 0:
        raise ValueError("duration_s must be positive")

    rng = np.random.default_rng(seed)
    n_samples = int(round(duration_s * config.sampling_rate_hz))
    excitations = np.asarray(GESTURE_SYNERGIES[gesture], dtype=np.float64)
    excitations = np.clip(excitations * subject.synergy_gain, 0.0, 1.0)

    sources = np.stack(
        [
            _muscle_signal(float(excitation), n_samples, config, rng, fatigue)
            for excitation in excitations
        ]
    )

    gain = attenuation_matrix(
        config.n_electrodes, subject.electrode_shift_rad, subject.geometry
    )
    channels = gain @ sources
    channels *= config.output_scale * subject.amplitude_scale

    channels += rng.normal(0.0, config.noise_std, size=channels.shape)
    if config.powerline_amplitude > 0.0:
        t = np.arange(n_samples, dtype=np.float64) / config.sampling_rate_hz
        channels += config.powerline_amplitude * np.sin(
            2.0 * np.pi * config.powerline_hz * t
        )

    return channels.astype(np.float32)


def _muscle_signal(
    excitation: float,
    n_samples: int,
    config: SimulatorConfig,
    rng: np.random.Generator,
    fatigue: float,
) -> np.ndarray:
    out = np.zeros(n_samples, dtype=np.float64)
    pool = config.motor_units_per_muscle
    log_recruitment = math.log(config.recruitment_range)
    log_amplitude = math.log(config.amplitude_range)

    for unit in range(1, pool + 1):
        threshold = math.exp(log_recruitment * unit / pool) / config.recruitment_range
        if excitation < threshold:
            continue

        rate = min(
            config.max_firing_rate_hz,
            config.min_firing_rate_hz
            + config.firing_rate_gain * (excitation - threshold),
        )
        amplitude = math.exp(log_amplitude * unit / pool) / config.amplitude_range
        tau = (
            config.muap_tau_s
            * (1.0 + 0.5 * unit / pool)
            * (1.0 + config.fatigue_tau_widening * fatigue)
        )
        kernel = _muap_kernel(tau, config.sampling_rate_hz, amplitude)
        half = kernel.size // 2

        for index in _spike_indices(rate, n_samples, config, rng):
            low, high = index - half, index + half + 1
            k_low, k_high = 0, kernel.size
            if low < 0:
                k_low = -low
                low = 0
            if high > n_samples:
                k_high -= high - n_samples
                high = n_samples
            if high > low:
                out[low:high] += kernel[k_low:k_high]

    return out


def _muap_kernel(tau_s: float, fs: int, amplitude: float) -> np.ndarray:
    """First-order Hermite-Rodriguez motor unit action potential, peak-normalised."""
    half = max(1, int(math.ceil(4.0 * tau_s * fs)))
    t = np.arange(-half, half + 1, dtype=np.float64) / fs
    scaled = t / tau_s
    shape = scaled * np.exp(-0.5 * np.square(scaled))
    peak = float(np.max(np.abs(shape)))
    if peak > 0.0:
        shape = shape / peak
    return amplitude * shape


def _spike_indices(
    rate_hz: float, n_samples: int, config: SimulatorConfig, rng: np.random.Generator
) -> np.ndarray:
    if rate_hz <= 0.0:
        return np.empty(0, dtype=np.int64)

    mean_isi = 1.0 / rate_hz
    refractory = 0.2 * mean_isi
    indices: list[int] = []
    time_s = float(rng.uniform(0.0, mean_isi))

    while True:
        index = int(time_s * config.sampling_rate_hz)
        if index >= n_samples:
            break
        indices.append(index)
        isi = mean_isi * (1.0 + config.isi_cv * float(rng.standard_normal()))
        time_s += max(isi, refractory)

    return np.asarray(indices, dtype=np.int64)
```

- [x] **Step 4: Run and confirm green**

```bash
cd d:/NeuroGrip/services/research && ../../.venv/Scripts/python.exe -m pytest tests/test_simulator.py -q
```
Expected: 10 passed. If `test_fatigue_lowers_median_frequency` fails, raise `fatigue_tau_widening`; if `test_rest_is_much_quieter_than_a_fist` fails, lower the `rest` synergy values. Do not weaken the assertions.

- [x] **Step 5: Commit**

```bash
cd d:/NeuroGrip
printf '%s\n' \
  'feat: add biophysical sEMG simulator with motor-unit pool model' '' \
  'Fuglevand-style recruitment and rate coding, Hermite-Rodriguez MUAPs, and' \
  'volume-conduction mixing. Fatigue emerges from MUAP widening rather than' \
  'being hard-coded, so it can serve as a test bed for the drift estimator.' '' \
  'Unblocks the whole project from the NinaPro registration dependency.' '' \
  'Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>' > /tmp/ng-msg.txt
git add services/research/neurogrip/simulator.py services/research/tests/test_simulator.py
git commit -F /tmp/ng-msg.txt
```

---

### Task 8: `fixtures.py` — golden-vector generator

**Files:**
- Create: `services/research/neurogrip/fixtures.py`
- Create: `fixtures/conformance/golden.json` (generated, committed)
- Test: `services/research/tests/test_fixtures.py`

**Interfaces:**
- Consumes: `neurogrip.{FEATURE_SPEC_VERSION}`, `neurogrip.features.*`, `neurogrip.simulator.*`, `neurogrip.spec.*`
- Produces:
  - `build_fixture_payload() -> dict` — the full golden-vector document
  - `write_fixtures(path: pathlib.Path) -> dict`
  - CLI: `python -m neurogrip.fixtures --out fixtures/conformance/golden.json`

The payload pins both the *inputs* and the *expected outputs*, so the TypeScript port has something unambiguous to be wrong against. Cases deliberately include the awkward ones: a silent channel, a DC-offset channel, a pure tone, a signal that touches zero without crossing, and a twelve-channel case that would expose lexicographic ordering.

Document shape:

```json
{
  "featureSpecVersion": 1,
  "perChannelFeatures": ["rms", "..."],
  "config": {"samplingRateHz": 2000, "zeroCrossingThreshold": 1e-05, "slopeSignThreshold": 1e-05},
  "cases": [
    {
      "name": "simulated_fist_8ch",
      "samplingRateHz": 2000,
      "channels": [[...], ...],
      "featureNames": ["ch1_rms", "..."],
      "featureVector": [0.0123, ...]
    }
  ]
}
```

- [x] **Step 1: Write the failing test**

```python
# services/research/tests/test_fixtures.py
import json

import numpy as np

from neurogrip import FEATURE_SPEC_VERSION
from neurogrip.features import FeatureConfig, extract_features, feature_vector
from neurogrip.fixtures import build_fixture_payload, write_fixtures
from neurogrip.spec import PER_CHANNEL_FEATURES


def test_payload_declares_the_spec_version_and_feature_table():
    payload = build_fixture_payload()
    assert payload["featureSpecVersion"] == FEATURE_SPEC_VERSION
    assert tuple(payload["perChannelFeatures"]) == PER_CHANNEL_FEATURES


def test_payload_covers_the_awkward_cases():
    names = {case["name"] for case in build_fixture_payload()["cases"]}
    for required in (
        "silent",
        "dc_offset",
        "pure_tone",
        "touches_zero",
        "twelve_channels",
    ):
        assert required in names


def test_every_case_vector_matches_a_fresh_extraction():
    """The fixture must be self-consistent or it pins the wrong answer."""
    payload = build_fixture_payload()
    for case in payload["cases"]:
        channels = np.asarray(case["channels"], dtype=np.float32)
        config = FeatureConfig(
            sampling_rate_hz=case["samplingRateHz"],
            zero_crossing_threshold=payload["config"]["zeroCrossingThreshold"],
            slope_sign_threshold=payload["config"]["slopeSignThreshold"],
        )
        expected = feature_vector(extract_features(channels, config), channels.shape[0])
        assert np.allclose(expected, case["featureVector"], rtol=1e-6, atol=1e-12)
        assert list(case["featureNames"]) == case["featureNames"]
        assert len(case["featureVector"]) == len(case["featureNames"])


def test_generation_is_deterministic():
    assert build_fixture_payload() == build_fixture_payload()


def test_write_fixtures_round_trips(tmp_path):
    target = tmp_path / "golden.json"
    written = write_fixtures(target)
    assert json.loads(target.read_text(encoding="utf-8")) == written
```

- [x] **Step 2: Run and watch it fail**

```bash
cd d:/NeuroGrip/services/research && ../../.venv/Scripts/python.exe -m pytest tests/test_fixtures.py -q
```
Expected: FAIL — `ModuleNotFoundError: No module named 'neurogrip.fixtures'`

- [x] **Step 3: Implement `fixtures.py`**

```python
"""Golden-vector generator: the Python side of the conformance contract.

Pins both inputs and expected outputs so the TypeScript port has something
unambiguous to be wrong against. Deliberately includes the awkward cases -
silence, DC offset, a pure tone, a signal that touches zero without crossing,
and a twelve-channel case that exposes lexicographic feature ordering.

Regenerate whenever FEATURE_SPEC_VERSION changes:
    python -m neurogrip.fixtures --out fixtures/conformance/golden.json
"""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np

from neurogrip import FEATURE_SPEC_VERSION
from neurogrip.features import FeatureConfig, extract_features, feature_vector
from neurogrip.simulator import SimulatorConfig, make_subject, simulate
from neurogrip.spec import PER_CHANNEL_FEATURES, feature_names

SAMPLING_RATE_HZ = 2000
WINDOW_SAMPLES = 400  # 200 ms at 2 kHz
ZERO_CROSSING_THRESHOLD = 1e-5
SLOPE_SIGN_THRESHOLD = 1e-5


def _simulated(gesture: str, n_electrodes: int, seed: int) -> np.ndarray:
    config = SimulatorConfig(
        sampling_rate_hz=SAMPLING_RATE_HZ, n_electrodes=n_electrodes
    )
    subject = make_subject(f"fixture-{seed}", seed=seed)
    signal = simulate(
        gesture, WINDOW_SAMPLES / SAMPLING_RATE_HZ, config, subject, seed=seed
    )
    return signal[:, :WINDOW_SAMPLES]


def _synthetic_cases() -> dict[str, np.ndarray]:
    t = np.arange(WINDOW_SAMPLES, dtype=np.float64) / SAMPLING_RATE_HZ
    ramp = np.linspace(-0.02, 0.02, WINDOW_SAMPLES)

    # Alternates 1, 0, 1, 0 ... : touches zero repeatedly but never crosses.
    touching = np.zeros(WINDOW_SAMPLES, dtype=np.float64)
    touching[::2] = 0.01

    return {
        "silent": np.zeros((2, WINDOW_SAMPLES)),
        "dc_offset": np.full((2, WINDOW_SAMPLES), 0.05),
        "pure_tone": np.stack(
            [np.sin(2 * np.pi * 120.0 * t) * 0.01, np.cos(2 * np.pi * 40.0 * t) * 0.01]
        ),
        "touches_zero": np.stack([touching, -touching]),
        "linear_ramp": np.stack([ramp, ramp[::-1]]),
    }


def build_fixture_payload() -> dict:
    config = FeatureConfig(
        sampling_rate_hz=SAMPLING_RATE_HZ,
        zero_crossing_threshold=ZERO_CROSSING_THRESHOLD,
        slope_sign_threshold=SLOPE_SIGN_THRESHOLD,
    )

    sources: list[tuple[str, np.ndarray]] = [
        ("simulated_rest_8ch", _simulated("rest", 8, seed=101)),
        ("simulated_fist_8ch", _simulated("fist", 8, seed=102)),
        ("simulated_pinch_8ch", _simulated("pinch", 8, seed=103)),
        ("simulated_wrist_extension_8ch", _simulated("wrist_extension", 8, seed=104)),
        # Twelve channels: lexicographic ordering would emit ch10 before ch2 here.
        ("twelve_channels", _simulated("spherical_grip", 12, seed=105)),
        ("single_channel", _simulated("fist", 1, seed=106)),
    ]
    sources.extend(_synthetic_cases().items())

    cases = []
    for name, channels in sources:
        matrix = np.asarray(channels, dtype=np.float32)
        n_channels = matrix.shape[0]
        vector = feature_vector(extract_features(matrix, config), n_channels)
        cases.append(
            {
                "name": name,
                "samplingRateHz": SAMPLING_RATE_HZ,
                "channels": [[float(v) for v in row] for row in matrix],
                "featureNames": list(feature_names(n_channels)),
                "featureVector": [float(v) for v in vector],
            }
        )

    return {
        "featureSpecVersion": FEATURE_SPEC_VERSION,
        "perChannelFeatures": list(PER_CHANNEL_FEATURES),
        "config": {
            "samplingRateHz": SAMPLING_RATE_HZ,
            "zeroCrossingThreshold": ZERO_CROSSING_THRESHOLD,
            "slopeSignThreshold": SLOPE_SIGN_THRESHOLD,
        },
        "cases": cases,
    }


def write_fixtures(path: pathlib.Path) -> dict:
    payload = build_fixture_payload()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate conformance golden vectors")
    parser.add_argument(
        "--out",
        type=pathlib.Path,
        default=pathlib.Path("fixtures/conformance/golden.json"),
    )
    args = parser.parse_args()
    payload = write_fixtures(args.out)
    print(
        f"wrote {len(payload['cases'])} cases "
        f"(spec v{payload['featureSpecVersion']}) to {args.out}"
    )


if __name__ == "__main__":
    main()
```

- [x] **Step 4: Run the tests, then generate the committed fixture**

```bash
cd d:/NeuroGrip/services/research && ../../.venv/Scripts/python.exe -m pytest tests/test_fixtures.py -q
cd d:/NeuroGrip && ./.venv/Scripts/python.exe -m neurogrip.fixtures --out fixtures/conformance/golden.json
ls -lh fixtures/conformance/golden.json
```
Expected: 5 passed, then `wrote 11 cases (spec v1) to fixtures/conformance/golden.json`.

- [x] **Step 5: Run the whole Python suite before handing off to TypeScript**

```bash
cd d:/NeuroGrip/services/research && ../../.venv/Scripts/python.exe -m pytest -q
```
Expected: all green. This is the last Python-only gate.

- [x] **Step 6: Commit**

```bash
cd d:/NeuroGrip
printf '%s\n' \
  'feat: add golden-vector generator and committed conformance fixtures' '' \
  'Pins inputs and expected outputs so the TypeScript port has something' \
  'unambiguous to be wrong against. Includes silence, DC offset, a pure tone,' \
  'a zero-touching signal, and a twelve-channel case that would expose' \
  'lexicographic feature ordering.' '' \
  'Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>' > /tmp/ng-msg.txt
git add services/research/neurogrip/fixtures.py services/research/tests/test_fixtures.py fixtures/conformance/golden.json
git commit -F /tmp/ng-msg.txt
```

---

### Task 9: npm workspace root and `@neurogrip/core` skeleton

**Files:**
- Create: `package.json`, `tsconfig.base.json`, `.gitignore` (append), `packages/core/package.json`, `packages/core/tsconfig.json`, `packages/core/vitest.config.ts`, `packages/core/src/index.ts`

**Interfaces:**
- Consumes: nothing
- Produces: working `npm test` and `npm run typecheck` at the repo root; `@neurogrip/core` importable as a workspace package

npm workspaces, not pnpm — pnpm is not installed and introducing it adds a setup step for no benefit at two packages.

- [x] **Step 1: Write the root `package.json`**

```json
{
  "name": "neurogrip",
  "private": true,
  "version": "0.2.0",
  "type": "module",
  "workspaces": ["packages/*", "apps/*"],
  "scripts": {
    "test": "vitest run",
    "test:watch": "vitest",
    "test:conformance": "vitest run packages/core/test/conformance.test.ts",
    "typecheck": "tsc --build --verbose",
    "fixtures": "python -m neurogrip.fixtures --out fixtures/conformance/golden.json"
  },
  "devDependencies": {
    "typescript": "^5.7.0",
    "vitest": "^3.0.0"
  }
}
```

- [x] **Step 2: Write `tsconfig.base.json`**

```json
{
  "compilerOptions": {
    "target": "ES2022",
    "lib": ["ES2022"],
    "module": "ESNext",
    "moduleResolution": "bundler",
    "strict": true,
    "noUncheckedIndexedAccess": true,
    "noImplicitOverride": true,
    "exactOptionalPropertyTypes": true,
    "declaration": true,
    "declarationMap": true,
    "sourceMap": true,
    "skipLibCheck": true,
    "verbatimModuleSyntax": true,
    "isolatedModules": true
  }
}
```

`noUncheckedIndexedAccess` is on deliberately: this package indexes typed arrays constantly and an off-by-one in a DSP loop is exactly the bug that would silently break conformance.

- [x] **Step 3: Write `packages/core/package.json`**

```json
{
  "name": "@neurogrip/core",
  "version": "0.2.0",
  "private": true,
  "type": "module",
  "main": "./src/index.ts",
  "types": "./src/index.ts",
  "exports": { ".": "./src/index.ts" },
  "scripts": {
    "test": "vitest run",
    "typecheck": "tsc --noEmit"
  }
}
```

- [x] **Step 4: Write `packages/core/tsconfig.json`**

```json
{
  "extends": "../../tsconfig.base.json",
  "compilerOptions": {
    "rootDir": ".",
    "outDir": "dist",
    "noEmit": true
  },
  "include": ["src/**/*.ts", "test/**/*.ts"]
}
```

- [x] **Step 5: Write `packages/core/vitest.config.ts`**

```typescript
import { defineConfig } from 'vitest/config';

export default defineConfig({
  test: {
    include: ['test/**/*.test.ts'],
    environment: 'node',
  },
});
```

- [x] **Step 6: Write a placeholder `packages/core/src/index.ts`**

```typescript
export const CORE_VERSION = '0.2.0';
```

- [x] **Step 7: Extend `.gitignore` for Node**

```bash
cd d:/NeuroGrip
printf '\n# Node\nnode_modules/\ndist/\n*.tsbuildinfo\n' >> .gitignore
```

- [x] **Step 8: Install and verify the toolchain runs**

```bash
cd d:/NeuroGrip
npm install
npx vitest run --passWithNoTests
npx tsc --noEmit -p packages/core/tsconfig.json
```
Expected: install succeeds, vitest exits 0 with no tests, tsc exits 0.

- [x] **Step 9: Commit**

```bash
cd d:/NeuroGrip
printf '%s\n' \
  'build: add npm workspace root and @neurogrip/core skeleton' '' \
  'Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>' > /tmp/ng-msg.txt
git add package.json package-lock.json tsconfig.base.json .gitignore packages/
git commit -F /tmp/ng-msg.txt
```

---

### Task 10: `dsp.ts` — port the numeric primitives

**Files:**
- Create: `packages/core/src/dsp.ts`
- Test: `packages/core/test/dsp.test.ts`

**Interfaces:**
- Consumes: nothing
- Produces:
  - `nextPow2(n: number): number`
  - `hann(n: number): Float64Array`
  - `periodogram(x: Float64Array, fs: number): { freqs: Float64Array; psd: Float64Array }`
  - `bandMeans(values: Float64Array, nBands: number): Float64Array`
  - `autocorrelate(x: Float64Array, maxLag: number): Float64Array`
  - `levinsonDurbin(r: Float64Array, order: number): Float64Array`
  - `medianFrequency(freqs: Float64Array, psd: Float64Array): number`
  - `meanFrequency(freqs: Float64Array, psd: Float64Array): number`

Port target is `services/research/neurogrip/dsp.py`, algorithm for algorithm. Twiddle factors are computed from a cached table using `Math.cos`/`Math.sin` per index rather than accumulated incrementally — incremental accumulation drifts over the butterfly stages, and conformance is worth more here than the microseconds saved.

- [x] **Step 1: Write the failing test**

```typescript
// packages/core/test/dsp.test.ts
import { describe, expect, it } from 'vitest';
import {
  autocorrelate,
  bandMeans,
  hann,
  levinsonDurbin,
  meanFrequency,
  medianFrequency,
  nextPow2,
  periodogram,
} from '../src/dsp.js';

describe('nextPow2', () => {
  it('rounds up to a power of two', () => {
    expect(nextPow2(1)).toBe(1);
    expect(nextPow2(2)).toBe(2);
    expect(nextPow2(3)).toBe(4);
    expect(nextPow2(400)).toBe(512);
  });
});

describe('hann', () => {
  it('is symmetric and zero at both ends', () => {
    const w = hann(9);
    expect(w[0]).toBeCloseTo(0, 12);
    expect(w[8]).toBeCloseTo(0, 12);
    expect(w[4]).toBeCloseTo(1, 12);
    expect(w[1]).toBeCloseTo(w[7]!, 12);
  });

  it('returns [1] for a single sample', () => {
    expect(Array.from(hann(1))).toEqual([1]);
  });
});

describe('periodogram', () => {
  it('locates a pure tone', () => {
    const fs = 1000;
    const x = new Float64Array(512);
    for (let i = 0; i < x.length; i++) x[i] = Math.sin((2 * Math.PI * 100 * i) / fs);
    const { freqs, psd } = periodogram(x, fs);
    let peak = 0;
    for (let i = 1; i < psd.length; i++) if (psd[i]! > psd[peak]!) peak = i;
    expect(freqs[peak]).toBeCloseTo(100, 0);
    expect(psd.length).toBe(512 / 2 + 1);
  });
});

describe('bandMeans', () => {
  it('uses numpy array_split sizing', () => {
    // 7 items into 3 bands -> 3, 2, 2
    const values = Float64Array.from([1, 2, 3, 4, 5, 6, 7]);
    const means = bandMeans(values, 3);
    expect(means[0]).toBeCloseTo(2, 12);
    expect(means[1]).toBeCloseTo(4.5, 12);
    expect(means[2]).toBeCloseTo(6.5, 12);
  });

  it('emits zero for bands with no samples', () => {
    const means = bandMeans(Float64Array.from([5]), 3);
    expect(Array.from(means)).toEqual([5, 0, 0]);
  });
});

describe('autocorrelate', () => {
  it('is biased and peaks at zero lag', () => {
    const x = Float64Array.from([1, 2, 3, 4]);
    const r = autocorrelate(x, 2);
    expect(r.length).toBe(3);
    expect(r[0]).toBeCloseTo((1 + 4 + 9 + 16) / 4, 12);
    expect(r[0]!).toBeGreaterThanOrEqual(r[1]!);
  });
});

describe('levinsonDurbin', () => {
  it('returns zeros for a signal with no power', () => {
    expect(Array.from(levinsonDurbin(new Float64Array(5), 4))).toEqual([0, 0, 0, 0]);
  });
});

describe('frequency moments', () => {
  it('are correct on a flat spectrum', () => {
    const freqs = Float64Array.from([0, 10, 20, 30, 40]);
    const psd = Float64Array.from([1, 1, 1, 1, 1]);
    expect(meanFrequency(freqs, psd)).toBeCloseTo(20, 12);
    expect(medianFrequency(freqs, psd)).toBeCloseTo(20, 12);
  });

  it('are zero when there is no power', () => {
    const freqs = Float64Array.from([0, 10, 20]);
    const psd = new Float64Array(3);
    expect(meanFrequency(freqs, psd)).toBe(0);
    expect(medianFrequency(freqs, psd)).toBe(0);
  });
});
```

- [x] **Step 2: Run and watch it fail**

```bash
cd d:/NeuroGrip && npx vitest run packages/core/test/dsp.test.ts
```
Expected: FAIL — cannot resolve `../src/dsp.js`

- [x] **Step 3: Implement `dsp.ts`**

```typescript
/**
 * Numeric primitives for the feature pipeline.
 *
 * A port of services/research/neurogrip/dsp.py, algorithm for algorithm.
 * The two implementations are pinned against each other by
 * test/conformance.test.ts. Do not "improve" anything here without changing
 * the Python side and regenerating the golden vectors.
 */

export function nextPow2(n: number): number {
  if (n < 1) throw new RangeError('n must be positive');
  let power = 1;
  while (power < n) power *= 2;
  return power;
}

export function hann(n: number): Float64Array {
  if (n < 1) throw new RangeError('n must be positive');
  const w = new Float64Array(n);
  if (n === 1) {
    w[0] = 1;
    return w;
  }
  for (let i = 0; i < n; i++) {
    w[i] = 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / (n - 1));
  }
  return w;
}

interface Twiddle {
  cos: Float64Array;
  sin: Float64Array;
}

const twiddleCache = new Map<number, Twiddle>();

function twiddles(n: number): Twiddle {
  const cached = twiddleCache.get(n);
  if (cached) return cached;
  const cos = new Float64Array(n / 2);
  const sin = new Float64Array(n / 2);
  for (let k = 0; k < n / 2; k++) {
    const angle = (-2 * Math.PI * k) / n;
    cos[k] = Math.cos(angle);
    sin[k] = Math.sin(angle);
  }
  const table = { cos, sin };
  twiddleCache.set(n, table);
  return table;
}

/**
 * In-place iterative radix-2 Cooley-Tukey FFT.
 *
 * Twiddles come from a precomputed exact table rather than being accumulated
 * incrementally across butterflies: incremental accumulation drifts, and
 * conformance with numpy matters more than the microseconds saved.
 */
function fftInPlace(real: Float64Array, imag: Float64Array): void {
  const n = real.length;
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) {
      [real[i], real[j]] = [real[j]!, real[i]!];
      [imag[i], imag[j]] = [imag[j]!, imag[i]!];
    }
  }

  const table = twiddles(n);
  for (let len = 2; len <= n; len <<= 1) {
    const half = len >> 1;
    const stride = n / len;
    for (let start = 0; start < n; start += len) {
      for (let k = 0; k < half; k++) {
        const wr = table.cos[k * stride]!;
        const wi = table.sin[k * stride]!;
        const a = start + k;
        const b = a + half;
        const vr = real[b]! * wr - imag[b]! * wi;
        const vi = real[b]! * wi + imag[b]! * wr;
        real[b] = real[a]! - vr;
        imag[b] = imag[a]! - vi;
        real[a] = real[a]! + vr;
        imag[a] = imag[a]! + vi;
      }
    }
  }
}

export function periodogram(
  x: Float64Array,
  fs: number,
): { freqs: Float64Array; psd: Float64Array } {
  const n = x.length;
  if (n < 1) throw new RangeError('signal must not be empty');

  const window = hann(n);
  const nfft = nextPow2(n);
  const real = new Float64Array(nfft);
  const imag = new Float64Array(nfft);

  let windowEnergy = 0;
  for (let i = 0; i < n; i++) {
    real[i] = x[i]! * window[i]!;
    windowEnergy += window[i]! * window[i]!;
  }

  fftInPlace(real, imag);

  const bins = nfft / 2 + 1;
  const psd = new Float64Array(bins);
  const scale = fs * windowEnergy;
  if (scale > 0) {
    for (let k = 0; k < bins; k++) {
      psd[k] = (real[k]! * real[k]! + imag[k]! * imag[k]!) / scale;
    }
    // Fold negative-frequency power into positive bins, excluding DC and Nyquist.
    for (let k = 1; k < bins - 1; k++) psd[k] = psd[k]! * 2;
  }

  const freqs = new Float64Array(bins);
  for (let k = 0; k < bins; k++) freqs[k] = (k * fs) / nfft;

  return { freqs, psd };
}

/** Mean of each of nBands contiguous groups, matching numpy array_split sizing. */
export function bandMeans(values: Float64Array, nBands: number): Float64Array {
  if (nBands < 1) throw new RangeError('nBands must be positive');
  const out = new Float64Array(nBands);
  const base = Math.floor(values.length / nBands);
  const remainder = values.length % nBands;

  let start = 0;
  for (let band = 0; band < nBands; band++) {
    const size = base + (band < remainder ? 1 : 0);
    if (size === 0) {
      out[band] = 0;
      continue;
    }
    let sum = 0;
    for (let i = 0; i < size; i++) sum += values[start + i]!;
    out[band] = sum / size;
    start += size;
  }
  return out;
}

/** Biased autocorrelation r[k] = sum(x[n] * x[n+k]) / N for k = 0..maxLag. */
export function autocorrelate(x: Float64Array, maxLag: number): Float64Array {
  if (maxLag < 0) throw new RangeError('maxLag must not be negative');
  const n = x.length;
  const out = new Float64Array(maxLag + 1);
  const limit = Math.min(maxLag, n - 1);
  for (let lag = 0; lag <= limit; lag++) {
    let sum = 0;
    for (let i = 0; i < n - lag; i++) sum += x[i]! * x[i + lag]!;
    out[lag] = sum / n;
  }
  return out;
}

/**
 * AR coefficients a[1..order] for x[n] = -sum(a[i] * x[n-i]) + e[n].
 * Returns zeros for a signal with no power, which is correct for a silent channel.
 */
export function levinsonDurbin(r: Float64Array, order: number): Float64Array {
  if (order < 1) throw new RangeError('order must be positive');
  const zeros = new Float64Array(order);
  if (r.length < order + 1 || r[0]! <= 0) return zeros;

  const a = new Float64Array(order + 1);
  a[0] = 1;
  let error = r[0]!;

  for (let m = 1; m <= order; m++) {
    let acc = r[m]!;
    for (let i = 1; i < m; i++) acc += a[i]! * r[m - i]!;
    const reflection = -acc / error;

    const previous = a.slice(1, m);
    a[m] = reflection;
    for (let i = 1; i < m; i++) {
      a[i] = previous[i - 1]! + reflection * previous[m - 1 - i]!;
    }

    error *= 1 - reflection * reflection;
    if (error <= 0) return zeros;
  }

  return a.slice(1);
}

export function medianFrequency(freqs: Float64Array, psd: Float64Array): number {
  let total = 0;
  for (let i = 0; i < psd.length; i++) total += psd[i]!;
  if (total <= 0) return 0;

  const target = 0.5 * total;
  let cumulative = 0;
  for (let i = 0; i < psd.length; i++) {
    cumulative += psd[i]!;
    if (cumulative >= target) return freqs[i]!;
  }
  return freqs[freqs.length - 1]!;
}

export function meanFrequency(freqs: Float64Array, psd: Float64Array): number {
  let total = 0;
  let weighted = 0;
  for (let i = 0; i < psd.length; i++) {
    total += psd[i]!;
    weighted += freqs[i]! * psd[i]!;
  }
  return total <= 0 ? 0 : weighted / total;
}
```

- [x] **Step 4: Run and confirm green**

```bash
cd d:/NeuroGrip && npx vitest run packages/core/test/dsp.test.ts
```
Expected: 10 passed

- [x] **Step 5: Commit**

```bash
cd d:/NeuroGrip
printf '%s\n' \
  'feat(core): port DSP primitives to TypeScript' '' \
  'Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>' > /tmp/ng-msg.txt
git add packages/core/src/dsp.ts packages/core/test/dsp.test.ts
git commit -F /tmp/ng-msg.txt
```

---

### Task 11: `spec.ts` and `features.ts` — port the feature pipeline

**Files:**
- Create: `packages/core/src/spec.ts`, `packages/core/src/features.ts`
- Modify: `packages/core/src/index.ts`
- Test: `packages/core/test/features.test.ts`

**Interfaces:**
- Consumes: `./dsp.js`
- Produces:
  - `FEATURE_SPEC_VERSION: 1`, `AR_ORDER: 4`, `PSD_BINS: 6`, `PER_CHANNEL_FEATURES: readonly string[]`
  - `featureNames(nChannels: number): string[]`, `featureCount(nChannels: number): number`
  - `FeatureConfig` interface — `samplingRateHz`, `zeroCrossingThreshold`, `slopeSignThreshold`
  - `DEFAULT_FEATURE_CONFIG: FeatureConfig`
  - `extractFeatures(channels: Float64Array[], config: FeatureConfig): Map<string, number>`
  - `featureVector(channels: Float64Array[], config: FeatureConfig): Float32Array`

`featureVector` takes the channels directly rather than a feature map, because the hot path in the Web Worker should never allocate a `Map` per window. `extractFeatures` exists for debugging and for the conformance test's named comparisons.

- [x] **Step 1: Write the failing test**

```typescript
// packages/core/test/features.test.ts
import { describe, expect, it } from 'vitest';
import { DEFAULT_FEATURE_CONFIG, extractFeatures, featureVector } from '../src/features.js';
import { PER_CHANNEL_FEATURES, featureCount, featureNames } from '../src/spec.js';

const config = { ...DEFAULT_FEATURE_CONFIG, zeroCrossingThreshold: 0, slopeSignThreshold: 0 };

describe('spec', () => {
  it('declares the seventeen per-channel features', () => {
    expect(PER_CHANNEL_FEATURES).toEqual([
      'rms', 'mav', 'wl', 'zc', 'ssc',
      'ar1', 'ar2', 'ar3', 'ar4',
      'psd1', 'psd2', 'psd3', 'psd4', 'psd5', 'psd6',
      'mdf', 'mnf',
    ]);
  });

  it('orders names channel-major, not lexicographically', () => {
    const names = featureNames(12);
    expect(names.indexOf('ch2_rms')).toBeLessThan(names.indexOf('ch10_rms'));
    expect(names).not.toEqual([...names].sort());
    expect(names.length).toBe(featureCount(12));
  });
});

describe('extractFeatures', () => {
  it('computes the time-domain values', () => {
    const features = extractFeatures([Float64Array.from([3, -4, 3, -4])], config);
    expect(features.get('ch1_rms')).toBeCloseTo(Math.sqrt((9 + 16 + 9 + 16) / 4), 10);
    expect(features.get('ch1_mav')).toBeCloseTo(3.5, 10);
    expect(features.get('ch1_wl')).toBeCloseTo(21, 10);
  });

  it('does not count a sample resting exactly at zero as a crossing', () => {
    const features = extractFeatures([Float64Array.from([1, 0, 1])], config);
    expect(features.get('ch1_zc')).toBe(0);
  });

  it('counts a genuine sign change', () => {
    const features = extractFeatures([Float64Array.from([1, -1, 1])], config);
    expect(features.get('ch1_zc')).toBe(2);
  });

  it('zeroes AR and spectral features for a silent channel', () => {
    const features = extractFeatures([new Float64Array(400)], DEFAULT_FEATURE_CONFIG);
    for (const name of ['ch1_ar1', 'ch1_ar2', 'ch1_ar3', 'ch1_ar4', 'ch1_mdf', 'ch1_mnf']) {
      expect(features.get(name)).toBe(0);
    }
  });

  it('rejects channels of differing length', () => {
    expect(() =>
      extractFeatures([new Float64Array(10), new Float64Array(9)], config),
    ).toThrow();
  });
});

describe('featureVector', () => {
  it('is float32 and follows spec order', () => {
    const channels = [new Float64Array(400).fill(0.01), new Float64Array(400).fill(0.02)];
    const vector = featureVector(channels, DEFAULT_FEATURE_CONFIG);
    const named = extractFeatures(channels, DEFAULT_FEATURE_CONFIG);
    expect(vector).toBeInstanceOf(Float32Array);
    expect(vector.length).toBe(featureCount(2));
    expect(vector[0]).toBeCloseTo(named.get('ch1_rms')!, 6);
    expect(vector[17]).toBeCloseTo(named.get('ch2_rms')!, 6);
  });
});
```

- [x] **Step 2: Run and watch it fail**

```bash
cd d:/NeuroGrip && npx vitest run packages/core/test/features.test.ts
```
Expected: FAIL — cannot resolve `../src/spec.js`

- [x] **Step 3: Implement `spec.ts`**

```typescript
/**
 * Canonical feature specification. Mirrors services/research/neurogrip/spec.py.
 * Bump FEATURE_SPEC_VERSION in BOTH files and regenerate fixtures on any change.
 */

export const FEATURE_SPEC_VERSION = 1;
export const AR_ORDER = 4;
export const PSD_BINS = 6;

export const PER_CHANNEL_FEATURES = [
  'rms', 'mav', 'wl', 'zc', 'ssc',
  'ar1', 'ar2', 'ar3', 'ar4',
  'psd1', 'psd2', 'psd3', 'psd4', 'psd5', 'psd6',
  'mdf', 'mnf',
] as const satisfies readonly string[];

/**
 * Channel-major ordering with a fixed intra-channel order.
 * Deliberately not lexicographic: sorting places ch10_* before ch2_*, which
 * silently permutes the vector at ten or more channels. NinaPro DB2 has twelve.
 */
export function featureNames(nChannels: number): string[] {
  if (nChannels < 1) throw new RangeError('nChannels must be at least 1');
  const names: string[] = [];
  for (let channel = 1; channel <= nChannels; channel++) {
    for (const feature of PER_CHANNEL_FEATURES) names.push(`ch${channel}_${feature}`);
  }
  return names;
}

export function featureCount(nChannels: number): number {
  if (nChannels < 1) throw new RangeError('nChannels must be at least 1');
  return nChannels * PER_CHANNEL_FEATURES.length;
}
```

- [x] **Step 4: Implement `features.ts`**

```typescript
/**
 * Handcrafted sEMG feature extraction.
 *
 * A port of services/research/neurogrip/features.py. Pinned against it by
 * test/conformance.test.ts using golden vectors generated from Python.
 */

import {
  autocorrelate,
  bandMeans,
  levinsonDurbin,
  meanFrequency,
  medianFrequency,
  periodogram,
} from './dsp.js';
import { AR_ORDER, PER_CHANNEL_FEATURES, PSD_BINS, featureNames } from './spec.js';

export interface FeatureConfig {
  samplingRateHz: number;
  zeroCrossingThreshold: number;
  slopeSignThreshold: number;
}

export const DEFAULT_FEATURE_CONFIG: FeatureConfig = {
  samplingRateHz: 2000,
  zeroCrossingThreshold: 1e-5,
  slopeSignThreshold: 1e-5,
};

function assertWellFormed(channels: Float64Array[]): void {
  if (channels.length < 1) throw new RangeError('at least one channel is required');
  const width = channels[0]!.length;
  if (width < 3) throw new RangeError('each channel needs at least three samples');
  for (const channel of channels) {
    if (channel.length !== width) {
      throw new RangeError('all channels must have the same sample count');
    }
  }
}

/** Write one channel's 17 features into `out` starting at `offset`. */
function writeChannelFeatures(
  values: Float64Array,
  config: FeatureConfig,
  out: Float32Array,
  offset: number,
): void {
  const n = values.length;

  let sumSquares = 0;
  let sumAbs = 0;
  for (let i = 0; i < n; i++) {
    sumSquares += values[i]! * values[i]!;
    sumAbs += Math.abs(values[i]!);
  }

  let waveformLength = 0;
  let zeroCrossings = 0;
  for (let i = 0; i < n - 1; i++) {
    const delta = values[i + 1]! - values[i]!;
    waveformLength += Math.abs(delta);
    // Strict product-negative test: Math.sign(0) is 0, so comparing signs
    // would count a sample resting at zero as crossing both neighbours.
    if (values[i]! * values[i + 1]! < 0 && Math.abs(delta) >= config.zeroCrossingThreshold) {
      zeroCrossings++;
    }
  }

  // Sign test on the product (dimensionless); magnitude test on the differences,
  // which share units with the threshold. Comparing the threshold against the
  // product directly is dimensionally inconsistent.
  let slopeSignChanges = 0;
  for (let i = 1; i < n - 1; i++) {
    const back = values[i]! - values[i - 1]!;
    const forward = values[i]! - values[i + 1]!;
    if (
      back * forward > 0 &&
      Math.max(Math.abs(back), Math.abs(forward)) >= config.slopeSignThreshold
    ) {
      slopeSignChanges++;
    }
  }

  out[offset + 0] = Math.sqrt(sumSquares / n);
  out[offset + 1] = sumAbs / n;
  out[offset + 2] = waveformLength;
  out[offset + 3] = zeroCrossings;
  out[offset + 4] = slopeSignChanges;

  const ar = levinsonDurbin(autocorrelate(values, AR_ORDER), AR_ORDER);
  for (let i = 0; i < AR_ORDER; i++) out[offset + 5 + i] = ar[i]!;

  const { freqs, psd } = periodogram(values, config.samplingRateHz);
  const bands = bandMeans(psd, PSD_BINS);
  for (let i = 0; i < PSD_BINS; i++) out[offset + 5 + AR_ORDER + i] = bands[i]!;

  const spectralOffset = offset + 5 + AR_ORDER + PSD_BINS;
  out[spectralOffset] = medianFrequency(freqs, psd);
  out[spectralOffset + 1] = meanFrequency(freqs, psd);
}

/**
 * The hot path. Allocates one Float32Array and no Map, because this runs on
 * every 20 ms hop inside the inference worker.
 */
export function featureVector(
  channels: Float64Array[],
  config: FeatureConfig,
): Float32Array {
  assertWellFormed(channels);
  const stride = PER_CHANNEL_FEATURES.length;
  const out = new Float32Array(channels.length * stride);
  for (let c = 0; c < channels.length; c++) {
    writeChannelFeatures(channels[c]!, config, out, c * stride);
  }
  return out;
}

/** Named view over featureVector. For debugging and conformance, not the hot path. */
export function extractFeatures(
  channels: Float64Array[],
  config: FeatureConfig,
): Map<string, number> {
  const vector = featureVector(channels, config);
  const names = featureNames(channels.length);
  const out = new Map<string, number>();
  for (let i = 0; i < names.length; i++) out.set(names[i]!, vector[i]!);
  return out;
}
```

- [x] **Step 5: Update `packages/core/src/index.ts`**

```typescript
export const CORE_VERSION = '0.2.0';

export * from './dsp.js';
export * from './features.js';
export * from './spec.js';
```

- [x] **Step 6: Run and confirm green**

```bash
cd d:/NeuroGrip && npx vitest run packages/core/test/features.test.ts && npx tsc --noEmit -p packages/core/tsconfig.json
```
Expected: 8 passed, tsc exits 0

- [x] **Step 7: Commit**

```bash
cd d:/NeuroGrip
printf '%s\n' \
  'feat(core): port feature spec and extraction to TypeScript' '' \
  'featureVector takes channels directly and allocates no Map, because it runs' \
  'on every 20 ms hop inside the inference worker.' '' \
  'Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>' > /tmp/ng-msg.txt
git add packages/core/src/spec.ts packages/core/src/features.ts packages/core/src/index.ts packages/core/test/features.test.ts
git commit -F /tmp/ng-msg.txt
```

---

### Task 12: `windowing.ts` and `signalQuality.ts` — the streaming front end

**Files:**
- Create: `packages/core/src/windowing.ts`, `packages/core/src/signalQuality.ts`
- Modify: `packages/core/src/index.ts`
- Test: `packages/core/test/windowing.test.ts`, `packages/core/test/signalQuality.test.ts`

**Interfaces:**
- Consumes: nothing
- Produces:
  - `WindowConfig` interface — `windowMs`, `hopMs`, `samplingRateHz`; `DEFAULT_WINDOW_CONFIG`
  - `windowSamples(config): number`, `hopSamples(config): number`
  - `class MultiChannelRingBuffer` — `constructor(nChannels, capacitySamples)`, `push(frame: Float64Array[]): void`, `readLatestWindow(width: number): Float64Array[] | null`, `get available(): number`
  - `SignalQualityConfig` interface, `DEFAULT_SIGNAL_QUALITY_CONFIG`
  - `checkSignalQuality(channels, config): SignalQualityResult` — `{ ok: boolean; reason?: string; failedChannel?: number }`

The ring buffer is the browser's streaming front end: the source pushes frames at the sample rate and the inference loop reads the most recent 200 ms every 20 ms. It preallocates and never grows, because allocating inside a 50 Hz loop invites garbage-collection pauses that would show up directly in the P95 latency number this project treats as a safety property.

`checkSignalQuality` ports `services/research/neurogrip/signal_quality.py` verbatim — same thresholds, same order of checks, same reason strings — so a rejection in the browser is explicable by the Python-side test suite.

- [x] **Step 1: Write the failing tests**

```typescript
// packages/core/test/windowing.test.ts
import { describe, expect, it } from 'vitest';
import {
  DEFAULT_WINDOW_CONFIG,
  MultiChannelRingBuffer,
  hopSamples,
  windowSamples,
} from '../src/windowing.js';

describe('window geometry', () => {
  it('derives sample counts from milliseconds and rate', () => {
    expect(windowSamples(DEFAULT_WINDOW_CONFIG)).toBe(400);
    expect(hopSamples(DEFAULT_WINDOW_CONFIG)).toBe(40);
  });

  it('rejects non-positive geometry', () => {
    expect(() => windowSamples({ ...DEFAULT_WINDOW_CONFIG, windowMs: 0 })).toThrow();
    expect(() => hopSamples({ ...DEFAULT_WINDOW_CONFIG, hopMs: 0 })).toThrow();
  });
});

describe('MultiChannelRingBuffer', () => {
  it('returns null until a full window is available', () => {
    const buffer = new MultiChannelRingBuffer(2, 16);
    expect(buffer.readLatestWindow(4)).toBeNull();
    buffer.push([Float64Array.from([1, 2, 3]), Float64Array.from([4, 5, 6])]);
    expect(buffer.readLatestWindow(4)).toBeNull();
    buffer.push([Float64Array.from([7]), Float64Array.from([8])]);
    expect(buffer.readLatestWindow(4)).not.toBeNull();
  });

  it('returns the most recent samples in order', () => {
    const buffer = new MultiChannelRingBuffer(1, 8);
    buffer.push([Float64Array.from([1, 2, 3, 4, 5])]);
    expect(Array.from(buffer.readLatestWindow(3)![0]!)).toEqual([3, 4, 5]);
  });

  it('wraps correctly once capacity is exceeded', () => {
    const buffer = new MultiChannelRingBuffer(1, 4);
    buffer.push([Float64Array.from([1, 2, 3, 4, 5, 6])]);
    expect(Array.from(buffer.readLatestWindow(4)![0]!)).toEqual([3, 4, 5, 6]);
  });

  it('caps available at capacity', () => {
    const buffer = new MultiChannelRingBuffer(1, 4);
    buffer.push([Float64Array.from([1, 2, 3, 4, 5, 6])]);
    expect(buffer.available).toBe(4);
  });

  it('rejects a window wider than capacity', () => {
    const buffer = new MultiChannelRingBuffer(1, 4);
    expect(() => buffer.readLatestWindow(5)).toThrow();
  });

  it('rejects a frame with the wrong channel count', () => {
    const buffer = new MultiChannelRingBuffer(2, 8);
    expect(() => buffer.push([Float64Array.from([1])])).toThrow();
  });
});
```

```typescript
// packages/core/test/signalQuality.test.ts
import { describe, expect, it } from 'vitest';
import { DEFAULT_SIGNAL_QUALITY_CONFIG, checkSignalQuality } from '../src/signalQuality.js';

const good = () => [new Float64Array(400).fill(0.01), new Float64Array(400).fill(0.02)];

describe('checkSignalQuality', () => {
  it('accepts a healthy window', () => {
    expect(checkSignalQuality(good(), DEFAULT_SIGNAL_QUALITY_CONFIG).ok).toBe(true);
  });

  it('rejects a channel below the noise floor and names it', () => {
    const channels = good();
    channels[1] = new Float64Array(400);
    const result = checkSignalQuality(channels, DEFAULT_SIGNAL_QUALITY_CONFIG);
    expect(result.ok).toBe(false);
    expect(result.reason).toBe('channel RMS below noise floor');
    expect(result.failedChannel).toBe(1);
  });

  it('rejects a channel exceeding the expected EMG range', () => {
    const channels = good();
    channels[0]![10] = 99;
    const result = checkSignalQuality(channels, DEFAULT_SIGNAL_QUALITY_CONFIG);
    expect(result.ok).toBe(false);
    expect(result.reason).toBe('channel amplitude exceeds expected EMG range');
    expect(result.failedChannel).toBe(0);
  });

  it('rejects non-finite samples before any other check', () => {
    const channels = good();
    channels[0]![5] = Number.NaN;
    const result = checkSignalQuality(channels, DEFAULT_SIGNAL_QUALITY_CONFIG);
    expect(result.ok).toBe(false);
    expect(result.reason).toBe('channel contains non-finite values');
  });

  it('rejects an empty channel list', () => {
    expect(checkSignalQuality([], DEFAULT_SIGNAL_QUALITY_CONFIG).ok).toBe(false);
  });
});
```

- [x] **Step 2: Run and watch them fail**

```bash
cd d:/NeuroGrip && npx vitest run packages/core/test/windowing.test.ts packages/core/test/signalQuality.test.ts
```
Expected: FAIL — modules do not resolve

- [x] **Step 3: Implement `windowing.ts`**

```typescript
/**
 * Streaming front end: window geometry and a preallocated ring buffer.
 *
 * Mirrors services/research/neurogrip/windowing.py for geometry. The 20 ms hop
 * (rather than the report's 100 ms) is what makes progressive actuation
 * possible: at a 10 Hz decision rate there is nothing to accumulate between
 * decisions.
 */

export interface WindowConfig {
  windowMs: number;
  hopMs: number;
  samplingRateHz: number;
}

export const DEFAULT_WINDOW_CONFIG: WindowConfig = {
  windowMs: 200,
  hopMs: 20,
  samplingRateHz: 2000,
};

export function windowSamples(config: WindowConfig): number {
  if (config.windowMs <= 0) throw new RangeError('windowMs must be positive');
  return Math.trunc((config.windowMs * config.samplingRateHz) / 1000);
}

export function hopSamples(config: WindowConfig): number {
  if (config.hopMs <= 0) throw new RangeError('hopMs must be positive');
  return Math.trunc((config.hopMs * config.samplingRateHz) / 1000);
}

/**
 * Fixed-capacity multi-channel ring buffer.
 *
 * Preallocates and never grows. Allocating inside the 50 Hz inference loop
 * invites garbage-collection pauses, which would land directly in the P95
 * latency figure this project treats as a safety property.
 */
export class MultiChannelRingBuffer {
  private readonly buffers: Float64Array[];
  private readonly scratch: Float64Array[];
  private writeIndex = 0;
  private filled = 0;

  constructor(
    public readonly nChannels: number,
    public readonly capacitySamples: number,
  ) {
    if (nChannels < 1) throw new RangeError('nChannels must be at least 1');
    if (capacitySamples < 1) throw new RangeError('capacitySamples must be at least 1');
    this.buffers = Array.from({ length: nChannels }, () => new Float64Array(capacitySamples));
    this.scratch = Array.from({ length: nChannels }, () => new Float64Array(capacitySamples));
  }

  get available(): number {
    return this.filled;
  }

  push(frame: Float64Array[]): void {
    if (frame.length !== this.nChannels) {
      throw new RangeError(`expected ${this.nChannels} channels, received ${frame.length}`);
    }
    const count = frame[0]!.length;
    for (const channel of frame) {
      if (channel.length !== count) throw new RangeError('channels must be equal length');
    }

    for (let i = 0; i < count; i++) {
      const slot = (this.writeIndex + i) % this.capacitySamples;
      for (let c = 0; c < this.nChannels; c++) {
        this.buffers[c]![slot] = frame[c]![i]!;
      }
    }
    this.writeIndex = (this.writeIndex + count) % this.capacitySamples;
    this.filled = Math.min(this.capacitySamples, this.filled + count);
  }

  /**
   * The most recent `width` samples per channel, oldest first, or null if not
   * enough data has arrived. The returned arrays are reused scratch views and
   * are invalidated by the next call — copy them if you need to retain them.
   */
  readLatestWindow(width: number): Float64Array[] | null {
    if (width < 1) throw new RangeError('width must be positive');
    if (width > this.capacitySamples) {
      throw new RangeError('width exceeds buffer capacity');
    }
    if (this.filled < width) return null;

    const start = (this.writeIndex - width + this.capacitySamples) % this.capacitySamples;
    const out: Float64Array[] = [];
    for (let c = 0; c < this.nChannels; c++) {
      const view = this.scratch[c]!.subarray(0, width);
      for (let i = 0; i < width; i++) {
        view[i] = this.buffers[c]![(start + i) % this.capacitySamples]!;
      }
      out.push(view);
    }
    return out;
  }
}
```

- [x] **Step 4: Implement `signalQuality.ts`**

```typescript
/**
 * Signal-quality gating. A verbatim port of
 * services/research/neurogrip/signal_quality.py: same thresholds, same order of
 * checks, same reason strings, so a rejection in the browser is explicable by
 * the Python test suite.
 */

export interface SignalQualityConfig {
  minChannelRms: number;
  maxAbsAmplitude: number;
}

export const DEFAULT_SIGNAL_QUALITY_CONFIG: SignalQualityConfig = {
  minChannelRms: 1e-4,
  maxAbsAmplitude: 5.0,
};

export interface SignalQualityResult {
  ok: boolean;
  reason?: string;
  failedChannel?: number;
}

export function checkSignalQuality(
  channels: Float64Array[],
  config: SignalQualityConfig,
): SignalQualityResult {
  if (channels.length < 1) {
    return { ok: false, reason: 'input must contain at least one channel' };
  }

  for (let index = 0; index < channels.length; index++) {
    const values = channels[index]!;
    if (values.length < 1) {
      return { ok: false, reason: 'channel is empty', failedChannel: index };
    }

    let sumSquares = 0;
    let maxAbs = 0;
    for (let i = 0; i < values.length; i++) {
      const value = values[i]!;
      if (!Number.isFinite(value)) {
        return { ok: false, reason: 'channel contains non-finite values', failedChannel: index };
      }
      sumSquares += value * value;
      const magnitude = Math.abs(value);
      if (magnitude > maxAbs) maxAbs = magnitude;
    }

    if (Math.sqrt(sumSquares / values.length) < config.minChannelRms) {
      return { ok: false, reason: 'channel RMS below noise floor', failedChannel: index };
    }
    if (maxAbs > config.maxAbsAmplitude) {
      return {
        ok: false,
        reason: 'channel amplitude exceeds expected EMG range',
        failedChannel: index,
      };
    }
  }

  return { ok: true };
}
```

- [x] **Step 5: Re-export from `index.ts`**

```typescript
export const CORE_VERSION = '0.2.0';

export * from './dsp.js';
export * from './features.js';
export * from './signalQuality.js';
export * from './spec.js';
export * from './windowing.js';
```

- [x] **Step 6: Run and confirm green**

```bash
cd d:/NeuroGrip && npx vitest run && npx tsc --noEmit -p packages/core/tsconfig.json
```
Expected: all suites pass, tsc exits 0

- [x] **Step 7: Commit**

```bash
cd d:/NeuroGrip
printf '%s\n' \
  'feat(core): add ring buffer and signal-quality gate' '' \
  'The ring buffer preallocates and never grows: allocating inside the 50 Hz' \
  'inference loop would invite GC pauses that land in the P95 latency figure.' '' \
  'Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>' > /tmp/ng-msg.txt
git add packages/core/src/windowing.ts packages/core/src/signalQuality.ts packages/core/src/index.ts packages/core/test/windowing.test.ts packages/core/test/signalQuality.test.ts
git commit -F /tmp/ng-msg.txt
```

---

### Task 13: `conformance.test.ts` — the gate

**Files:**
- Create: `packages/core/test/conformance.test.ts`

**Interfaces:**
- Consumes: `fixtures/conformance/golden.json`, `@neurogrip/core` feature pipeline
- Produces: the pass/fail signal that governs every downstream phase

**This is the task the whole phase exists for.** If Python and TypeScript disagree on features, every model trained in Python sees a different input distribution than the browser feeds it at inference, and the failure is silent — accuracy simply degrades with no error anywhere. Nothing in Phase 1 may start until this is green.

The test reports the single worst-disagreeing feature by name when it fails. A bare "arrays differ" would leave the next engineer bisecting seventeen features by hand.

- [x] **Step 1: Write the test (it should fail only if the port is wrong)**

```typescript
// packages/core/test/conformance.test.ts
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { describe, expect, it } from 'vitest';
import { featureVector } from '../src/features.js';
import { FEATURE_SPEC_VERSION, PER_CHANNEL_FEATURES, featureNames } from '../src/spec.js';

const RTOL = 1e-5;
const ATOL = 1e-8;

interface GoldenCase {
  name: string;
  samplingRateHz: number;
  channels: number[][];
  featureNames: string[];
  featureVector: number[];
}

interface Golden {
  featureSpecVersion: number;
  perChannelFeatures: string[];
  config: {
    samplingRateHz: number;
    zeroCrossingThreshold: number;
    slopeSignThreshold: number;
  };
  cases: GoldenCase[];
}

const golden: Golden = JSON.parse(
  readFileSync(
    fileURLToPath(new URL('../../../fixtures/conformance/golden.json', import.meta.url)),
    'utf-8',
  ),
);

describe('Python/TypeScript feature conformance', () => {
  it('fixture was generated by the same spec version this build implements', () => {
    expect(golden.featureSpecVersion).toBe(FEATURE_SPEC_VERSION);
    expect(golden.perChannelFeatures).toEqual([...PER_CHANNEL_FEATURES]);
  });

  it('has cases to check', () => {
    expect(golden.cases.length).toBeGreaterThan(0);
  });

  for (const goldenCase of golden.cases) {
    it(`matches Python on "${goldenCase.name}"`, () => {
      const channels = goldenCase.channels.map((row) => Float64Array.from(row));
      const actual = featureVector(channels, {
        samplingRateHz: goldenCase.samplingRateHz,
        zeroCrossingThreshold: golden.config.zeroCrossingThreshold,
        slopeSignThreshold: golden.config.slopeSignThreshold,
      });

      const names = featureNames(channels.length);
      expect(names).toEqual(goldenCase.featureNames);
      expect(actual.length).toBe(goldenCase.featureVector.length);

      // Report the single worst disagreement by name. A bare "arrays differ"
      // would leave the next engineer bisecting seventeen features by hand.
      let worstName = '';
      let worstError = 0;
      let worstExpected = 0;
      let worstActual = 0;

      for (let i = 0; i < actual.length; i++) {
        const expectedValue = goldenCase.featureVector[i]!;
        const actualValue = actual[i]!;
        const tolerance = ATOL + RTOL * Math.abs(expectedValue);
        const error = Math.abs(actualValue - expectedValue) - tolerance;
        if (error > worstError) {
          worstError = error;
          worstName = names[i]!;
          worstExpected = expectedValue;
          worstActual = actualValue;
        }
      }

      expect(
        worstError,
        worstError > 0
          ? `${worstName}: python=${worstExpected} typescript=${worstActual}`
          : 'within tolerance',
      ).toBeLessThanOrEqual(0);
    });
  }
});
```

- [x] **Step 2: Run the gate**

```bash
cd d:/NeuroGrip && npm run test:conformance
```
Expected: every case passes.

**If a case fails**, the message names the offending feature. Debug in this order, because these are the divergences this port is actually likely to hit:
1. `*_psd*`, `*_mdf`, `*_mnf` — check `nextPow2` padding, the Hann window (symmetric, not periodic), the `1 .. bins-2` doubling range, and `array_split` band sizing.
2. `*_ar*` — check the Levinson-Durbin reversal index. Python's `previous[::-1]` corresponds to `previous[m - 1 - i]` in the TypeScript loop; getting this backwards is the classic port bug and it only shows at order ≥ 3.
3. `*_zc`, `*_ssc` — confirm both sides use the strict product tests, not sign comparison.
4. `*_rms`, `*_mav`, `*_wl` — a mismatch here means float32 versus float64 accumulation somewhere; Python casts input to float32 in `_as_channel_matrix` then computes in float64.

Do not widen `RTOL` to make a failure go away. The tolerance is the contract.

- [x] **Step 3: Prove the gate actually bites**

Temporarily break the port and confirm the test catches it:

```bash
cd d:/NeuroGrip
sed -i 's|out\[offset + 1\] = sumAbs / n;|out[offset + 1] = sumAbs / n * 1.001;|' packages/core/src/features.ts
npm run test:conformance
```
Expected: FAIL naming `ch1_mav`. Then revert:

```bash
cd d:/NeuroGrip && git checkout packages/core/src/features.ts && npm run test:conformance
```
Expected: PASS. A gate that has never been seen to fail is not known to work.

- [x] **Step 4: Commit**

```bash
cd d:/NeuroGrip
printf '%s\n' \
  'test(core): add Python/TypeScript feature conformance gate' '' \
  'Without this, a divergence between the training-time and inference-time' \
  'feature implementations degrades accuracy silently, with no error anywhere.' \
  'Nothing in Phase 1 starts until this is green.' '' \
  'Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>' > /tmp/ng-msg.txt
git add packages/core/test/conformance.test.ts
git commit -F /tmp/ng-msg.txt
```

---

### Task 14: CI and Phase 0 exit

**Files:**
- Create: `.github/workflows/ci.yml`
- Create: `CLAUDE.md`
- Modify: `README.md`

**Interfaces:**
- Consumes: everything above
- Produces: a CI run that fails on Python test failure, TypeScript test failure, type error, or fixture staleness

The fixture-staleness check is the important one. If someone edits `features.py` without regenerating `golden.json`, the conformance test keeps passing against a stale fixture and the gate becomes decorative. CI regenerates the fixture and fails if the working tree changes.

- [x] **Step 1: Write `.github/workflows/ci.yml`**

```yaml
name: CI

on:
  push:
    branches: [main]
  pull_request:

jobs:
  python:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.12'
      - run: pip install -e "./services/research[dev]"
      - run: python -m pytest services/research/tests -q

  typescript:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with:
          node-version: '24'
          cache: npm
      - run: npm ci
      - run: npx tsc --noEmit -p packages/core/tsconfig.json
      - run: npx vitest run

  fixtures-are-current:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.12'
      - run: pip install -e "./services/research[dev]"
      - name: Regenerate golden vectors
        run: python -m neurogrip.fixtures --out fixtures/conformance/golden.json
      - name: Fail if the committed fixture is stale
        run: |
          if ! git diff --exit-code -- fixtures/conformance/golden.json; then
            echo "::error::golden.json is stale. Run 'npm run fixtures' and commit the result."
            exit 1
          fi
```

- [x] **Step 2: Write `CLAUDE.md`**

This is the document the user asked to be able to refer back to every time. Phase 0 seeds it; each later phase appends its own section.

```markdown
# NeuroGrip

Real-time sEMG hand-gesture decoding for prosthetic control.
Harsh Bavaskar · Anisa D'Souza · Shruti Shanklesha — BTECH CAP 501.

**Research prototype. Not a medical device. Not validated for clinical use.**

## What this is

A dual-mode application (prosthesis wearer + clinician) built around four novel
mechanisms. See `docs/superpowers/specs/2026-09-04-neurogrip-design.md` for the
full design and the prior-art analysis behind each.

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
|---|---|
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

Bump `FEATURE_SPEC_VERSION` in **both** `neurogrip/__init__.py` and
`packages/core/src/spec.ts` whenever the feature set or its ordering changes.

Never widen the conformance tolerance to make a failure pass. The tolerance is
the contract.

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
  decisions; at a 10 Hz decision rate there is none. Costs 5x the inference
  calls, each under 2 ms, so the 10 ms P95 budget still holds.
- **No `scipy.signal.welch`.** Its detrending, segment averaging and scaling
  conventions are a porting hazard for the browser runtime and buy nothing at a
  single 200 ms window. The periodogram is hand-implemented in both languages.
- **Feature ordering is channel-major, never sorted.** `sorted()` places
  `ch10_rms` before `ch2_rms`; NinaPro DB2 has twelve channels.
```

- [x] **Step 3: Rewrite `README.md`**

Replace the file wholesale. The current one documents `pclm`, the placeholder model and the FastAPI demo payload, none of which still exist.

```markdown
# NeuroGrip

Real-time sEMG hand-gesture decoding for prosthetic control.

**Research prototype. Not a medical device. Not validated for clinical use.**

Harsh Bavaskar · Anisa D'Souza · Shruti Shanklesha
BTECH CAP 501, Project Life Cycle Management — ATLAS SkillTech / uGDX

## Status

Phase 0 (Foundations) complete: biophysical sEMG simulator, feature pipeline in
Python and TypeScript, and a conformance gate pinning them together. No trained
model or user interface yet — those are Phase 1.

The project runs entirely on simulated data. No dataset download or
registration is needed to build, test, or demo it.

## Quick start

```bash
pip install -e "./services/research[dev]"
python -m pytest services/research/tests -q

npm install
npm test
```

See `CLAUDE.md` for architecture, the conformance rule, and full commands.
```

- [x] **Step 4: Run the complete Phase 0 verification**

```bash
cd d:/NeuroGrip
./.venv/Scripts/python.exe -m pytest services/research/tests -q
npm run typecheck
npm test
npm run fixtures && git diff --exit-code -- fixtures/conformance/golden.json && echo "fixtures current"
npm run test:conformance
```
Expected: all green, and `fixtures current` printed.

- [x] **Step 5: Commit**

```bash
cd d:/NeuroGrip
printf '%s\n' \
  'ci: add CI with a fixture-staleness gate; document Phase 0' '' \
  'CI regenerates golden.json and fails if the committed copy differs. Without' \
  'that check, editing features.py without regenerating fixtures would leave' \
  'the conformance test passing against a stale fixture.' '' \
  'Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>' > /tmp/ng-msg.txt
git add .github/workflows/ci.yml CLAUDE.md README.md
git commit -F /tmp/ng-msg.txt
```

---

## Phase 0 exit criteria

All must hold before Phase 1 begins:

- [x] `python -m pytest services/research/tests -q` — green
- [x] `npm test` — green
- [x] `npm run typecheck` — clean
- [x] `npm run test:conformance` — green, and observed to fail when the port is deliberately broken (Task 13 Step 3)
- [x] `npm run fixtures` leaves the working tree clean
- [x] The simulator produces ten distinguishable gestures, fatigue measurably lowers median frequency, and electrode shift measurably changes the channel amplitude profile
- [x] `CLAUDE.md` exists and states the conformance rule
- [x] No references to `pclm` remain anywhere in the tree

---

## Self-review

**Spec coverage.** This plan implements spec §4 "highest-risk constraint"
(conformance harness), the simulator and design-token prerequisites of Phase 0,
and the `features.py` defects identified in spec §1. Deliberately **not** in
scope here and deferred to their own plans: design tokens and the icon
construction sheet (Phase 1, where the first pixel is drawn), the auto-download
dataset loader and NinaPro `.mat` loader (Phase 1, alongside training), and all
four pillars' algorithms (Phases 1–3). Phase 0 delivers the substrate every one
of those depends on.

**Type consistency.** `FeatureConfig` fields are `sampling_rate_hz` /
`zero_crossing_threshold` / `slope_sign_threshold` in Python and the camelCase
equivalents in TypeScript; the fixture JSON uses camelCase and the Python
generator converts explicitly. `feature_vector` takes `(features, n_channels)`
in Python but `(channels, config)` in TypeScript — a deliberate divergence,
documented in Task 11, because the browser hot path must not allocate a Map.
`featureNames` / `feature_names` agree on output exactly, which is what the
conformance test asserts.

**Known simplification.** `MultiChannelRingBuffer.readLatestWindow` returns
reused scratch views rather than copies. This is correct for the single-consumer
inference loop and is documented in the method's doc comment, but a second
consumer would silently observe mutated data. Phase 1 must not add one without
copying.
