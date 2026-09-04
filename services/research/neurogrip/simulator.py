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
#
# Gestures must differ in PATTERN - which muscle dominates - never by a uniform
# amplitude offset. Per-subject amplitude gain spans 0.7-1.4x, so two classes
# separated only by overall level are mathematically indistinguishable under
# leave-one-subject-out, no matter how good the decoder is.
#
# test_no_two_gestures_are_gain_degenerate enforces this: cosine similarity
# between any two non-rest synergy vectors must stay below 0.98. Cosine is
# scale-invariant, so it detects gain-only separation exactly. `rest` is exempt
# because it is legitimately distinguished by amplitude - at 0.02 excitation no
# motor unit passes its recruitment threshold, so rest is baseline noise.
#
# fist/spherical_grip and pinch/point remain the deliberately hard pairs: a
# trivially separable vocabulary would give Pillars 1 and 3 nothing to work on.
GESTURE_SYNERGIES: dict[str, tuple[float, ...]] = {
    #                    fcr    fds    fcu    ecu    edc    ecr
    "rest":            (0.02,  0.02,  0.02,  0.02,  0.02,  0.02),
    "fist":            (0.45,  0.95,  0.50,  0.25,  0.10,  0.20),
    "open_hand":       (0.10,  0.08,  0.10,  0.45,  0.95,  0.50),
    "pinch":           (0.25,  0.55,  0.12,  0.20,  0.25,  0.18),
    "point":           (0.20,  0.50,  0.25,  0.30,  0.65,  0.25),
    "wrist_flexion":   (0.90,  0.25,  0.85,  0.08,  0.08,  0.10),
    "wrist_extension": (0.10,  0.10,  0.08,  0.85,  0.30,  0.90),
    "thumb_up":        (0.60,  0.15,  0.10,  0.15,  0.25,  0.65),
    "two_finger":      (0.20,  0.55,  0.50,  0.20,  0.20,  0.15),
    "spherical_grip":  (0.55,  0.65,  0.55,  0.45,  0.40,  0.50),
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
    # Baseline instrumentation noise, in volts. Real surface-EMG baseline sits
    # around 10-20 uV; a strong contraction reaches a few hundred uV. Setting
    # this too high buries the motor-unit signal in noise.
    noise_std: float = 1.5e-5
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
