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
