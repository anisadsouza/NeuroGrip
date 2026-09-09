import json

import numpy as np
import pytest

from neurogrip import FEATURE_SPEC_VERSION
from neurogrip.replay import VOLTS_PER_COUNT, build_bundle, write_bundle
from neurogrip.simulator import GESTURES


@pytest.fixture(scope="module")
def bundle():
    return build_bundle(duration_s=0.5)


def test_manifest_describes_the_binary(bundle):
    manifest, samples = bundle
    assert manifest["gestures"] == list(GESTURES)
    assert samples.shape == (
        len(GESTURES),
        manifest["nChannels"],
        manifest["samplesPerGesture"],
    )
    assert samples.dtype == np.int16


def test_manifest_pins_the_feature_spec_version(bundle):
    """A replay recorded under a different feature spec would feed the decoder
    a distribution it was not trained on."""
    manifest, _ = bundle
    assert manifest["featureSpecVersion"] == FEATURE_SPEC_VERSION


def test_quantisation_round_trips_within_one_count(bundle):
    _, samples = bundle
    volts = samples.astype(np.float64) * VOLTS_PER_COUNT
    # Every value must be recoverable to better than half a count.
    recovered = np.round(volts / VOLTS_PER_COUNT)
    assert np.array_equal(recovered, samples.astype(np.float64))


def test_no_clipping_headroom_is_reported(bundle):
    manifest, samples = bundle
    assert manifest["peakCounts"] == int(np.max(np.abs(samples)))
    assert manifest["peakCounts"] < np.iinfo(np.int16).max
    # At least 4x headroom, or the scale is too tight for another subject.
    assert manifest["peakCounts"] < np.iinfo(np.int16).max / 4


def test_replayed_amplitudes_are_physiological(bundle):
    """Rest is baseline noise; contractions reach hundreds of microvolts."""
    manifest, samples = bundle
    volts = samples.astype(np.float64) * VOLTS_PER_COUNT
    rms = np.sqrt(np.mean(volts**2, axis=(1, 2)))
    rest_index = manifest["gestures"].index("rest")

    assert 5e-6 < rms[rest_index] < 5e-5, "rest should be baseline noise"
    active = np.delete(rms, rest_index)
    assert np.all(active > 5e-5), "contractions should be well above baseline"
    assert np.all(active < 2e-3), "contractions should stay in the sEMG range"


def test_bundle_is_deterministic():
    first, first_samples = build_bundle(duration_s=0.3)
    second, second_samples = build_bundle(duration_s=0.3)
    assert first == second
    assert np.array_equal(first_samples, second_samples)


def test_write_bundle_emits_both_files(tmp_path):
    manifest = write_bundle(tmp_path)
    binary = tmp_path / "emg-replay.bin"
    written = json.loads((tmp_path / "emg-replay.json").read_text(encoding="utf-8"))

    assert binary.exists()
    assert written == manifest
    expected = (
        len(manifest["gestures"])
        * manifest["nChannels"]
        * manifest["samplesPerGesture"]
        * 2
    )
    assert manifest["byteLength"] == expected == binary.stat().st_size


def test_binary_is_little_endian_and_reads_back(tmp_path):
    manifest = write_bundle(tmp_path)
    raw = (tmp_path / "emg-replay.bin").read_bytes()
    values = np.frombuffer(raw, dtype="<i2")
    assert values.size == (
        len(manifest["gestures"])
        * manifest["nChannels"]
        * manifest["samplesPerGesture"]
    )
    reshaped = values.reshape(
        len(manifest["gestures"]), manifest["nChannels"], manifest["samplesPerGesture"]
    )
    _, samples = build_bundle()
    assert np.array_equal(reshaped, samples)


def test_manifest_states_that_the_signal_is_simulated(tmp_path):
    manifest = write_bundle(tmp_path)
    assert "simulated" in manifest["caveat"].lower()
    assert manifest["source"] == "neurogrip.simulator"
