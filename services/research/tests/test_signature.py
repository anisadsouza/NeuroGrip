import json
import pathlib

import numpy as np
import pytest

from neurogrip.signature import DEFAULT_PATH, build_payload, channel_rms_by_gesture
from neurogrip.simulator import GESTURES

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
SIGNATURE_PATH = REPO_ROOT / DEFAULT_PATH


def _committed() -> dict:
    if not SIGNATURE_PATH.exists():
        pytest.skip(f"signature not committed at {SIGNATURE_PATH}")
    return json.loads(SIGNATURE_PATH.read_text(encoding="utf-8"))


def test_simulator_output_matches_the_committed_signature():
    """Guards against silent simulator drift.

    The conformance fixtures store their own input waveforms, so they cannot
    detect a change to the simulator. This can. If it fails and the change was
    intended, regenerate with `python -m neurogrip.signature` and review the
    diff -- do not simply re-bless it, because every downstream accuracy number
    was produced by the old simulator.
    """
    committed = _committed()["channel_rms"]
    current = channel_rms_by_gesture()

    assert set(committed) == set(current) == set(GESTURES)

    for gesture in GESTURES:
        np.testing.assert_allclose(
            current[gesture],
            committed[gesture],
            rtol=1e-3,
            atol=1e-12,
            err_msg=(
                f"simulator output for '{gesture}' changed. If intended, run "
                f"'python -m neurogrip.signature' and review the diff."
            ),
        )


def test_signature_generation_is_deterministic():
    assert channel_rms_by_gesture() == channel_rms_by_gesture()


def test_payload_records_the_seeds_that_produced_it():
    payload = build_payload()
    for key in ("subject_seed", "simulate_seed", "duration_s", "n_electrodes"):
        assert key in payload


def test_write_signature_round_trips(tmp_path):
    target = tmp_path / "sig.json"
    from neurogrip.signature import write_signature

    written = write_signature(target)
    assert json.loads(target.read_text(encoding="utf-8")) == written
