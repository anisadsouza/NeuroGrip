import numpy as np
import pytest

from neurogrip.ninapro_rest_calibration import (
    _rest_window_percentiles,
    build_report,
)


def test_rest_calibration_report_combines_recordings_for_each_subject(
    tmp_path, monkeypatch
):
    from neurogrip import ninapro_rest_calibration

    first = tmp_path / "S12_E1_A1.mat"
    second = tmp_path / "S12_E2_A1.mat"
    first.touch()
    second.touch()

    def fake_load_session(path):
        baseline = 100e-6 if path == first else 0
        emg = np.full((8, 20), 10e-6, dtype=np.float32)
        emg[0] = baseline
        labels = np.zeros(20, dtype=np.int16)
        return emg, labels, np.ones(20, dtype=np.int16)

    monkeypatch.setattr(ninapro_rest_calibration, "load_session", fake_load_session)

    report = build_report(tmp_path)

    assert len(report["subjects"]) == 1
    subject = report["subjects"][0]
    assert subject["subject"] == "S12"
    assert subject["sessions"] == [first.name, second.name]
    assert subject["restSamples"] == 40
    assert subject["flaggedChannels"] == [1]
    assert subject["channelRms"][0] == pytest.approx(100e-6 / (2**0.5))


def test_rest_calibration_report_requires_local_recordings(tmp_path):
    with pytest.raises(FileNotFoundError, match="no NinaPro DB2 session files"):
        build_report(tmp_path)


def test_rest_window_percentiles_exclude_windows_with_non_rest_labels():
    emg = np.full((8, 800), 10e-6, dtype=np.float32)
    emg[0, 400:] = 100e-6
    labels = np.zeros(800, dtype=np.int16)
    labels[400:] = 1

    result = _rest_window_percentiles(emg, labels)

    assert result is not None
    assert result["nWindows"] == 1
    assert result["p50Microvolts"][0] == pytest.approx(10)
