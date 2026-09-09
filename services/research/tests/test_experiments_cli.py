"""CLI dispatch tests.

These exist because of a real defect: an edit to `main()` silently failed to
apply, leaving every subcommand running `compare`. Nothing caught it, because
the expensive experiment functions were never exercised from the CLI in a test.
These tests pin the dispatch and the argument validation without running any
experiment, by stubbing the functions the CLI calls.
"""

from __future__ import annotations

import pathlib
import sys

import pytest

from neurogrip import experiments


@pytest.fixture
def spy(monkeypatch):
    calls: dict[str, object] = {}

    def fake_compare(config, calibration_folds=3, posteriors_stem=None):
        calls["compare"] = config
        calls["posteriors_stem"] = posteriors_stem
        return {"best_model": "stub", "models": {}, "paired_significance": {}}

    def fake_export(config, model_name, out_dir):
        calls["export"] = (config, model_name, out_dir)
        return {"model": model_name}

    monkeypatch.setattr(experiments, "compare_models", fake_compare)
    monkeypatch.setattr(experiments, "export_best", fake_export)
    return calls


def _run(monkeypatch, *argv: str) -> None:
    monkeypatch.setattr(sys, "argv", ["neurogrip.experiments", *argv])
    experiments.main()


def test_compare_dispatches_to_compare_only(monkeypatch, spy, tmp_path):
    _run(monkeypatch, "compare", "--out", str(tmp_path / "c.json"))
    assert "compare" in spy
    assert "export" not in spy


def test_export_dispatches_to_export_only(monkeypatch, spy, tmp_path):
    """The regression this file exists for."""
    _run(monkeypatch, "export", "--out", str(tmp_path / "models"))
    assert "export" in spy
    assert "compare" not in spy


def test_export_passes_the_requested_model(monkeypatch, spy, tmp_path):
    _run(monkeypatch, "export", "--out", str(tmp_path / "m"), "--model", "lda")
    _, model_name, _ = spy["export"]
    assert model_name == "lda"


def test_export_creates_the_output_directory(monkeypatch, spy, tmp_path):
    target = tmp_path / "nested" / "models"
    _run(monkeypatch, "export", "--out", str(target))
    assert target.is_dir()


def test_compare_writes_json_to_the_requested_path(monkeypatch, spy, tmp_path):
    target = tmp_path / "out" / "result.json"
    _run(monkeypatch, "compare", "--out", str(target))
    assert target.exists()


def test_compare_rejects_a_directory_target(monkeypatch, spy, tmp_path):
    with pytest.raises(SystemExit):
        _run(monkeypatch, "compare", "--out", str(tmp_path / "adir"))


def test_export_rejects_a_file_target(monkeypatch, spy, tmp_path):
    with pytest.raises(SystemExit):
        _run(monkeypatch, "export", "--out", str(tmp_path / "afile.onnx"))


def test_corpus_flags_reach_the_experiment(monkeypatch, spy, tmp_path):
    _run(
        monkeypatch, "compare", "--out", str(tmp_path / "c.json"),
        "--subjects", "3", "--reps", "2", "--seconds", "0.5",
    )
    config = spy["compare"]
    assert config.n_subjects == 3
    assert config.reps_per_gesture == 2
    assert config.rep_seconds == 0.5


def test_unknown_experiment_is_rejected(monkeypatch, spy):
    with pytest.raises(SystemExit):
        _run(monkeypatch, "train-a-transformer")


def test_defaults_do_not_require_out(monkeypatch, spy, tmp_path):
    monkeypatch.chdir(tmp_path)
    _run(monkeypatch, "compare")
    assert (tmp_path / "artifacts" / "model_comparison.json").exists()


def test_posteriors_flag_reaches_the_experiment(monkeypatch, spy, tmp_path):
    """The posterior export is what TTUM is computed from, and it is opt-in.

    A flag that parsed but never reached compare_models would leave the TTUM
    run reading a stale file from a previous corpus, which is worse than
    reading none: the numbers would be wrong rather than absent.
    """
    stem = tmp_path / "posteriors"
    _run(monkeypatch, "compare", "--out", str(tmp_path / "c.json"), "--posteriors", str(stem))
    assert spy["posteriors_stem"] == stem


def test_posteriors_are_not_written_unless_asked(monkeypatch, spy, tmp_path):
    _run(monkeypatch, "compare", "--out", str(tmp_path / "c.json"))
    assert spy["posteriors_stem"] is None
