import numpy as np
import pytest

from neurogrip.datasets import CorpusConfig, build_corpus
from neurogrip.export import (
    ParityResult,
    export_to_onnx,
    load_session,
    predict_onnx,
    verify_parity,
)
from neurogrip.train import fit_final_model, model_by_name


@pytest.fixture(scope="module")
def tiny_corpus():
    return build_corpus(
        CorpusConfig(n_subjects=2, reps_per_gesture=2, rep_seconds=0.3, n_electrodes=8)
    )


@pytest.fixture(scope="module")
def fitted(tiny_corpus):
    return fit_final_model(tiny_corpus, model_by_name("lda"), calibration_folds=2)


@pytest.fixture(scope="module")
def exported(tmp_path_factory, tiny_corpus, fitted):
    path = tmp_path_factory.mktemp("onnx") / "decoder.onnx"
    return export_to_onnx(fitted, tiny_corpus.n_features, path)


def test_export_writes_a_loadable_model(exported):
    assert exported.exists()
    assert exported.stat().st_size > 0
    assert load_session(exported) is not None


def test_onnx_emits_probabilities_as_a_plain_tensor(exported, tiny_corpus):
    session = load_session(exported)
    labels, probabilities = predict_onnx(session, tiny_corpus.features[:8])
    assert labels.shape == (8,)
    assert probabilities.shape == (8, len(tiny_corpus.gestures))
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, rtol=1e-4)


def test_onnx_matches_sklearn_probabilities(exported, fitted, tiny_corpus):
    """The parity gate. A divergence here means the browser scores differently
    from the model that was evaluated, and the LOSO number stops being true of
    what actually ships."""
    result = verify_parity(fitted, exported, tiny_corpus.features[:200])
    assert isinstance(result, ParityResult)
    assert result.ok, (
        f"max abs diff {result.max_abs_diff:.3e} exceeds tolerance; "
        f"worst sample {result.worst_sample}"
    )


def test_parity_reports_a_mismatch_when_probabilities_are_perturbed(
    exported, fitted, tiny_corpus
):
    """A gate never observed failing is not known to work."""

    class Perturbed:
        def predict_proba(self, features):
            probabilities = fitted.predict_proba(features)
            probabilities[:, 0] += 0.05
            return probabilities

    result = verify_parity(Perturbed(), exported, tiny_corpus.features[:50])
    assert not result.ok
    assert result.max_abs_diff > 0.01


def test_onnx_labels_agree_with_sklearn(exported, fitted, tiny_corpus):
    session = load_session(exported)
    labels, _ = predict_onnx(session, tiny_corpus.features[:100])
    assert np.array_equal(labels, fitted.predict(tiny_corpus.features[:100]))


def test_export_rejects_a_feature_count_mismatch(tmp_path, fitted):
    with pytest.raises(ValueError):
        export_to_onnx(fitted, 0, tmp_path / "bad.onnx")
