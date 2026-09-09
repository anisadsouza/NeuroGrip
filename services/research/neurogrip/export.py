"""ONNX export and the Python-to-ONNX parity gate.

Python trains the decoder; ONNX Runtime Web runs it in the browser. If the two
disagree, the leave-one-subject-out figure stops being true of the thing that
actually ships, and the discrepancy is silent -- exactly the failure mode the
Python/TypeScript feature conformance gate exists to prevent, one stage further
down the pipeline. `verify_parity` closes it.

ZipMap is disabled on export so the model emits probabilities as a plain
(n_samples, n_classes) tensor rather than a sequence of dictionaries. The
dictionary form is awkward in Python and worse in the browser runtime.
"""

from __future__ import annotations

import pathlib
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

DEFAULT_RTOL = 1e-4
DEFAULT_ATOL = 1e-5


class SupportsPredictProba(Protocol):
    def predict_proba(self, features: np.ndarray) -> np.ndarray: ...


def export_to_onnx(
    model: Any, n_features: int, path: pathlib.Path, opset: int = 15
) -> pathlib.Path:
    """Convert a fitted sklearn pipeline to ONNX and write it to `path`."""
    if n_features < 1:
        raise ValueError("n_features must be at least 1")

    from skl2onnx import to_onnx

    sample = np.zeros((1, n_features), dtype=np.float32)
    onnx_model = to_onnx(
        model,
        sample,
        target_opset=opset,
        options={"zipmap": False},
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(onnx_model.SerializeToString())
    return path


def load_session(path: pathlib.Path) -> Any:
    """Open an ONNX Runtime session, CPU only.

    CPU is deliberate: the deployment target is a browser WASM runtime and a
    low-cost embedded controller, so a GPU-assisted number here would be
    misleading about the latency the system will actually see.
    """
    import onnxruntime

    return onnxruntime.InferenceSession(
        str(path), providers=["CPUExecutionProvider"]
    )


def predict_onnx(session: Any, features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Run inference. Returns (labels, probabilities)."""
    batch = np.ascontiguousarray(features, dtype=np.float32)
    if batch.ndim != 2:
        raise ValueError("features must have shape (n_samples, n_features)")

    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: batch})

    labels = np.asarray(outputs[0]).ravel()
    probabilities = np.asarray(outputs[1], dtype=np.float64)
    if probabilities.ndim != 2:
        raise ValueError(
            "expected a 2-D probability tensor; export with zipmap disabled"
        )
    return labels, probabilities


@dataclass(frozen=True)
class ParityResult:
    ok: bool
    max_abs_diff: float
    max_rel_diff: float
    n_samples: int
    worst_sample: int
    rtol: float
    atol: float

    def summary(self) -> dict[str, object]:
        return {
            "ok": self.ok,
            "max_abs_diff": float(f"{self.max_abs_diff:.3e}"),
            "max_rel_diff": float(f"{self.max_rel_diff:.3e}"),
            "n_samples": self.n_samples,
            "rtol": self.rtol,
            "atol": self.atol,
        }


def verify_parity(
    model: SupportsPredictProba,
    onnx_path: pathlib.Path,
    features: np.ndarray,
    rtol: float = DEFAULT_RTOL,
    atol: float = DEFAULT_ATOL,
) -> ParityResult:
    """Compare sklearn and ONNX probabilities over a batch of real windows."""
    reference = np.asarray(model.predict_proba(features), dtype=np.float64)
    _, actual = predict_onnx(load_session(onnx_path), features)

    if reference.shape != actual.shape:
        raise ValueError(
            f"shape mismatch: sklearn {reference.shape} vs onnx {actual.shape}"
        )

    absolute = np.abs(actual - reference)
    relative = absolute / np.maximum(np.abs(reference), 1e-12)
    worst = int(np.unravel_index(int(absolute.argmax()), absolute.shape)[0])

    return ParityResult(
        ok=bool(np.all(absolute <= atol + rtol * np.abs(reference))),
        max_abs_diff=float(absolute.max()),
        max_rel_diff=float(relative.max()),
        n_samples=int(reference.shape[0]),
        worst_sample=worst,
        rtol=rtol,
        atol=atol,
    )
