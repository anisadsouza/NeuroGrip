"""Export out-of-fold posterior sequences, grouped into trials.

The evidence accumulator that turns posteriors into motion exists only in
TypeScript, and porting it to Python would create a second implementation
owing a second conformance gate -- the same argument that keeps the simulator
out of the browser. So the Time-to-Useful-Motion metric is computed by feeding
these sequences through the one real accumulator, and this module is the seam:
Python decides what the decoder believed, TypeScript decides when the hand
would have moved.

The posteriors are out of fold. Each window's probabilities come from a model
that never saw that window's subject, which is what makes a TTUM figure derived
from them a statement about a new wearer rather than about the training set.
"""

from __future__ import annotations

import json
import pathlib

import numpy as np

from neurogrip import FEATURE_SPEC_VERSION
from neurogrip.datasets import Corpus

#: Bumped when the layout of the .bin or the manifest's fields change.
POSTERIOR_FORMAT_VERSION = 1


def group_trials(corpus: Corpus) -> list[dict]:
    """Contiguous runs of one (subject, gesture, repetition) in corpus order.

    `build_corpus` appends window by window through nested loops, so a trial's
    windows are already adjacent and in time order. That is an invariant of how
    the corpus is built rather than something the array records, and a shuffle
    added upstream would break it without changing any shape -- so it is
    asserted rather than assumed.
    """
    trials: list[dict] = []
    keys = list(zip(corpus.subjects.tolist(), corpus.labels.tolist(), corpus.reps.tolist()))

    start = 0
    for index in range(1, len(keys) + 1):
        if index == len(keys) or keys[index] != keys[start]:
            subject, label, rep = keys[start]
            trials.append(
                {
                    "subject": int(subject),
                    "label": int(label),
                    "rep": int(rep),
                    "start": int(start),
                    "length": int(index - start),
                }
            )
            start = index
    return trials


def write_posteriors(
    corpus: Corpus,
    probabilities: np.ndarray,
    model: str,
    out_stem: pathlib.Path,
) -> dict:
    """Write `<stem>.bin` and `<stem>.json`, and return the manifest.

    Float32 rather than float64: the accumulator takes a log ratio of one
    probability to another, and the seventh significant figure of a posterior
    does not survive that. Halving the file is worth more than a precision
    nobody can use.
    """
    if probabilities.shape[0] != corpus.n_windows:
        raise ValueError(
            f"probabilities cover {probabilities.shape[0]} windows, "
            f"corpus has {corpus.n_windows}"
        )

    trials = group_trials(corpus)
    covered = sum(trial["length"] for trial in trials)
    if covered != corpus.n_windows:
        raise ValueError(f"trials cover {covered} windows, corpus has {corpus.n_windows}")

    out_stem.parent.mkdir(parents=True, exist_ok=True)
    binary = out_stem.with_suffix(".bin")
    binary.write_bytes(np.ascontiguousarray(probabilities, dtype=np.float32).tobytes())

    manifest = {
        "formatVersion": POSTERIOR_FORMAT_VERSION,
        "featureSpecVersion": FEATURE_SPEC_VERSION,
        "model": model,
        "outOfFold": True,
        "gestures": list(corpus.gestures),
        "nWindows": int(corpus.n_windows),
        "nClasses": int(probabilities.shape[1]),
        "dtype": "float32",
        "layout": "row-major (nWindows, nClasses), corpus order",
        "windowMs": 200,
        "hopMs": 20,
        "trials": trials,
        "source": "simulator",
        "caveat": (
            "Every repetition holds a constant excitation for its whole "
            "duration: the simulator applies no onset envelope. Evidence "
            "therefore accrues from an already-active steady contraction, so "
            "any time-to-motion derived from these sequences measures decoder "
            "accrual and not a wearer's reaction time. It is a lower bound on "
            "what a person would experience."
        ),
    }
    out_stem.with_suffix(".json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
