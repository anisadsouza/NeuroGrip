"""End-to-end error routing on simulated signals (Pillars 3 and 4 together).

`attribution.attribution_demo` hand-builds four feature vectors to check the
router's logic. This module does what the spec's end-to-end demo asks for
instead: generate a cued session through the biophysical simulator, decode it,
take the decoder's actual misclassifications, compute every evidence channel
from the signals, and route each error. Behavioural routes are passed on to
the anatomical coach (`coaching.py`) for a sentence.

Four sessions, one injected fault each:

- **behavioural** -- one cued gesture (`point`) is performed with excess wrist
  flexor activity; everything else is clean.
- **drift** -- the whole session is strongly fatigued (MUAP widening, so
  median frequency falls), on a decoder trained fresh.
- **physiological** -- a subject who cannot differentiate `pinch` from
  `point`: both cues are produced as the same muscle pattern, in training and
  in the session. (A finger-atrophy subject at 30% reach was tried first; its
  own decoder still scored 98.8% and produced one error, so the fault never
  materialised.)
- **model capacity** -- a clean session decoded by a deliberately
  under-capacity model (two principal components before LDA), while the
  stronger RBF-SVM reference is trained on the same data.

Every threshold the router applies is the one `attribution.py` already
declares; the CUSUM and median-frequency thresholds below match the values the
synthetic demo used. Nothing here was tuned until a session routed correctly,
and a session that routes wrongly is reported as wrong.

Simulator only. The session is cued, so the intended gesture is known -- that
is what makes the evidence computable, and it is also why this describes a
drill or calibration session, not free use.
"""

from __future__ import annotations

import dataclasses
from collections import Counter

import numpy as np
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from neurogrip.anatomy import MUSCLE_GROUPS, MUSCLES
from neurogrip.attribution import ErrorObservation, attribute_error
from neurogrip.coaching import LinearDecoder, _attempt, _windows, explain_error
from neurogrip.simulator import GESTURES, GESTURE_SYNERGIES, SimulatorConfig, make_subject, simulate_excitation
from neurogrip.spec import PER_CHANNEL_FEATURES

#: One-sided CUSUM allowance and alarm level, in units of the training
#: distribution of distance-to-own-centroid. h = 8 is the value the synthetic
#: demo declared.
CUSUM_ALLOWANCE = 0.5
CUSUM_THRESHOLD = 8.0

#: Median-frequency shift that counts as full drift evidence. Same as the demo.
MDF_THRESHOLD_HZ = 20.0

BEHAVIOURAL_GESTURE = "point"
BEHAVIOURAL_GROUP = "wrist_flexor"
BEHAVIOURAL_STEP = 0.35
DRIFT_FATIGUE = 0.9

_MDF_INDEX = PER_CHANNEL_FEATURES.index("mdf")


def other_gesture_cusum(
    z_scores: np.ndarray, labels: np.ndarray, index: int, intended: int
) -> float:
    """CUSUM up to `index` over windows cued as gestures other than `intended`.

    Drift displaces every gesture; a behavioural fault displaces the one being
    attempted. Reading drift evidence off the gesture in error would count the
    behavioural deviation twice -- once as B and again as D -- and the first
    run of this session tied the two on exactly that.
    """
    window = slice(0, index + 1)
    mask = np.asarray(labels)[window] != intended
    if not mask.any():
        return 0.0
    return float(cusum(np.asarray(z_scores)[window][mask])[-1])


def cusum(z_scores: np.ndarray, allowance: float = CUSUM_ALLOWANCE) -> np.ndarray:
    """One-sided upper CUSUM, S_t = max(0, S_{t-1} + z_t - k)."""
    statistic = np.zeros(len(z_scores))
    running = 0.0
    for index, value in enumerate(np.asarray(z_scores, dtype=np.float64)):
        running = max(0.0, running + float(value) - allowance)
        statistic[index] = running
    return statistic


def _mean_mdf(x: np.ndarray, n_channels: int) -> np.ndarray:
    per = len(PER_CHANNEL_FEATURES)
    return np.atleast_2d(x).reshape(-1, n_channels, per)[:, :, _MDF_INDEX].mean(axis=1)


@dataclasses.dataclass
class _Training:
    x: np.ndarray
    y: np.ndarray
    scaler: StandardScaler
    centroids: np.ndarray  # standardized, per class
    distance_mean: float
    distance_std: float
    mdf: np.ndarray  # per class, training mean of channel-mean MDF


def _record(subject, excitations, sim, seed_base, fatigue=0.0, seconds=1.0):
    rows, labels, cues = [], [], []
    for cue, (label, excitation) in enumerate(excitations):
        seed = seed_base + len(labels) * 7 + label * 101
        signal = simulate_excitation(_attempt(excitation, seed), seconds, sim, subject, seed, fatigue)
        for vector in _windows(signal, sim):
            rows.append(vector)
            labels.append(label)
            cues.append(cue)
    return np.asarray(rows), np.asarray(labels), np.asarray(cues)


#: The physiological fault: these two cues come out as one muscle pattern.
MERGED_GESTURES = ("pinch", "point")


def _excitation(gesture: str, fault: str) -> np.ndarray:
    if fault == "physiological" and gesture in MERGED_GESTURES:
        return np.mean([GESTURE_SYNERGIES[g] for g in MERGED_GESTURES], axis=0)
    return np.asarray(GESTURE_SYNERGIES[gesture], dtype=np.float64)


def _train(subject, sim, seed, reps, fault) -> _Training:
    cues = [
        (label, _excitation(g, fault))
        for rep in range(reps)
        for label, g in enumerate(GESTURES)
    ]
    x, y, cue = _record(subject, cues, sim, seed_base=seed * 7919 + 600_000)
    scaler = StandardScaler().fit(x)
    z = scaler.transform(x)
    centroids = np.stack([z[y == k].mean(axis=0) for k in range(len(GESTURES))])
    # Leave one repetition out: a window's distance to a centroid it helped
    # form is smaller than a fresh window's, and fresh windows are what the
    # session compares against this baseline.
    repetition = cue // len(GESTURES)
    distances = np.empty(len(y))
    for held in np.unique(repetition):
        fit, out = repetition != held, repetition == held
        for k in range(len(GESTURES)):
            centroid = z[fit & (y == k)].mean(axis=0)
            rows = out & (y == k)
            distances[rows] = np.linalg.norm(z[rows] - centroid, axis=1)
    distances /= np.sqrt(z.shape[1])
    mdf = _mean_mdf(x, sim.n_electrodes)
    return _Training(
        x=x,
        y=y,
        scaler=scaler,
        centroids=centroids,
        distance_mean=float(distances.mean()),
        distance_std=float(max(distances.std(ddof=1), 1e-9)),
        mdf=np.array([mdf[y == k].mean() for k in range(len(GESTURES))]),
    )


def _session_cues(fault: str) -> list[tuple[int, np.ndarray]]:
    names = [m.name for m in MUSCLES]
    cues = []
    for label, gesture in enumerate(GESTURES):
        if gesture == "rest":
            continue
        excitation = _excitation(gesture, fault)
        if fault == "behavioral" and gesture == BEHAVIOURAL_GESTURE:
            columns = [names.index(n) for n in MUSCLE_GROUPS[BEHAVIOURAL_GROUP]]
            excitation = excitation.copy()
            excitation[columns] = np.clip(excitation[columns] + BEHAVIOURAL_STEP, 0.0, 1.0)
        cues.append((label, excitation))
    return cues


def run_session(
    fault: str,
    seed: int = 0,
    reps: int = 4,
    sim: SimulatorConfig = SimulatorConfig(),
    max_examples: int = 3,
) -> dict[str, object]:
    if fault not in {"behavioral", "drift", "physiological", "model_capacity"}:
        raise ValueError(f"unknown fault {fault!r}")
    subject = make_subject(f"router{seed:02d}", seed=3000 + seed)
    training = _train(subject, sim, seed, reps, fault)

    surrogate = make_pipeline(
        StandardScaler(), LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
    ).fit(training.x, training.y)
    if fault == "model_capacity":
        deployed = make_pipeline(
            StandardScaler(), PCA(n_components=2, random_state=seed), LinearDiscriminantAnalysis()
        ).fit(training.x, training.y)
        deployed_name = "PCA(2) + LDA (deliberately under-capacity)"
    else:
        deployed = surrogate
        deployed_name = "shrinkage LDA"
    reference = make_pipeline(StandardScaler(), SVC(C=10.0, gamma="scale")).fit(
        training.x, training.y
    )
    linear = LinearDecoder.from_pipeline(surrogate, GESTURES)

    x, y, _ = _record(
        subject,
        _session_cues(fault),
        sim,
        seed_base=seed * 7919 + 700_000,
        fatigue=DRIFT_FATIGUE if fault == "drift" else 0.0,
    )
    predicted = deployed.predict(x)
    referenced = reference.predict(x)

    z = training.scaler.transform(x)
    distance = np.linalg.norm(z - training.centroids[y], axis=1) / np.sqrt(z.shape[1])
    z_distance = (distance - training.distance_mean) / training.distance_std
    statistic = cusum(z_distance)
    session_mdf = _mean_mdf(x, sim.n_electrodes)
    mdf_shift = np.cumsum(session_mdf - training.mdf[y]) / np.arange(1, len(y) + 1)

    train_z = training.scaler.transform(training.x)
    routed: Counter[str] = Counter()
    examples = []
    n_errors = 0
    for index in np.flatnonzero(predicted != y):
        n_errors += 1
        intended, decoded = GESTURES[int(y[index])], GESTURES[int(predicted[index])]
        result = attribute_error(
            ErrorObservation(
                name=f"{fault}_{index}",
                intended=intended,
                decoded=decoded,
                observed=z[index],
                intended_repetitions=train_z[training.y == y[index]],
                decoded_repetitions=train_z[training.y == predicted[index]],
                reference_prediction=GESTURES[int(referenced[index])],
                drift_cusum=other_gesture_cusum(z_distance, y, int(index), int(y[index])),
                drift_threshold=CUSUM_THRESHOLD,
                mdf_shift_hz=float(mdf_shift[index]),
                mdf_threshold_hz=MDF_THRESHOLD_HZ,
            )
        )
        routed[str(result["cause"])] += 1
        if result["cause"] == "behavioral":
            anatomy = explain_error(
                linear,
                x[index],
                training.x[training.y == y[index]].mean(axis=0),
                intended,
                decoded,
            )
            result["anatomy"] = {
                "muscle_groups": anatomy["muscle_groups"],
                "channels": anatomy["channels"],
                "coaching": anatomy["coaching"],
            }
            result["coaching"] = anatomy["coaching"]["instruction"]
        if len(examples) < max_examples:
            examples.append({**result, "window": int(index)})

    expected = fault
    plurality = routed.most_common(1)[0][0] if routed else None
    coached = [
        item for item in examples if item.get("anatomy", {}).get("coaching", {}).get("confident")
    ]
    return {
        "fault": fault,
        "injected_cause": expected,
        "subject": subject.subject_id,
        "deployed_model": deployed_name,
        "reference_model": "RBF SVM (C=10, gamma=scale)",
        "injection": _describe(fault),
        "n_windows": int(len(y)),
        "n_error_windows": n_errors,
        "deployed_accuracy": round(float(np.mean(predicted == y)), 4),
        "reference_accuracy": round(float(np.mean(referenced == y)), 4),
        "routed": {cause: routed.get(cause, 0) for cause in (
            "physiological", "behavioral", "model_capacity", "drift", "inconclusive"
        )},
        "plurality_route": plurality,
        "routes_to_injected_cause": plurality == expected,
        "share_routed_to_injected_cause": round(routed.get(expected, 0) / n_errors, 4)
        if n_errors
        else None,
        "final_cusum_ratio": round(float(statistic[-1] / CUSUM_THRESHOLD), 4),
        "final_mdf_shift_hz": round(float(mdf_shift[-1]), 3),
        "coached_examples": len(coached),
        "examples": examples,
    }


def _describe(fault: str) -> str:
    if fault == "behavioral":
        return (
            f"`{BEHAVIOURAL_GESTURE}` performed with wrist-flexor excitation raised by "
            f"{BEHAVIOURAL_STEP}; all other cues clean."
        )
    if fault == "drift":
        return f"Whole session simulated at fatigue {DRIFT_FATIGUE} (MUAP widening)."
    if fault == "physiological":
        return (
            "Subject produces `pinch` and `point` as one muscle pattern (their "
            "average), in training and in the session."
        )
    return "Deployed decoder restricted to two principal components before LDA."


def router_session_demo(seed: int = 0, reps: int = 4) -> dict[str, object]:
    sessions = [
        run_session(fault, seed=seed, reps=reps)
        for fault in ("physiological", "behavioral", "model_capacity", "drift")
    ]
    return {
        "source": "simulated_sessions",
        "seed": seed,
        "reps": reps,
        "thresholds": {
            "cusum_allowance": CUSUM_ALLOWANCE,
            "cusum_threshold": CUSUM_THRESHOLD,
            "mdf_threshold_hz": MDF_THRESHOLD_HZ,
        },
        "sessions_routed": sum(bool(s["routes_to_injected_cause"]) for s in sessions),
        "n_sessions": len(sessions),
        "sessions": sessions,
        "method": (
            "Each session is generated by the motor-unit simulator and decoded. "
            "Every misclassified window is routed: P from overlap of the two "
            "classes' tightest training repetitions; B from channel deviation "
            "against the intended class's training spread; M from whether an "
            "RBF-SVM reference is right where the deployed model is wrong; D from "
            "a one-sided CUSUM of distance to the cued class centroid and the "
            "running median-frequency shift. A session routes correctly when the "
            "plurality of its errors reaches the injected cause. Behavioural "
            "routes receive anatomical coaching from an LDA surrogate."
        ),
        "caveat": (
            "Simulated subjects and injected faults. Sessions are cued, so the "
            "intended gesture is known. Within-attempt windows are correlated, "
            "which inflates CUSUM even in clean sessions. This checks the integrated pipeline on "
            "simulator signals; it is not attribution accuracy on recorded or "
            "human errors, and the coaching is not validated with people."
        ),
    }


FAULTS = ("physiological", "behavioral", "model_capacity", "drift")


def router_session_cohort(seeds: tuple[int, ...] = (0, 1, 2, 3, 4), reps: int = 4) -> dict[str, object]:
    """Every seed's sessions, summarised per fault; seed `seeds[0]` kept in
    full for the browser. The seed list is fixed in advance and every seed is
    reported, including the ones that route wrongly."""
    runs = [router_session_demo(seed=seed, reps=reps) for seed in seeds]
    per_fault = {}
    for fault in FAULTS:
        sessions = [next(s for s in run["sessions"] if s["fault"] == fault) for run in runs]
        errors = sum(s["n_error_windows"] for s in sessions)
        per_fault[fault] = {
            "sessions_routed": sum(bool(s["routes_to_injected_cause"]) for s in sessions),
            "n_sessions": len(sessions),
            "n_error_windows": errors,
            "n_errors_routed_to_injected_cause": sum(
                s["routed"][fault] for s in sessions
            ),
            "error_windows_per_session": [s["n_error_windows"] for s in sessions],
        }
    detail = runs[0]
    return {
        **{key: value for key, value in detail.items() if key not in {"seed", "sessions_routed", "n_sessions"}},
        "seeds": list(seeds),
        "sessions_routed": sum(run["sessions_routed"] for run in runs),
        "n_sessions": sum(run["n_sessions"] for run in runs),
        "per_seed_routed": {str(run["seed"]): run["sessions_routed"] for run in runs},
        "per_fault": per_fault,
        "detail_seed": detail["seed"],
        "sessions": detail["sessions"],
        "history": (
            "First run (seed 0): 2/4. Three estimator defects were then fixed on "
            "principle, without changing any routing threshold: P used a "
            "diagonal Bhattacharyya coefficient that collapsed to ~1e-50 in 136 "
            "dimensions; D compared fresh windows with in-sample training "
            "distances; D read drift off the gesture in error, double-counting "
            "behavioural deviation. The physiological injection was changed "
            "from 30% finger reach, which produced one error, to two cues "
            "produced as one pattern. The cohort below was run once after those "
            "changes, over a seed list fixed beforehand."
        ),
    }
