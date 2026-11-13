"""Anatomical attribution and prescriptive coaching (Pillar 4).

Three steps turn one misclassified window into one sentence a wearer can act
on, and each is checkable on its own:

1. **Attribution.** For a linear decoder the score margin between the decoded
   and the intended class is exactly a sum over features. Taking the intended
   class's own training mean as the baseline,

       margin(x) - margin(mu) = sum_f (w_decoded - w_intended)_f * (z_f - zmu_f)

   in standardized space. Integrated gradients from that baseline reduce to
   precisely this sum for a linear model, so no sampling or approximation is
   involved; `margin_residual` reports the arithmetic check.
2. **Back-projection.** Channel attributions are given a sign from the channel's
   activity -- above or below the intended template -- and projected onto the
   six modelled muscles through a ridge inverse of the *nominal* volume-
   conduction matrix in `anatomy.py`, then summed into the four functional
   groups. Nominal, not the subject's: a real system does not know how far a
   wearer's electrodes have rotated, and the simulator study must pay for that.
3. **Prescription.** A fixed grammar maps the leading group and direction to an
   instruction. It names muscle groups, never single fingers: an eight-channel
   ring cannot resolve one finger from another (`anatomy.MUSCLE_GROUPS`).

What this does **not** do, and why:

- It explains a linear surrogate (shrinkage LDA), not the deployed RBF SVM.
- It has no time axis. The spec's "start your press 60 ms earlier" needs an
  onset envelope, and the simulator applies none -- excitation is constant for
  a whole repetition -- so a timing instruction would be invented.
- It is validated only against simulated injections. Whether the sentences help
  a person is a human-participant question.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from neurogrip.anatomy import MUSCLE_GROUPS, MUSCLES, ForearmGeometry, attenuation_matrix
from neurogrip.cartography import _attempt as _jitter
from neurogrip.features import FeatureConfig, extract_features, feature_vector
from neurogrip.simulator import (
    GESTURE_SYNERGIES,
    SimulatorConfig,
    make_subject,
    simulate_excitation,
)
from neurogrip.spec import PER_CHANNEL_FEATURES
from neurogrip.windowing import WindowConfig, frame_windows

#: Ridge strength, as a fraction of the mean diagonal of G^T G. The muscle
#: columns of G are strongly correlated (every electrode sees every muscle), so
#: an unregularised inverse amplifies noise into alternating-sign muscles.
RIDGE_FRACTION = 0.1

#: The leading group must hold at least this share of the absolute group
#: evidence before a sentence is issued. Declared before the study was run.
MIN_LEAD_SHARE = 0.40

#: A second group earns a clause in the sentence above this share.
SECONDARY_SHARE = 0.25

_RMS_INDEX = PER_CHANNEL_FEATURES.index("rms")

GROUP_NAMES = {
    "digit_flexor": "finger flexors",
    "digit_extensor": "finger extensors",
    "wrist_flexor": "wrist flexors",
    "wrist_extensor": "wrist extensors",
}

#: (group, direction) -> imperative. "excess": the attempt had more activity
#: there than the user's own template; "deficit": less.
PHRASES = {
    ("digit_flexor", "excess"): "ease off the finger squeeze",
    ("digit_flexor", "deficit"): "curl your fingers more firmly",
    ("digit_extensor", "excess"): "relax the effort to straighten your fingers",
    ("digit_extensor", "deficit"): "straighten your fingers a little more",
    ("wrist_flexor", "excess"): "keep your wrist from bending inward",
    ("wrist_flexor", "deficit"): "bend your wrist in a little more",
    ("wrist_extensor", "excess"): "keep your wrist from pulling back",
    ("wrist_extensor", "deficit"): "bend your wrist back a little more",
}

SCOPE_NOTE = (
    "Estimated from the electrode ring for groups of forearm muscles. It does "
    "not identify a single finger or muscle."
)


@dataclass(frozen=True)
class LinearDecoder:
    """A standardizer plus linear class scores: z = (x - mean) / scale,
    score = z @ coef.T + intercept."""

    mean: np.ndarray
    scale: np.ndarray
    coef: np.ndarray  # (n_classes, n_features), standardized space
    intercept: np.ndarray  # (n_classes,)
    classes: tuple[str, ...]

    @classmethod
    def from_pipeline(cls, pipeline: Pipeline, classes: tuple[str, ...]) -> "LinearDecoder":
        scaler = pipeline.named_steps["standardscaler"]
        model = pipeline.steps[-1][1]
        coef = np.asarray(model.coef_, dtype=np.float64)
        intercept = np.asarray(model.intercept_, dtype=np.float64)
        if coef.shape[0] != len(classes):
            raise ValueError("decoder needs one score row per class")
        return cls(
            mean=np.asarray(scaler.mean_, dtype=np.float64),
            scale=np.asarray(scaler.scale_, dtype=np.float64),
            coef=coef,
            intercept=intercept,
            classes=tuple(classes),
        )

    def standardize(self, x: np.ndarray) -> np.ndarray:
        return (np.asarray(x, dtype=np.float64) - self.mean) / self.scale

    def scores(self, x: np.ndarray) -> np.ndarray:
        return self.standardize(x) @ self.coef.T + self.intercept

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.argmax(np.atleast_2d(self.scores(x)), axis=1)


def margin_contributions(
    decoder: LinearDecoder,
    observed: np.ndarray,
    baseline: np.ndarray,
    intended: int,
    decoded: int,
) -> np.ndarray:
    """Per-feature share of the decoded-minus-intended margin change from the
    baseline. Positive values pushed the window toward the error."""
    weights = decoder.coef[decoded] - decoder.coef[intended]
    return weights * (decoder.standardize(observed) - decoder.standardize(baseline))


def channel_attribution(contributions: np.ndarray, n_channels: int) -> np.ndarray:
    values = np.asarray(contributions, dtype=np.float64)
    if values.size != n_channels * len(PER_CHANNEL_FEATURES):
        raise ValueError("contributions must hold complete channel-major groups")
    return values.reshape(n_channels, len(PER_CHANNEL_FEATURES)).sum(axis=1)


def activity_deviation(observed: np.ndarray, baseline: np.ndarray, n_channels: int) -> np.ndarray:
    """Log ratio of each channel's RMS to the intended template's RMS."""
    per = len(PER_CHANNEL_FEATURES)
    rms = np.asarray(observed, dtype=np.float64).reshape(n_channels, per)[:, _RMS_INDEX]
    template = np.asarray(baseline, dtype=np.float64).reshape(n_channels, per)[:, _RMS_INDEX]
    return np.log(np.maximum(rms, 1e-12) / np.maximum(template, 1e-12))


def back_project(
    channel_evidence: np.ndarray,
    n_channels: int,
    geometry: ForearmGeometry | None = None,
) -> dict[str, float]:
    """Signed channel evidence -> signed evidence per functional muscle group."""
    gains = attenuation_matrix(n_channels, 0.0, geometry or ForearmGeometry())
    gram = gains.T @ gains
    ridge = RIDGE_FRACTION * float(np.mean(np.diag(gram)))
    muscles = np.linalg.solve(gram + ridge * np.eye(gram.shape[0]), gains.T @ channel_evidence)
    names = [muscle.name for muscle in MUSCLES]
    return {
        group: float(sum(muscles[names.index(name)] for name in members))
        for group, members in MUSCLE_GROUPS.items()
    }


def _label(gesture: str) -> str:
    return gesture.replace("_", " ")


def coach(intended: str, decoded: str, group_scores: dict[str, float]) -> dict[str, object]:
    """Turn group evidence into one sentence, or decline."""
    magnitude = {group: abs(value) for group, value in group_scores.items()}
    total = sum(magnitude.values())
    ranked = sorted(magnitude, key=lambda group: magnitude[group], reverse=True)
    opening = f"Your {_label(intended)} attempt was read as {_label(decoded)}."
    if total <= 1e-12 or magnitude[ranked[0]] / total < MIN_LEAD_SHARE:
        return {
            "confident": False,
            "group": None,
            "direction": None,
            "share": round(magnitude[ranked[0]] / total, 4) if total > 1e-12 else 0.0,
            "instruction": (
                f"{opening} No single muscle group explains it; repeat the "
                "gesture slowly and compare it with your own stable attempts."
            ),
            "scope": SCOPE_NOTE,
        }

    def direction(group: str) -> str:
        return "excess" if group_scores[group] > 0 else "deficit"

    lead = ranked[0]
    sentence = f"{opening} Try to {PHRASES[(lead, direction(lead))]}"
    second = ranked[1]
    if magnitude[second] / total >= SECONDARY_SHARE:
        sentence += f", and {PHRASES[(second, direction(second))]}"
    return {
        "confident": True,
        "group": lead,
        "group_name": GROUP_NAMES[lead],
        "direction": direction(lead),
        "share": round(magnitude[lead] / total, 4),
        "instruction": sentence + ".",
        "scope": SCOPE_NOTE,
    }


def explain_error(
    decoder: LinearDecoder,
    observed: np.ndarray,
    baseline: np.ndarray,
    intended: str,
    decoded: str,
) -> dict[str, object]:
    """Attribution, back-projection and coaching for one misclassified window."""
    n_channels = np.asarray(observed).size // len(PER_CHANNEL_FEATURES)
    i, d = decoder.classes.index(intended), decoder.classes.index(decoded)
    contributions = margin_contributions(decoder, observed, baseline, i, d)
    scores, base = decoder.scores(observed), decoder.scores(baseline)
    residual = abs(
        float(contributions.sum()) - float((scores[d] - scores[i]) - (base[d] - base[i]))
    )

    attribution = channel_attribution(contributions, n_channels)
    activity = activity_deviation(observed, baseline, n_channels)
    # Magnitude from what pushed the margin, sign from which way activity moved.
    evidence = np.maximum(attribution, 0.0) * np.sign(activity)
    groups = back_project(evidence, n_channels)
    pushed = np.maximum(attribution, 0.0)
    share = pushed / pushed.sum() if pushed.sum() > 0 else pushed

    return {
        "intended": intended,
        "decoded": decoded,
        "margin_change": round(float(contributions.sum()), 6),
        "margin_residual": residual,
        "channels": [
            {
                "channel": index + 1,
                "attribution_share": round(float(share[index]), 4),
                "activity_log_ratio": round(float(activity[index]), 4),
            }
            for index in range(n_channels)
        ],
        "muscle_groups": {group: round(value, 4) for group, value in groups.items()},
        "coaching": coach(intended, decoded, groups),
    }


# --- Simulator validation ----------------------------------------------------

#: Excess: the group's muscles gain this much excitation. Deficit: they keep
#: this fraction. A deficit is only injected where there is activity to lose.
EXCESS_STEP = 0.35
DEFICIT_KEEP = 0.3
MIN_DEFICIT_EXCITATION = 0.15


def _windows(signal: np.ndarray, sim: SimulatorConfig) -> list[np.ndarray]:
    window = WindowConfig(window_ms=200, hop_ms=100, sampling_rate_hz=sim.sampling_rate_hz)
    config = FeatureConfig(sampling_rate_hz=sim.sampling_rate_hz)
    return [
        feature_vector(extract_features(frame, config), sim.n_electrodes)
        for frame in frame_windows(signal, window)
    ]


def _attempt(excitation: np.ndarray, seed: int) -> np.ndarray:
    # Same execution-variability model as Cartography's repeat probe.
    return _jitter(excitation, np.random.default_rng(seed))


def train_linear_decoder(
    subject_seed: int,
    gestures: tuple[str, ...],
    reps: int,
    sim: SimulatorConfig,
    rep_seconds: float = 1.0,
):
    """A shrinkage-LDA decoder on clean simulated repetitions, plus each
    class's raw training windows."""
    subject = make_subject(f"coach{subject_seed:02d}", seed=2000 + subject_seed)
    rows, labels = [], []
    for label, gesture in enumerate(gestures):
        excitation = np.asarray(GESTURE_SYNERGIES[gesture], dtype=np.float64)
        for rep in range(reps):
            seed = subject_seed * 7919 + 300_000 + label * 101 + rep
            signal = simulate_excitation(_attempt(excitation, seed), rep_seconds, sim, subject, seed)
            for vector in _windows(signal, sim):
                rows.append(vector)
                labels.append(label)
    x, y = np.asarray(rows), np.asarray(labels)
    pipeline = make_pipeline(
        StandardScaler(), LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
    ).fit(x, y)
    return subject, LinearDecoder.from_pipeline(pipeline, gestures), x, y


def localisation_study(
    seed: int = 0,
    gestures: tuple[str, ...] | None = None,
    reps: int = 4,
    sim: SimulatorConfig = SimulatorConfig(),
) -> dict[str, object]:
    """Inject a known muscle-group deviation into each gesture and measure how
    often the coaching names the injected group and direction.

    Every misclassified window counts in the denominator. Declined windows
    count as not recovered: a system that only answers when it is easy would
    otherwise report its easiest cases as its accuracy.
    """
    names = tuple(gestures) if gestures is not None else tuple(
        g for g in GESTURE_SYNERGIES if g != "rest"
    )
    vocabulary = ("rest", *names)
    subject, decoder, x, y = train_linear_decoder(seed, vocabulary, reps, sim)
    templates = {
        gesture: x[y == index].mean(axis=0) for index, gesture in enumerate(vocabulary)
    }
    muscle_names = [muscle.name for muscle in MUSCLES]

    cases = []
    per_group = {group: {"error_windows": 0, "recovered": 0} for group in MUSCLE_GROUPS}
    totals = {"recovered": 0, "wrong": 0, "declined": 0, "windows": 0, "error_windows": 0}
    for g_index, gesture in enumerate(names):
        base = np.asarray(GESTURE_SYNERGIES[gesture], dtype=np.float64)
        for k_index, (group, members) in enumerate(sorted(MUSCLE_GROUPS.items())):
            columns = [muscle_names.index(name) for name in members]
            for direction in ("excess", "deficit"):
                excitation = base.copy()
                if direction == "excess":
                    excitation[columns] = np.clip(excitation[columns] + EXCESS_STEP, 0.0, 1.0)
                else:
                    if float(base[columns].mean()) < MIN_DEFICIT_EXCITATION:
                        continue
                    excitation[columns] *= DEFICIT_KEEP
                attempt_seed = seed * 7919 + 500_000 + g_index * 211 + k_index * 13 + (
                    direction == "deficit"
                )
                signal = simulate_excitation(
                    _attempt(excitation, attempt_seed), 1.0, sim, subject, attempt_seed
                )
                windows = _windows(signal, sim)
                predicted = decoder.predict(np.asarray(windows))
                recovered = wrong = declined = 0
                example = None
                for vector, label in zip(windows, predicted):
                    decoded = vocabulary[int(label)]
                    if decoded == gesture:
                        continue
                    result = explain_error(decoder, vector, templates[gesture], gesture, decoded)
                    advice = result["coaching"]
                    if not advice["confident"]:
                        declined += 1
                    elif advice["group"] == group and advice["direction"] == direction:
                        recovered += 1
                    else:
                        wrong += 1
                    example = example or result
                n_errors = recovered + wrong + declined
                totals["windows"] += len(windows)
                totals["error_windows"] += n_errors
                totals["recovered"] += recovered
                totals["wrong"] += wrong
                totals["declined"] += declined
                per_group[group]["error_windows"] += n_errors
                per_group[group]["recovered"] += recovered
                cases.append(
                    {
                        "gesture": gesture,
                        "injected_group": group,
                        "direction": direction,
                        "windows": len(windows),
                        "error_windows": n_errors,
                        "recovered": recovered,
                        "wrong_group_or_direction": wrong,
                        "declined": declined,
                        "example": example,
                    }
                )

    n_errors = totals["error_windows"]
    return {
        "source": "simulator",
        "seed": seed,
        "subject": subject.subject_id,
        "electrode_shift_rad": round(subject.electrode_shift_rad, 4),
        "decoder": "shrinkage LDA surrogate on clean simulated repetitions",
        "back_projection_geometry": "nominal (no electrode shift, default conduction length)",
        "injection": {
            "excess_step": EXCESS_STEP,
            "deficit_keep": DEFICIT_KEEP,
            "min_deficit_excitation": MIN_DEFICIT_EXCITATION,
        },
        "min_lead_share": MIN_LEAD_SHARE,
        "n_injections": len(cases),
        "n_windows": totals["windows"],
        "n_error_windows": n_errors,
        "n_recovered": totals["recovered"],
        "n_wrong_group_or_direction": totals["wrong"],
        "n_declined": totals["declined"],
        "recovery_rate": round(totals["recovered"] / n_errors, 4) if n_errors else None,
        "chance_rate": 1 / 8,
        "per_group": {
            group: {
                **counts,
                "recovery_rate": (
                    round(counts["recovered"] / counts["error_windows"], 4)
                    if counts["error_windows"]
                    else None
                ),
            }
            for group, counts in per_group.items()
        },
        "cases": cases,
        "caveat": (
            "Simulated subject, simulated injections, linear surrogate decoder. "
            "Recovery means the sentence named the injected muscle group and "
            "direction; it is not evidence that the sentence helps a person. "
            "No timing advice: the simulator has no onset envelope."
        ),
    }


def localisation_cohort(n_subjects: int = 5, reps: int = 4) -> dict[str, object]:
    """`localisation_study` over a fixed list of simulated subjects, pooled.

    Subject 0's injections are kept in full for the browser; the others are
    summarised. Each subject has its own random electrode rotation, which the
    nominal back-projection geometry does not know about.
    """
    studies = [localisation_study(seed=seed, reps=reps) for seed in range(n_subjects)]
    pooled = {key: sum(study[key] for study in studies) for key in (
        "n_windows", "n_error_windows", "n_recovered", "n_wrong_group_or_direction", "n_declined"
    )}
    per_group = {}
    for group in MUSCLE_GROUPS:
        windows = sum(study["per_group"][group]["error_windows"] for study in studies)
        recovered = sum(study["per_group"][group]["recovered"] for study in studies)
        per_group[group] = {
            "error_windows": windows,
            "recovered": recovered,
            "recovery_rate": round(recovered / windows, 4) if windows else None,
        }
    detail = studies[0]
    return {
        "source": "simulator",
        "n_subjects": n_subjects,
        "decoder": detail["decoder"],
        "back_projection_geometry": detail["back_projection_geometry"],
        "injection": detail["injection"],
        "min_lead_share": MIN_LEAD_SHARE,
        **pooled,
        "recovery_rate": round(pooled["n_recovered"] / pooled["n_error_windows"], 4)
        if pooled["n_error_windows"]
        else None,
        "chance_rate": 1 / 8,
        "per_group": per_group,
        "per_subject": [
            {
                "subject": study["subject"],
                "electrode_shift_rad": study["electrode_shift_rad"],
                "n_error_windows": study["n_error_windows"],
                "recovery_rate": study["recovery_rate"],
            }
            for study in studies
        ],
        "detail_subject": detail["subject"],
        "cases": detail["cases"],
        "caveat": detail["caveat"] + (
            " The back-projection inverts the same family of volume-conduction "
            "model the simulator generates from; only the electrode rotation and "
            "conduction length differ. Recorded anatomy will not be this kind, "
            "so this is an upper bound on localisation, not an estimate of it."
        ),
    }
