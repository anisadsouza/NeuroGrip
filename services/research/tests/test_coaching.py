"""Pillar 4: anatomical back-projection and prescriptive coaching.

The coaching sentence is only as trustworthy as two links behind it: the
attribution must be an exact account of the decoder's margin, and the
back-projection must put activity from a muscle group back on that group.
Both links are tested in isolation here, and the simulator study tests them
together against injected deviations whose true muscle group is known.
"""

import numpy as np
import pytest
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from neurogrip.anatomy import MUSCLE_GROUPS, MUSCLES, ForearmGeometry, attenuation_matrix
from neurogrip.coaching import (
    LinearDecoder,
    back_project,
    channel_attribution,
    coach,
    explain_error,
    localisation_study,
    margin_contributions,
)
from neurogrip.spec import PER_CHANNEL_FEATURES

N_CHANNELS = 8
N_FEATURES = N_CHANNELS * len(PER_CHANNEL_FEATURES)


def _fitted_decoder(seed: int = 3) -> tuple[LinearDecoder, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    centres = rng.normal(0.0, 2.0, size=(3, N_FEATURES))
    x = np.vstack([centre + rng.normal(size=(40, N_FEATURES)) for centre in centres])
    y = np.repeat(np.arange(3), 40)
    pipeline = make_pipeline(
        StandardScaler(), LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
    ).fit(x, y)
    return LinearDecoder.from_pipeline(pipeline, ("a", "b", "c")), x, y


def test_contributions_account_exactly_for_the_margin_change():
    decoder, x, y = _fitted_decoder()
    baseline = x[y == 0].mean(axis=0)
    observed = x[y == 2][0]

    contributions = margin_contributions(decoder, observed, baseline, intended=0, decoded=2)
    scores, base_scores = decoder.scores(observed), decoder.scores(baseline)
    expected = (scores[2] - scores[0]) - (base_scores[2] - base_scores[0])

    assert contributions.shape == (N_FEATURES,)
    assert contributions.sum() == pytest.approx(expected, rel=1e-10, abs=1e-10)


def test_channel_attribution_sums_each_channel_major_group():
    contributions = np.zeros(N_FEATURES)
    per = len(PER_CHANNEL_FEATURES)
    contributions[3 * per : 4 * per] = 0.5

    channels = channel_attribution(contributions, N_CHANNELS)

    assert channels.shape == (N_CHANNELS,)
    assert channels[3] == pytest.approx(0.5 * per)
    assert np.count_nonzero(channels) == 1


@pytest.mark.parametrize("group", sorted(MUSCLE_GROUPS))
@pytest.mark.parametrize("sign", [1.0, -1.0])
def test_back_projection_returns_a_group_s_own_footprint_to_that_group(group, sign):
    gains = attenuation_matrix(N_CHANNELS, 0.0, ForearmGeometry())
    names = [muscle.name for muscle in MUSCLES]
    columns = [names.index(name) for name in MUSCLE_GROUPS[group]]
    footprint = sign * gains[:, columns].sum(axis=1)

    scores = back_project(footprint, N_CHANNELS)

    lead = max(scores, key=lambda name: abs(scores[name]))
    assert lead == group
    assert np.sign(scores[lead]) == sign


def test_coaching_names_a_muscle_group_and_never_a_single_finger():
    result = coach(
        "open_hand",
        "fist",
        {"digit_flexor": 0.9, "digit_extensor": 0.1, "wrist_flexor": 0.05, "wrist_extensor": 0.0},
    )

    assert result["confident"] is True
    assert result["group"] == "digit_flexor"
    assert result["direction"] == "excess"
    text = result["instruction"].lower()
    assert "open hand" in text and "fist" in text
    for finger in ("index", "middle", "ring", "little", "thumb"):
        assert finger not in text


def test_coaching_declines_when_no_group_dominates():
    result = coach(
        "point",
        "pinch",
        {"digit_flexor": 0.3, "digit_extensor": -0.3, "wrist_flexor": 0.28, "wrist_extensor": -0.27},
    )

    assert result["confident"] is False
    assert result["group"] is None
    assert "no single muscle group" in result["instruction"].lower()


def test_explain_error_reports_channels_groups_and_an_exact_residual():
    decoder, x, y = _fitted_decoder()
    baseline = x[y == 0].mean(axis=0)
    result = explain_error(decoder, x[y == 1][0], baseline, intended="a", decoded="b")

    assert len(result["channels"]) == N_CHANNELS
    assert set(result["muscle_groups"]) == set(MUSCLE_GROUPS)
    assert result["margin_residual"] < 1e-9
    assert isinstance(result["coaching"]["instruction"], str)


def test_simulated_localisation_beats_chance_and_reports_its_denominator():
    report = localisation_study(seed=0, gestures=("fist", "point", "wrist_flexion"), reps=3)

    assert report["source"] == "simulator"
    assert report["chance_rate"] == pytest.approx(1 / 8)
    assert report["n_error_windows"] > 0
    assert (
        report["n_recovered"] + report["n_wrong_group_or_direction"] + report["n_declined"]
        == report["n_error_windows"]
    )
    # Pre-declared: at least twice chance on the error windows it explains.
    assert report["recovery_rate"] > 2 * report["chance_rate"]
