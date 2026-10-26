import numpy as np
import pytest

from neurogrip.cartography import (
    NATURAL_SYNERGIES,
    BabbleConfig,
    SynthesisWeights,
    _pattern_distance,
    babble_excitations,
    compare_vocabularies,
    default_min_distance,
    fit_cartograph,
    record_babble,
    synthesize,
)
from neurogrip.simulator import SimulatorConfig, make_subject


@pytest.fixture(scope="module")
def subject():
    return make_subject("carto", seed=11)


@pytest.fixture(scope="module")
def babble(subject):
    return record_babble(BabbleConfig(n_poses=120, seed=3), SimulatorConfig(), subject)


@pytest.fixture(scope="module")
def cartograph(babble):
    return fit_cartograph(babble, n_synergies=4, seed=3)


def test_babble_is_a_smooth_walk_not_independent_draws():
    excitations = babble_excitations(BabbleConfig(n_poses=200, seed=1))
    steps = np.linalg.norm(np.diff(excitations, axis=0), axis=1)
    jumps = np.linalg.norm(excitations - excitations[::-1], axis=1)

    assert excitations.shape == (200, NATURAL_SYNERGIES.shape[1])
    assert excitations.min() >= 0.0 and excitations.max() <= 1.0
    # Consecutive poses are close; poses far apart in time are not.
    assert steps.mean() < 0.5 * jumps.mean()


def test_babble_is_seeded_and_leaves_global_rng_alone():
    np.random.seed(123)
    expected = np.random.random()
    np.random.seed(123)
    a = babble_excitations(BabbleConfig(n_poses=50, seed=9))
    actual = np.random.random()
    b = babble_excitations(BabbleConfig(n_poses=50, seed=9))

    assert actual == expected
    np.testing.assert_array_equal(a, b)


def test_seed_zero_is_valid(subject):
    # Seed arithmetic once went negative at seed 0, which numpy rejects.
    babble = record_babble(BabbleConfig(n_poses=8, seed=0), SimulatorConfig(), subject)
    assert babble.rms.shape[0] == 8


def test_recording_has_one_nonnegative_rms_row_per_pose(babble):
    assert babble.rms.shape == (120, SimulatorConfig().n_electrodes)
    assert np.all(babble.rms >= 0.0)
    assert babble.excitations.shape == (120, 6)


def test_synergies_explain_most_of_the_babble(cartograph):
    # Four synergies were planted; four NMF components should recover the
    # structure. Below 0.9 the embedding is drawing noise.
    assert cartograph.explained_variance >= 0.9


def test_embedding_projects_babble_into_the_unit_square(cartograph, babble):
    coords = cartograph.project(babble.rms)
    np.testing.assert_allclose(coords, cartograph.coords, atol=1e-6)
    assert coords.min() >= 0.0 and coords.max() <= 1.0


def test_density_is_high_where_babble_went_and_low_where_it_did_not(cartograph):
    occupied = cartograph.density_at(cartograph.activations).mean()
    beyond = cartograph.density_at(cartograph.activations.max(axis=0) * 3.0)[0]
    assert occupied > 10 * beyond


def test_effort_rises_with_activation(cartograph, babble):
    effort = cartograph.effort_at(cartograph.activations)
    strength = babble.rms.mean(axis=1)
    assert np.corrcoef(effort, strength)[0, 1] > 0.5


def test_level_counts_for_less_than_pattern():
    # A uniform shift in log-RMS is a gain change; a person cannot hold level
    # precisely enough for it to separate two gestures on its own.
    level = _pattern_distance(np.full(8, 1.0), np.zeros(8))
    pattern = _pattern_distance(np.array([1.0, -1.0] * 4), np.zeros(8))
    assert level < 0.5 * pattern


def test_synthesis_respects_spacing_and_reachability(cartograph):
    weights = SynthesisWeights()
    spacing = default_min_distance(cartograph)
    prototypes = synthesize(cartograph, k=5, weights=weights)
    signatures = cartograph.signature_at(prototypes)
    rest = cartograph.signature_at(cartograph.rest)[0]

    assert prototypes.shape == (5, cartograph.activations.shape[1])
    gaps = _pattern_distance(signatures[:, None], signatures[None, :])
    assert gaps[np.triu_indices(5, 1)].min() >= spacing - 1e-9
    assert _pattern_distance(signatures, rest).min() >= spacing - 1e-9
    assert np.all(cartograph.density_at(prototypes) >= weights.reach_floor)


def test_prototypes_are_poses_the_user_actually_made(cartograph):
    prototypes = synthesize(cartograph, k=4)
    for point in prototypes:
        assert np.any(np.all(cartograph.activations == point, axis=1))


def test_synthesis_is_deterministic(cartograph):
    a = synthesize(cartograph, k=4)
    b = synthesize(cartograph, k=4)
    np.testing.assert_array_equal(a, b)


def test_prototypes_decode_to_excitations_a_hand_can_produce(cartograph):
    prototypes = synthesize(cartograph, k=4)
    targets = cartograph.excitation_at(prototypes)

    assert targets.shape == (4, 6)
    assert targets.min() >= 0.0 and targets.max() <= 1.0


def test_impossible_spacing_raises_rather_than_returning_fewer(cartograph):
    with pytest.raises(ValueError):
        synthesize(cartograph, k=30, min_distance=10 * default_min_distance(cartograph))


def test_repeat_probe_measures_scatter_per_target(subject):
    from neurogrip.cartography import ProbeConfig, probe_repeatability

    targets = np.array([[0.5] * 6, [0.1, 0.1, 0.1, 0.6, 0.9, 0.6]])
    repeats = probe_repeatability(
        targets, subject, SimulatorConfig(), ProbeConfig(repeats=2, seconds=0.4), seed=1
    )
    assert repeats.means.shape == (2, SimulatorConfig().n_electrodes)
    assert np.all(repeats.spreads > 0.0)


def test_synthesis_from_repeats_chooses_only_probed_poses(cartograph, subject):
    from neurogrip.cartography import (
        REST_EXCITATION,
        ProbeConfig,
        probe_repeatability,
        shortlist_poses,
        synthesize_from_repeats,
    )

    config = ProbeConfig(shortlist=12, repeats=2, seconds=0.4)
    shortlist = shortlist_poses(cartograph, config.shortlist)
    sim = SimulatorConfig()
    rest = probe_repeatability(REST_EXCITATION, subject, sim, config, seed=1)
    repeats = probe_repeatability(
        cartograph.excitations[shortlist], subject, sim, config, seed=2
    )
    chosen = synthesize_from_repeats(cartograph, shortlist, repeats, rest, k=3)

    assert len(set(chosen)) == 3
    assert set(chosen) <= set(shortlist)


def test_fatigue_probe_shows_a_larger_mdf_drop_for_a_harder_hold(subject):
    from neurogrip.cartography import fatigue_probe

    drops = fatigue_probe(
        np.array([[0.15] * 6, [0.8] * 6]), subject, SimulatorConfig(), seed=4
    )
    assert drops.shape == (2,)
    assert drops[1] > drops[0] > -0.05


def test_effort_weighting_never_selects_a_harder_vocabulary():
    from neurogrip.cartography import SelectionWeights, select_weighted

    rng = np.random.default_rng(0)
    points = rng.normal(size=(10, 4))  # row 0 is the rest anchor
    distances = np.linalg.norm(points[:, None] - points[None, :], axis=2)
    effort = np.linspace(0.1, 1.0, 9)
    names = [f"g{i}" for i in range(9)]

    plain = select_weighted(
        distances, effort, effort, names, 4, SelectionWeights(effort=0.0, fatigue=0.0)
    )
    weighted = select_weighted(
        distances, effort, effort, names, 4, SelectionWeights(effort=2.0, fatigue=2.0)
    )

    def cost(chosen):
        return np.mean([effort[names.index(name)] for name in chosen])

    assert len(set(weighted)) == 4
    assert cost(weighted) <= cost(plain)


def test_comparison_reports_both_vocabularies(cartograph, subject):
    result = compare_vocabularies(
        cartograph, subject, SimulatorConfig(), k=4, reps=3, seed=5
    )

    assert set(result) >= {"synthesized", "catalogue", "weighted_catalogue"}
    # Every catalogue gesture's measured cost is recorded, not only the chosen.
    assert len(result["selection_costs"]) == 9
    for cost in result["selection_costs"].values():
        assert set(cost) == {"effort", "fatigue"}
    for side in ("synthesized", "catalogue", "weighted_catalogue"):
        assert len(result[side]["classes"]) == 5  # k prototypes + rest
        assert result[side]["min_pairwise_separation"] > 0.0
        assert 0.0 <= result[side]["cv_accuracy"] <= 1.0
