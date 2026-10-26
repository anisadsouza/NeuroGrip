"""Neuromotor Cartography: map a user's reachable activation manifold and
synthesise gesture prototypes inside it (Pillar 1).

Prior art selects gestures from a fixed catalogue. This invents coordinates in
the user's own manifold that no catalogue contains, then decodes each one back
to a muscle pattern the user can be taught to reach.

The pipeline, in the order the spec lays it out:

1. Babble -- unstructured exploration, plus a cued rest anchor.
2. Synergy embedding -- NMF on per-channel RMS, plus a 2-D projection of the
   synergy activations for display.
3. Fields over synergy space -- reachability rho(x), effort E(x), and
   separability S(x) against prototypes already placed.
4. Synthesis -- greedy farthest-point placement of k prototypes maximising
   w_s*S + w_r*rho - w_e*E subject to pairwise spacing, then coordinate ascent
   over the babble poses. Coordinate ascent over a finite candidate set is the
   discrete form of the spec's projected gradient refinement; the feasible set
   (reachable, spaced) is non-convex, so a projection would be a search anyway.

Repeatability comes from a repeat probe (`ProbeConfig`): the wearer re-attempts
a shortlist of babble poses, and spacing is measured in units of their own
repeat scatter. The catalogue baseline is probed identically.

**The Phase 2 gate does not pass in this simulator.** `cartography_gate` writes
the evidence. The planning metric ranks pairs the way the decoder does; the
limit is the candidate pool. Even pure max-min over the probed shortlist,
ignoring effort and reach, plans a worst pair at 2.5-4.1 repeat-scatter units
against 4.7-6.2 for catalogue selection. Babble here is a dense walk in which
every synergy is partly active, so its poses are co-contractions; the
catalogue's hand-designed patterns are sparse. Whether real babble is sparser
is an empirical question about human exploration, not one to settle by
editing the walk until the gate passes.

So the cartograph currently serves the spec's fallback (`select_weighted`):
catalogue selection weighted by effort and fatigue. It is a trade, not a free
win -- the gate artifact records what it gives up in worst-pair separability
and what it saves in effort and fatigue, per subject.

Not yet here: the effort field's time-to-reach and median-frequency-drop terms,
which need sustained holds rather than babble poses. Effort is amplitude only.
Teach-back and risk-weighted assignment (spec steps 5 and 6) are not built.

Everything is seeded. Nothing here may touch global RNG state.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field

import numpy as np
from sklearn.decomposition import NMF
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.preprocessing import StandardScaler

from neurogrip.features import FeatureConfig, extract_features, feature_vector
from neurogrip.simulator import (
    GESTURE_SYNERGIES,
    SimulatorConfig,
    SubjectProfile,
    make_subject,
    simulate_excitation,
)
from neurogrip.windowing import WindowConfig, frame_windows

# Planted synergies over (fcr, fds, fcu, ecu, edc, ecr). Babble is a walk in
# the space these span: people explore by co-activating muscle groups, not by
# driving six muscles independently. The two deviation synergies are there
# because without them babble could not reach `thumb_up` (fcr + ecr), a pattern
# the same simulated user performs on cue -- the map would have excluded a
# gesture the user can make.
NATURAL_SYNERGIES = np.array(
    [
        #  fcr   fds   fcu   ecu   edc   ecr
        [0.30, 1.00, 0.30, 0.10, 0.00, 0.10],  # finger flexion
        [0.00, 0.00, 0.10, 0.30, 1.00, 0.30],  # finger extension
        [1.00, 0.20, 1.00, 0.00, 0.00, 0.10],  # wrist flexion
        [0.10, 0.00, 0.00, 1.00, 0.30, 1.00],  # wrist extension
        [1.00, 0.10, 0.00, 0.00, 0.20, 1.00],  # radial deviation
        [0.00, 0.10, 1.00, 1.00, 0.00, 0.00],  # ulnar deviation
    ]
)

REST_EXCITATION = np.asarray(GESTURE_SYNERGIES["rest"], dtype=np.float64)


@dataclass(frozen=True)
class BabbleConfig:
    n_poses: int = 180
    pose_seconds: float = 0.25
    rest_seconds: float = 1.0
    # Ornstein-Uhlenbeck walk over synergy activations. The instruction to the
    # wearer is "explore your whole range": a timid walk (mean 0.25, step 0.08)
    # never reached a full fist or open hand, and the map had no room for them.
    reversion: float = 0.05
    step_std: float = 0.15
    mean_activation: float = 0.35
    seed: int = 0


@dataclass(frozen=True)
class BabbleRecording:
    excitations: np.ndarray  # (n_poses, n_muscles), ground truth, simulation only
    rms: np.ndarray  # (n_poses, n_channels), what an electrode ring measures
    rest_rms: np.ndarray  # (n_channels,), the cued rest anchor


@dataclass(frozen=True)
class SynthesisWeights:
    separability: float = 1.0
    reachability: float = 0.5
    effort: float = 0.5
    # Density below this is treated as unreachable: the user never went there.
    reach_floor: float = 0.15


def babble_excitations(config: BabbleConfig) -> np.ndarray:
    """A smooth random walk in synergy space, mapped to muscle excitations."""
    rng = np.random.default_rng(config.seed)
    n_synergies = NATURAL_SYNERGIES.shape[0]
    activation = np.full(n_synergies, config.mean_activation)
    walk = np.empty((config.n_poses, n_synergies))

    for pose in range(config.n_poses):
        activation = activation + config.reversion * (
            config.mean_activation - activation
        )
        activation = activation + rng.normal(0.0, config.step_std, n_synergies)
        activation = np.clip(activation, 0.0, 1.0)
        walk[pose] = activation

    return np.clip(REST_EXCITATION + walk @ NATURAL_SYNERGIES, 0.0, 1.0)


def _rms(channels: np.ndarray) -> np.ndarray:
    signal = np.asarray(channels, dtype=np.float64)
    return np.sqrt(np.mean(np.square(signal), axis=1))


def record_babble(
    config: BabbleConfig, sim: SimulatorConfig, subject: SubjectProfile
) -> BabbleRecording:
    excitations = babble_excitations(config)
    rms = np.stack(
        [
            _rms(
                simulate_excitation(
                    excitation,
                    config.pose_seconds,
                    sim,
                    subject,
                    seed=config.seed * 100_003 + pose,
                )
            )
            for pose, excitation in enumerate(excitations)
        ]
    )
    # The rest anchor takes the seed after the last pose; seeds stay non-negative.
    rest = simulate_excitation(
        REST_EXCITATION,
        config.rest_seconds,
        sim,
        subject,
        seed=config.seed * 100_003 + config.n_poses,
    )
    return BabbleRecording(excitations=excitations, rms=rms, rest_rms=_rms(rest))


@dataclass
class Cartograph:
    """A fitted map: synergy model, embeddings, and the fields over them.

    Two spaces, deliberately. Prototypes are placed in **synergy space** -- the
    NMF activations -- because that is where separation survives into the
    decoder's features. The 2-D **map** is a PCA of synergy space for display
    only: it keeps roughly three quarters of the variance, and prototypes
    spaced on it decoded to co-contraction blends no decoder could separate.
    """

    nmf: NMF = field(repr=False)
    scale: float
    activations: np.ndarray  # (n_poses, n_synergies), babble in synergy space
    bandwidth: float
    effort: np.ndarray  # (n_poses,)
    excitations: np.ndarray  # (n_poses, n_muscles)
    rms: np.ndarray  # (n_poses, n_channels)
    rest: np.ndarray  # (n_synergies,), the cued rest anchor in synergy space
    explained_variance: float
    rest_rms: np.ndarray = field(default_factory=lambda: np.zeros(0))
    map_mean: np.ndarray = field(default_factory=lambda: np.zeros(0))
    map_axes: np.ndarray = field(default_factory=lambda: np.zeros(0))
    map_offset: np.ndarray = field(default_factory=lambda: np.zeros(2))
    map_span: float = 1.0
    density_norm: float = 1.0

    def embed(self, rms: np.ndarray) -> np.ndarray:
        """RMS vectors to synergy space. Live teach-back runs this."""
        return self.nmf.transform(np.atleast_2d(rms) / self.scale)

    def to_map(self, points: np.ndarray) -> np.ndarray:
        """Synergy-space points to the 2-D display map, clipped to the unit square."""
        raw = (np.atleast_2d(points) - self.map_mean) @ self.map_axes
        return np.clip((raw - self.map_offset) / self.map_span, 0.0, 1.0)

    def project(self, rms: np.ndarray) -> np.ndarray:
        return self.to_map(self.embed(rms))

    @property
    def coords(self) -> np.ndarray:
        """The babble on the display map."""
        return self.to_map(self.activations)

    def _kernel(self, points: np.ndarray) -> np.ndarray:
        points = np.atleast_2d(points)
        sq = np.sum(
            np.square(points[:, None, :] - self.activations[None, :, :]), axis=2
        )
        return np.exp(-0.5 * sq / self.bandwidth**2)

    def density_at(self, points: np.ndarray) -> np.ndarray:
        """Reachability rho(x): kernel density of babble, 1 at the densest pose."""
        return self._kernel(points).sum(axis=1) / self.density_norm

    def _smooth(self, points: np.ndarray, values: np.ndarray, unknown: float):
        # Nadaraya-Watson, pulled toward `unknown` where there is no babble:
        # a region nobody visited is not known to be easy.
        weights = self._kernel(points)
        prior = 1e-3
        total = weights.sum(axis=1, keepdims=True) + prior
        values = values.reshape(len(self.activations), -1)
        return (weights @ values + prior * unknown) / total

    def effort_at(self, points: np.ndarray) -> np.ndarray:
        return self._smooth(points, self.effort, unknown=1.0)[:, 0]

    def excitation_at(self, points: np.ndarray) -> np.ndarray:
        """Decode to muscle excitations (simulation ground truth only)."""
        return self._smooth(points, self.excitations, unknown=0.0)

    def signature_at(self, points: np.ndarray) -> np.ndarray:
        """Expected log-RMS at synergy-space points: the space spacing is judged in.

        Reconstructed through the synergies rather than read off the raw pose,
        because a single 0.25 s pose is a noisy RMS estimate and max-min
        spacing on noisy samples selects the noise. Baseline noise adds in
        power, so it is folded in before the log rather than ignored.
        """
        template = self.rms_at(points)
        return np.log(np.sqrt(np.square(template) + np.square(self.rest_rms)))

    @property
    def signatures(self) -> np.ndarray:
        return self.signature_at(self.activations)

    def rms_at(self, points: np.ndarray) -> np.ndarray:
        """The RMS template a teach-back target shows: exact, from the synergies."""
        return np.atleast_2d(points) @ self.nmf.components_ * self.scale


def fit_cartograph(
    babble: BabbleRecording,
    n_synergies: int = NATURAL_SYNERGIES.shape[0],
    seed: int = 0,
) -> Cartograph:
    scale = float(babble.rms.max())
    data = babble.rms / scale
    nmf = NMF(n_components=n_synergies, init="nndsvda", random_state=seed, max_iter=2000)
    nmf.fit(data)
    # transform, not fit_transform: the map must be reproducible by embed().
    activations = nmf.transform(data)
    residual = data - activations @ nmf.components_
    explained = 1.0 - float(np.sum(np.square(residual)) / np.sum(np.square(data)))

    # Scott's rule: the kernel widens with dimension and narrows with data.
    n, d = activations.shape
    spread = float(np.mean(activations.std(axis=0)))
    bandwidth = spread * n ** (-1.0 / (d + 4))

    strength = babble.rms.mean(axis=1)
    cartograph = Cartograph(
        nmf=nmf,
        scale=scale,
        activations=activations,
        bandwidth=bandwidth,
        effort=strength / strength.max(),
        excitations=babble.excitations,
        rms=babble.rms,
        rest=np.zeros(n_synergies),
        explained_variance=explained,
        rest_rms=babble.rest_rms,
    )
    cartograph.rest = cartograph.embed(babble.rest_rms)[0]
    cartograph.density_norm = float(cartograph._kernel(activations).sum(axis=1).max())

    mean = activations.mean(axis=0)
    _, _, vt = np.linalg.svd(activations - mean, full_matrices=False)
    axes = vt[:2].T
    # Deterministic orientation: SVD sign is arbitrary.
    axes = axes * np.where(axes[np.argmax(np.abs(axes), axis=0), [0, 1]] < 0, -1.0, 1.0)
    raw = (activations - mean) @ axes
    # One scale for both axes keeps the display isotropic; a 10% margin keeps
    # babble off the border.
    low, high = raw.min(axis=0), raw.max(axis=0)
    span = float((high - low).max()) * 1.25
    cartograph.map_mean = mean
    cartograph.map_axes = axes
    cartograph.map_offset = (low + high) / 2.0 - span / 2.0
    cartograph.map_span = span
    return cartograph


# How much a uniform shift in log-RMS counts toward separation, relative to a
# change of pattern. A person cannot hold a contraction level to within the
# 10-15% that separates two same-pattern prototypes, and a gain-only
# difference is exactly what CLAUDE.md forbids between catalogue gestures.
LEVEL_WEIGHT = 0.25


def _pattern_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    diff = a - b
    level = diff.mean(axis=-1, keepdims=True)
    shape = np.linalg.norm(diff - level, axis=-1)
    return np.sqrt(np.square(shape) + np.square(LEVEL_WEIGHT * level[..., 0]))


def _separability(distance: np.ndarray, min_distance: float) -> np.ndarray:
    return np.clip(distance / (2.0 * min_distance), 0.0, 1.0)


def _default_spacing(distances: np.ndarray) -> float:
    """A quarter of the median pairwise distance: scaled to the user's own
    range, so a weaker user is not refused a vocabulary outright."""
    return 0.25 * float(np.median(distances[np.triu_indices(len(distances), 1)]))


def default_min_distance(cartograph: Cartograph) -> float:
    signatures = cartograph.signatures
    return _default_spacing(_pattern_distance(signatures[:, None], signatures[None, :]))


# Execution variability, as log-normal standard deviations: overall contraction
# level, and each muscle independently.
LEVEL_JITTER = 0.15
MUSCLE_JITTER = 0.08


def _attempt(excitation: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """One attempt at a target. Nobody reproduces a target exactly: each
    attempt drifts in overall level and per muscle. Without this, two
    prototypes 10% apart in level would score as perfectly separable."""
    excitation = np.asarray(excitation, dtype=np.float64)
    drift = rng.normal(0.0, LEVEL_JITTER) + rng.normal(
        0.0, MUSCLE_JITTER, excitation.shape
    )
    return np.clip(excitation * np.exp(drift), 0.0, 1.0)


@dataclass(frozen=True)
class ProbeConfig:
    """The repeat probe: the spec's "local covariance of repeats".

    After babble, the wearer re-attempts a shortlist of poses a few times. At
    the defaults that is 40 x 3 x 0.5 s, about a minute. A single babble pose
    says where a user has been once; only repeats say whether they can get
    back there. Planning on single samples chose saturated poses whose
    repeated attempts collapsed onto each other.
    """

    shortlist: int = 40
    repeats: int = 3
    seconds: float = 0.5


@dataclass(frozen=True)
class Repeats:
    means: np.ndarray  # (m, n_channels), mean window log-RMS per target
    spreads: np.ndarray  # (m,), pattern-weighted variance across windows


def probe_repeatability(
    targets: np.ndarray,
    subject: SubjectProfile,
    sim: SimulatorConfig,
    config: ProbeConfig,
    seed: int,
) -> Repeats:
    """Attempt each target `config.repeats` times and summarise the windows.

    Spread is measured on 200 ms windows, the unit the decoder classifies, so
    it includes both attempt-to-attempt drift and within-window noise.
    """
    window = WindowConfig(window_ms=200, hop_ms=100, sampling_rate_hz=sim.sampling_rate_hz)
    means, spreads = [], []
    for index, target in enumerate(np.atleast_2d(targets)):
        windows = []
        for repeat in range(config.repeats):
            attempt_seed = seed * 7919 + 200_000 + index * 31 + repeat
            attempt = _attempt(target, np.random.default_rng(attempt_seed))
            signal = simulate_excitation(
                attempt, config.seconds, sim, subject, seed=attempt_seed
            )
            windows.extend(np.log(_rms(frame)) for frame in frame_windows(signal, window))
        windows = np.asarray(windows)
        mean = windows.mean(axis=0)
        means.append(mean)
        spreads.append(float(np.mean(np.square(_pattern_distance(windows, mean)))))
    return Repeats(means=np.asarray(means), spreads=np.asarray(spreads))


def _fisher_distances(means: np.ndarray, spreads: np.ndarray) -> np.ndarray:
    """Pattern distance between means over the pooled spread: one unit is one
    standard deviation of the user's own repeat scatter."""
    gap = _pattern_distance(means[:, None], means[None, :])
    pooled = np.sqrt((spreads[:, None] + spreads[None, :]) / 2.0)
    return gap / pooled


def _place(
    distances: np.ndarray,
    base: np.ndarray,
    k: int,
    min_distance: float,
    weights: SynthesisWeights,
    sweeps: int,
) -> list[int]:
    """Greedy farthest-point placement, then coordinate ascent.

    `distances` is (m + 1, m + 1) with row 0 the rest anchor; `base` is the
    reachability-minus-effort score of the m candidates. Returns candidate
    indices, anchor excluded.
    """
    to_all = distances[1:]  # (m, m + 1): candidate to anchor-or-candidate

    def fixed(indices: list[int]) -> list[int]:
        return [0] + [index + 1 for index in indices]

    placed: list[int] = []
    for _ in range(k):
        nearest = to_all[:, fixed(placed)].min(axis=1)
        score = weights.separability * _separability(nearest, min_distance) + base
        score[nearest < min_distance] = -np.inf
        if not np.isfinite(score).any():
            raise ValueError(
                f"cannot place {k} prototypes {min_distance:.3g} apart "
                "in the reachable set"
            )
        placed.append(int(np.argmax(score)))

    for _ in range(sweeps):
        moved = False
        for i in range(k):
            rest_of = [index for j, index in enumerate(placed) if j != i]
            others = fixed(rest_of)
            to_candidate = to_all[:, others]  # column 0 is the anchor

            # Each other prototype's nearest neighbour, excluding prototype i.
            between = distances[np.ix_(others, others)].copy()
            np.fill_diagonal(between, np.inf)
            others_nearest = between.min(axis=1)[1:]

            own = _separability(to_candidate.min(axis=1), min_distance)
            theirs = _separability(
                np.minimum(to_candidate[:, 1:], others_nearest[None, :]), min_distance
            ).sum(axis=1)
            score = weights.separability * (own + theirs) + base
            score[to_candidate.min(axis=1) < min_distance] = -np.inf
            score[rest_of] = -np.inf

            best = int(np.argmax(score))
            if score[best] > score[placed[i]] + 1e-12:
                placed[i] = best
                moved = True
        if not moved:
            break

    return placed


def _base_score(
    cartograph: Cartograph, weights: SynthesisWeights
) -> tuple[np.ndarray, np.ndarray]:
    density = cartograph.density_at(cartograph.activations)
    base = weights.reachability * density - weights.effort * cartograph.effort_at(
        cartograph.activations
    )
    return base, density >= weights.reach_floor


def synthesize(
    cartograph: Cartograph,
    k: int,
    weights: SynthesisWeights = SynthesisWeights(),
    min_distance: float | None = None,
    sweeps: int = 10,
) -> np.ndarray:
    """Place k prototypes in synergy space; see `synthesize_poses`."""
    poses = synthesize_poses(cartograph, k, weights, min_distance, sweeps)
    return cartograph.activations[poses]


def synthesize_poses(
    cartograph: Cartograph,
    k: int,
    weights: SynthesisWeights = SynthesisWeights(),
    min_distance: float | None = None,
    sweeps: int = 10,
) -> list[int]:
    """Choose k babble poses as prototypes from babble alone, without repeats.

    Candidates are the babble poses themselves: every prototype is somewhere
    the user has demonstrably been, which is what makes it teachable.

    Spacing is measured in log-RMS, not synergy space. Surface EMG noise grows
    with amplitude, so raw distance rewards loud prototypes that are no easier
    to tell apart; the log stabilises the variance. Max-min spacing in raw
    synergy space lost to catalogue selection on every subject tried.

    Raises rather than returning fewer than k: a short vocabulary handed to the
    assignment step would silently leave a prosthetic function unmapped.
    """
    base, feasible = _base_score(cartograph, weights)
    poses = np.flatnonzero(feasible)
    points = np.vstack(
        [cartograph.signature_at(cartograph.rest), cartograph.signatures[poses]]
    )
    distances = _pattern_distance(points[:, None], points[None, :])
    if min_distance is None:
        min_distance = default_min_distance(cartograph)
    chosen = _place(distances, base[poses], k, min_distance, weights, sweeps)
    return [int(poses[index]) for index in chosen]


def shortlist_poses(
    cartograph: Cartograph, size: int, weights: SynthesisWeights = SynthesisWeights()
) -> list[int]:
    """Farthest-point sample of reachable poses, starting from rest: the poses
    worth spending repeat-probe time on."""
    _, feasible = _base_score(cartograph, weights)
    poses = np.flatnonzero(feasible)
    signatures = cartograph.signatures[poses]
    nearest = _pattern_distance(signatures, cartograph.signature_at(cartograph.rest)[0])
    chosen: list[int] = []
    for _ in range(min(size, len(poses))):
        best = int(np.argmax(nearest))
        chosen.append(best)
        nearest = np.minimum(nearest, _pattern_distance(signatures, signatures[best]))
        nearest[chosen] = -np.inf
    return [int(poses[index]) for index in chosen]


def synthesize_from_repeats(
    cartograph: Cartograph,
    poses: list[int],
    repeats: Repeats,
    rest: Repeats,
    k: int,
    weights: SynthesisWeights = SynthesisWeights(),
    min_distance: float | None = None,
    sweeps: int = 10,
) -> list[int]:
    """Choose k of the probed poses, spacing them in repeat-scatter units."""
    base, _ = _base_score(cartograph, weights)
    distances = _fisher_distances(
        np.vstack([rest.means, repeats.means]),
        np.concatenate([rest.spreads, repeats.spreads]),
    )
    if min_distance is None:
        min_distance = _default_spacing(distances[1:, 1:])
    chosen = _place(distances, base[poses], k, min_distance, weights, sweeps)
    return [poses[index] for index in chosen]


def _select_catalogue(distances: np.ndarray, names: list[str], k: int) -> list[str]:
    """Prior art: greedy max-min selection from a fixed catalogue.

    `distances` is (n + 1, n + 1) with row 0 the rest anchor.
    """
    chosen: list[int] = []
    for _ in range(k):
        fixed = [0] + [index + 1 for index in chosen]
        nearest = distances[1:, fixed].min(axis=1)
        nearest[chosen] = -np.inf
        chosen.append(int(np.argmax(nearest)))
    return [names[index] for index in chosen]


@dataclass(frozen=True)
class SelectionWeights:
    separability: float = 1.0
    effort: float = 0.5
    fatigue: float = 0.5


# Fatigue accrued over a sustained hold, per unit of mean excitation. This is a
# declared simulator assumption, not a measurement: fatigue here is an input to
# the simulator, so the MDF drop it produces tracks excitation closely and the
# fatigue term largely duplicates the effort term. On recorded holds the two
# separate -- that is why both are kept.
HOLD_FATIGUE_PER_EXCITATION = 1.0


def _median_frequency(signal: np.ndarray, sim: SimulatorConfig) -> float:
    window = WindowConfig(window_ms=200, hop_ms=100, sampling_rate_hz=sim.sampling_rate_hz)
    config = FeatureConfig(sampling_rate_hz=sim.sampling_rate_hz)
    values = [
        value
        for frame in frame_windows(signal, window)
        for name, value in extract_features(frame, config).items()
        if name.endswith("_mdf")
    ]
    return float(np.mean(values))


def fatigue_probe(
    targets: np.ndarray,
    subject: SubjectProfile,
    sim: SimulatorConfig,
    seed: int,
    seconds: float = 0.6,
) -> np.ndarray:
    """Relative median-frequency drop from the start to the end of a hold.

    Median-frequency decline is the standard surface-EMG fatigue marker. Start
    and end share one seed, so the spike trains are identical and the drop is
    the fatigue alone.
    """
    drops = []
    for index, target in enumerate(np.atleast_2d(targets)):
        hold_seed = seed * 7919 + 400_000 + index
        attempt = _attempt(target, np.random.default_rng(hold_seed))
        accrued = min(1.0, HOLD_FATIGUE_PER_EXCITATION * float(attempt.mean()))
        fresh = _median_frequency(
            simulate_excitation(attempt, seconds, sim, subject, seed=hold_seed), sim
        )
        tired = _median_frequency(
            simulate_excitation(
                attempt, seconds, sim, subject, seed=hold_seed, fatigue=accrued
            ),
            sim,
        )
        drops.append((fresh - tired) / fresh)
    return np.asarray(drops)


def select_weighted(
    distances: np.ndarray,
    effort: np.ndarray,
    fatigue: np.ndarray,
    names: list[str],
    k: int,
    weights: SelectionWeights = SelectionWeights(),
    min_distance: float | None = None,
) -> list[str]:
    """The Pillar 1 fallback: catalogue selection weighted by effort and fatigue.

    Synthesis did not beat catalogue selection in the simulator (see the module
    docstring), so this is what the cartograph is for until it does: choose
    the k catalogue gestures that stay separable for this user at the least
    effort and fatigue, rather than the most separable regardless of cost.

    `distances` is (n + 1, n + 1) with row 0 the rest anchor. Effort and
    fatigue are scaled to their largest value so the weights compare.
    """

    def scaled(values: np.ndarray) -> np.ndarray:
        top = float(np.max(values))
        return values / top if top > 0.0 else values

    base = -weights.effort * scaled(np.asarray(effort, dtype=np.float64)) - (
        weights.fatigue * scaled(np.clip(np.asarray(fatigue, dtype=np.float64), 0.0, None))
    )
    if min_distance is None:
        min_distance = _default_spacing(distances[1:, 1:])
    placement = SynthesisWeights(separability=weights.separability)
    chosen = _place(distances, base, k, min_distance, placement, sweeps=10)
    return [names[index] for index in chosen]


def _vocabulary_report(
    classes: dict[str, np.ndarray],
    subject: SubjectProfile,
    sim: SimulatorConfig,
    reps: int,
    rep_seconds: float,
    seed: int,
) -> dict[str, object]:
    window = WindowConfig(window_ms=200, hop_ms=100, sampling_rate_hz=sim.sampling_rate_hz)
    feature_config = FeatureConfig(sampling_rate_hz=sim.sampling_rate_hz)
    rows, labels, groups = [], [], []

    for label, (name, excitation) in enumerate(classes.items()):
        for rep in range(reps):
            rep_seed = seed * 7919 + label * 101 + rep
            attempt = _attempt(excitation, np.random.default_rng(rep_seed))
            signal = simulate_excitation(attempt, rep_seconds, sim, subject, seed=rep_seed)
            for frame in frame_windows(signal, window):
                features = extract_features(frame, feature_config)
                rows.append(feature_vector(features, sim.n_electrodes))
                labels.append(label)
                groups.append(rep)

    x = StandardScaler().fit_transform(np.asarray(rows))
    y = np.asarray(labels)
    names = list(classes)

    worst = (np.inf, "", "")
    for a in range(len(names)):
        for b in range(a + 1, len(names)):
            xa, xb = x[y == a], x[y == b]
            spread = np.sqrt((xa.var(axis=0).sum() + xb.var(axis=0).sum()) / 2.0)
            gap = float(np.linalg.norm(xa.mean(axis=0) - xb.mean(axis=0)) / spread)
            if gap < worst[0]:
                worst = (gap, names[a], names[b])

    model = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
    folds = GroupKFold(n_splits=min(reps, 3))
    accuracy = cross_val_score(model, x, y, groups=groups, cv=folds).mean()

    return {
        "classes": names,
        "min_pairwise_separation": round(worst[0], 4),
        "worst_pair": [worst[1], worst[2]],
        "cv_accuracy": round(float(accuracy), 4),
        "mean_excitation": round(
            float(np.mean([classes[name].mean() for name in names if name != "rest"])), 4
        ),
    }


def compare_vocabularies(
    cartograph: Cartograph,
    subject: SubjectProfile,
    sim: SimulatorConfig,
    k: int = 5,
    reps: int = 4,
    rep_seconds: float = 1.0,
    seed: int = 0,
    weights: SynthesisWeights = SynthesisWeights(),
    min_distance: float | None = None,
    probe: ProbeConfig | None = ProbeConfig(),
    selection: SelectionWeights = SelectionWeights(),
) -> dict[str, object]:
    """The Phase 2 gate: synthesised vocabulary versus catalogue selection.

    Both sides get rest plus k gestures, and both are scored with the decoder's
    real feature set on fresh attempts. The catalogue side is the strongest
    selection baseline, not the default vocabulary: greedy max-min over the
    catalogue. With a probe, both sides plan from the same repeat probe -- the
    catalogue's nine gestures are probed exactly as the shortlist is -- so the
    only difference between them is where the candidates come from.
    """
    catalogue_names = [g for g in GESTURE_SYNERGIES if g != "rest"]
    catalogue_targets = np.asarray([GESTURE_SYNERGIES[g] for g in catalogue_names])

    if probe is None:
        poses = synthesize_poses(cartograph, k, weights, min_distance)
        points = [cartograph.signature_at(cartograph.rest)[0]]
        for offset, target in enumerate(catalogue_targets):
            signal = simulate_excitation(
                target, 1.0, sim, subject, seed=seed * 7919 + 50_000 + offset
            )
            points.append(cartograph.signature_at(cartograph.embed(_rms(signal)))[0])
        points = np.asarray(points)
        catalogue_distances = _pattern_distance(points[:, None], points[None, :])
    else:
        rest = probe_repeatability(REST_EXCITATION, subject, sim, probe, seed=seed + 1)
        shortlist = shortlist_poses(cartograph, probe.shortlist, weights)
        repeats = probe_repeatability(
            cartograph.excitations[shortlist], subject, sim, probe, seed=seed + 2
        )
        poses = synthesize_from_repeats(
            cartograph, shortlist, repeats, rest, k, weights, min_distance
        )
        probed = probe_repeatability(catalogue_targets, subject, sim, probe, seed=seed + 3)
        catalogue_distances = _fisher_distances(
            np.vstack([rest.means, probed.means]),
            np.concatenate([rest.spreads, probed.spreads]),
        )

    # Teach-back targets the chosen pose itself. Kernel-averaging neighbours
    # instead pulled prototypes toward each other: one pair planned 0.99 apart
    # in log-RMS was realised 0.36 apart.
    synthesized = {"rest": REST_EXCITATION}
    for index, pose in enumerate(poses, start=1):
        synthesized[f"proto_{index}"] = cartograph.excitations[pose]

    def vocabulary(names: list[str]) -> dict[str, np.ndarray]:
        chosen = {"rest": REST_EXCITATION}
        for name in names:
            chosen[name] = np.asarray(GESTURE_SYNERGIES[name], dtype=np.float64)
        return chosen

    catalogue = vocabulary(_select_catalogue(catalogue_distances, catalogue_names, k))

    # Effort is read off the probe the user already did, relative to the
    # loudest babble pose; fatigue needs one sustained hold per gesture.
    if probe is None:
        effort = catalogue_targets.mean(axis=1)
    else:
        effort = np.exp(probed.means).mean(axis=1) / float(cartograph.rms.mean(axis=1).max())
    fatigue = fatigue_probe(catalogue_targets, subject, sim, seed=seed + 4)
    weighted = vocabulary(
        select_weighted(catalogue_distances, effort, fatigue, catalogue_names, k, selection)
    )
    costs = {
        name: {"effort": round(float(e), 4), "fatigue": round(float(f), 4)}
        for name, e, f in zip(catalogue_names, effort, fatigue)
    }

    def report(classes: dict[str, np.ndarray]) -> dict[str, object]:
        # Scoring seeds sit far from every probe seed: planning and scoring
        # must never see the same attempts.
        result = _vocabulary_report(classes, subject, sim, reps, rep_seconds, seed + 1_000)
        named = [name for name in classes if name in costs]
        if named:
            result["mean_fatigue"] = round(
                float(np.mean([costs[name]["fatigue"] for name in named])), 4
            )
        return result

    return {
        "k": k,
        "probe": None if probe is None else dataclasses.asdict(probe),
        "selection_weights": dataclasses.asdict(selection),
        "prototypes_map": [
            [round(float(v), 4) for v in p]
            for p in cartograph.to_map(cartograph.activations[poses])
        ],
        "selection_costs": costs,
        "synthesized": report(synthesized),
        "catalogue": report(catalogue),
        "weighted_catalogue": report(weighted),
    }


# Residual-limb stand-in: finger flexor and extensor (fds, edc) at 30% of their
# usual reach, wrist muscles intact. The catalogue assumes an intact forearm;
# this is the case Pillar 1 claims to help.
FINGER_ATROPHY = (1.0, 0.3, 1.0, 1.0, 0.3, 1.0)


def cartography_gate(
    n_subjects: int = 5,
    k: int = 5,
    reps: int = 4,
    babble: BabbleConfig = BabbleConfig(),
    sim: SimulatorConfig = SimulatorConfig(),
) -> dict[str, object]:
    """Run the Phase 2 gate over intact and impaired simulated cohorts.

    Two findings per subject. The gate proper -- a synthesised vocabulary
    beats catalogue selection on worst-pair separability -- is reported with a
    win count, not a mean, because one subject for whom synthesis fails is the
    finding. The fallback -- effort- and fatigue-weighted catalogue selection
    against max-min selection -- is reported as what it costs in separability
    and what it saves in effort and fatigue, as cohort medians.
    """
    cohorts: dict[str, object] = {}
    for cohort, reach in (("intact", None), ("finger_atrophy", FINGER_ATROPHY)):
        subjects = []
        for index in range(n_subjects):
            subject = make_subject(f"carto{index:02d}", seed=1000 + index)
            if reach is not None:
                subject = dataclasses.replace(
                    subject, synergy_gain=subject.synergy_gain * np.asarray(reach)
                )
            recording = record_babble(dataclasses.replace(babble, seed=index), sim, subject)
            cartograph = fit_cartograph(recording, seed=index)
            result = compare_vocabularies(cartograph, subject, sim, k=k, reps=reps, seed=index)
            result["subject"] = subject.subject_id
            result["explained_variance"] = round(cartograph.explained_variance, 4)
            result["synthesis_wins"] = bool(
                result["synthesized"]["min_pairwise_separation"]
                > result["catalogue"]["min_pairwise_separation"]
            )
            plain, weighted = result["catalogue"], result["weighted_catalogue"]
            result["weighted_vs_maxmin"] = {
                "separability_ratio": round(
                    weighted["min_pairwise_separation"] / plain["min_pairwise_separation"], 4
                ),
                "excitation_change": round(
                    weighted["mean_excitation"] - plain["mean_excitation"], 4
                ),
                "fatigue_change": round(weighted["mean_fatigue"] - plain["mean_fatigue"], 4),
                "accuracy_change": round(weighted["cv_accuracy"] - plain["cv_accuracy"], 4),
            }
            subjects.append(result)

        def median(key: str) -> float:
            return round(
                float(np.median([row["weighted_vs_maxmin"][key] for row in subjects])), 4
            )

        cohorts[cohort] = {
            "wins": sum(int(row["synthesis_wins"]) for row in subjects),
            "n_subjects": n_subjects,
            "weighted_selection": {
                key: median(key)
                for key in (
                    "separability_ratio",
                    "excitation_change",
                    "fatigue_change",
                    "accuracy_change",
                )
            },
            "subjects": subjects,
        }

    return {
        "k": k,
        "reps": reps,
        "level_jitter": LEVEL_JITTER,
        "muscle_jitter": MUSCLE_JITTER,
        "level_weight": LEVEL_WEIGHT,
        "hold_fatigue_per_excitation": HOLD_FATIGUE_PER_EXCITATION,
        "cohorts": cohorts,
        "caveat": (
            "Simulated babble and simulated users. Fatigue accrual is a simulator "
            "assumption proportional to excitation, so the fatigue term largely "
            "duplicates effort here. Learnability (teach-back) is not measured. "
            "Not a human result."
        ),
    }
