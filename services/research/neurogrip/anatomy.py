"""Forearm cross-section geometry: muscles, electrode ring, volume conduction.

Defined once and shared by the simulator (which uses it to mix muscle sources
into electrode channels) and by the Pillar 4 attribution ring (which uses it to
back-project per-channel attributions onto anatomy). Angles are measured
counter-clockwise from the radial (thumb) side, viewed distally.

**Where this cross-section is.** Around the forearm at the height of the
radio-humeral joint -- the elbow joint line -- which is where NinaPro places
its eight-electrode ring and where the bellies of the superficial finger
flexors and extensors sit. That proximal third of the forearm is the only place
a surface ring can see the muscles that move the fingers: they are extrinsic
muscles, with bellies near the elbow and long tendons running to the digits.

Electrodes further down the forearm sit over tendon rather than muscle and
read far less; electrodes on the upper arm see biceps and triceps, which move
the elbow and say nothing about the hand. NinaPro DB2 does carry two upper-arm
channels, but as extra context alongside the ring, never in place of it.

The model is a single cross-section and has no longitudinal extent, so nothing
here depends on that placement numerically. It is recorded because it is what
the geometry means, and because it is the first thing to get wrong when real
electrodes are fitted.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Muscle:
    name: str
    angle_rad: float
    label: str


# Six superficial forearm muscles at representative cross-sectional angles.
MUSCLES: tuple[Muscle, ...] = (
    Muscle("fcr", math.radians(30.0), "Flexor carpi radialis"),
    Muscle("fds", math.radians(80.0), "Flexor digitorum superficialis"),
    Muscle("fcu", math.radians(140.0), "Flexor carpi ulnaris"),
    Muscle("ecu", math.radians(210.0), "Extensor carpi ulnaris"),
    Muscle("edc", math.radians(265.0), "Extensor digitorum communis"),
    Muscle("ecr", math.radians(320.0), "Extensor carpi radialis"),
)


@dataclass(frozen=True)
class ForearmGeometry:
    skin_radius_mm: float = 40.0
    muscle_radius_mm: float = 25.0
    conduction_lambda_mm: float = 15.0


def _ring(radius_mm: float, angles: np.ndarray) -> np.ndarray:
    return np.stack([radius_mm * np.cos(angles), radius_mm * np.sin(angles)], axis=1)


def electrode_positions(
    n_electrodes: int, shift_rad: float, geometry: ForearmGeometry
) -> np.ndarray:
    """Evenly spaced electrodes on the skin circle, rotated by shift_rad.

    shift_rad models the donning variation that makes electrode shift the
    dominant real-world failure mode for myoelectric pattern recognition.
    """
    if n_electrodes < 1:
        raise ValueError("n_electrodes must be at least 1")
    angles = np.arange(n_electrodes, dtype=np.float64) * (2.0 * np.pi / n_electrodes)
    return _ring(geometry.skin_radius_mm, angles + shift_rad)


def muscle_positions(geometry: ForearmGeometry) -> np.ndarray:
    angles = np.array([m.angle_rad for m in MUSCLES], dtype=np.float64)
    return _ring(geometry.muscle_radius_mm, angles)


def attenuation_matrix(
    n_electrodes: int, shift_rad: float, geometry: ForearmGeometry
) -> np.ndarray:
    """Volume-conduction gain from each muscle to each electrode.

    Exponential decay with distance, exp(-d / lambda). Every electrode sees
    every muscle to some degree, which is precisely the crosstalk that makes
    single-channel thresholding inadequate and pattern recognition necessary.
    """
    electrodes = electrode_positions(n_electrodes, shift_rad, geometry)
    muscles = muscle_positions(geometry)
    distances = np.linalg.norm(electrodes[:, None, :] - muscles[None, :, :], axis=2)
    return np.exp(-distances / geometry.conduction_lambda_mm)


#: Functional grouping of the six modelled muscles.
#:
#: The forearm flexors and extensors split twice over: by direction (flex or
#: extend) and by what they act on (the digits or the wrist). Those four groups
#: are what an eight-channel ring can actually resolve. It cannot resolve one
#: finger from another -- `fds` flexes all four and `edc` extends all four -- so
#: any display claiming per-finger sensing from these electrodes would be
#: claiming more than the signal contains.
MUSCLE_GROUPS: dict[str, tuple[str, ...]] = {
    "digit_flexor": ("fds",),
    "digit_extensor": ("edc",),
    "wrist_flexor": ("fcr", "fcu"),
    "wrist_extensor": ("ecu", "ecr"),
}


def muscle_group_weights(
    n_electrodes: int,
    shift_rad: float = 0.0,
    geometry: ForearmGeometry | None = None,
) -> dict[str, list[float]]:
    """Per-electrode weights for each functional muscle group.

    The volume-conduction matrix says how much of each muscle each electrode
    sees. Summing the columns of a group and normalising gives the weighting
    that best reads that group's drive off the electrode ring.

    Exported into the replay manifest so the browser can tint the hand by which
    muscles are driving it without reimplementing the anatomy -- the same
    argument that keeps the simulator out of the browser.
    """
    geometry = geometry or ForearmGeometry()
    names = [m.name for m in MUSCLES]
    gains = attenuation_matrix(n_electrodes, shift_rad, geometry)

    weights: dict[str, list[float]] = {}
    for group, members in MUSCLE_GROUPS.items():
        columns = [names.index(name) for name in members]
        summed = gains[:, columns].sum(axis=1)
        total = float(summed.sum())
        if total <= 0.0:
            raise ValueError(f"group {group!r} has no gain to any electrode")
        weights[group] = (summed / total).tolist()
    return weights
