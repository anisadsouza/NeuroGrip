"""Forearm cross-section geometry: muscles, electrode ring, volume conduction.

Defined once and shared by the simulator (which uses it to mix muscle sources
into electrode channels) and by the Pillar 4 attribution ring (which uses it to
back-project per-channel attributions onto anatomy). Angles are measured
counter-clockwise from the radial (thumb) side, viewed distally.
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
