import math

import numpy as np
import pytest

from neurogrip.anatomy import (
    MUSCLE_GROUPS,
    MUSCLES,
    ForearmGeometry,
    attenuation_matrix,
    electrode_positions,
    muscle_group_weights,
    muscle_positions,
)


def test_six_named_muscles_at_distinct_angles():
    assert len(MUSCLES) == 6
    angles = [m.angle_rad for m in MUSCLES]
    assert len(set(angles)) == 6
    assert all(0.0 <= a < 2 * np.pi for a in angles)


def test_electrodes_sit_on_the_skin_circle():
    geometry = ForearmGeometry()
    positions = electrode_positions(8, 0.0, geometry)
    assert positions.shape == (8, 2)
    radii = np.linalg.norm(positions, axis=1)
    assert np.allclose(radii, geometry.skin_radius_mm)


def test_electrode_shift_rotates_the_ring():
    geometry = ForearmGeometry()
    unshifted = electrode_positions(8, 0.0, geometry)
    shifted = electrode_positions(8, np.pi / 4, geometry)
    # An eighth turn on an eight-electrode ring maps electrode 0 onto electrode 1.
    assert np.allclose(shifted[0], unshifted[1], atol=1e-9)


def test_muscles_sit_inside_the_skin_circle():
    geometry = ForearmGeometry()
    radii = np.linalg.norm(muscle_positions(geometry), axis=1)
    assert np.allclose(radii, geometry.muscle_radius_mm)
    assert geometry.muscle_radius_mm < geometry.skin_radius_mm


def test_attenuation_falls_off_with_distance_and_is_bounded():
    geometry = ForearmGeometry()
    matrix = attenuation_matrix(8, 0.0, geometry)
    assert matrix.shape == (8, 6)
    assert np.all(matrix > 0.0)
    assert np.all(matrix <= 1.0)


def test_nearest_electrode_sees_each_muscle_most_strongly():
    """This is what produces realistic crosstalk: every electrode sees every
    muscle, but the closest one dominates."""
    geometry = ForearmGeometry()
    matrix = attenuation_matrix(16, 0.0, geometry)
    electrodes = electrode_positions(16, 0.0, geometry)
    muscles = muscle_positions(geometry)
    for muscle_index in range(len(MUSCLES)):
        distances = np.linalg.norm(electrodes - muscles[muscle_index], axis=1)
        assert int(np.argmax(matrix[:, muscle_index])) == int(np.argmin(distances))


def test_rejects_bad_electrode_counts():
    with pytest.raises(ValueError):
        electrode_positions(0, 0.0, ForearmGeometry())


def test_muscle_group_weights_are_normalised_per_group():
    weights = muscle_group_weights(8)
    assert set(weights) == set(MUSCLE_GROUPS)
    for group, values in weights.items():
        assert len(values) == 8, group
        assert all(value >= 0.0 for value in values), group
        assert math.isclose(sum(values), 1.0, abs_tol=1e-12), group


def test_every_muscle_belongs_to_exactly_one_group():
    grouped = [name for members in MUSCLE_GROUPS.values() for name in members]
    assert sorted(grouped) == sorted(m.name for m in MUSCLES)
    assert len(grouped) == len(set(grouped))


def test_flexor_and_extensor_groups_peak_on_opposite_sides_of_the_ring():
    """The whole point of the weighting: it has to separate the antagonists.

    If digit flexion and digit extension weighted the same electrodes, tinting
    the hand by group would show the same colour whichever way it moved.
    """
    weights = muscle_group_weights(8)
    flexor_peak = max(range(8), key=lambda i: weights["digit_flexor"][i])
    extensor_peak = max(range(8), key=lambda i: weights["digit_extensor"][i])
    separation = min((flexor_peak - extensor_peak) % 8, (extensor_peak - flexor_peak) % 8)
    assert separation >= 3, (flexor_peak, extensor_peak)


def test_group_weights_follow_electrode_shift():
    """Shift the ring and the weighting must follow it, not stay behind."""
    base = muscle_group_weights(8, shift_rad=0.0)["digit_flexor"]
    shifted = muscle_group_weights(8, shift_rad=math.pi / 4)["digit_flexor"]
    assert base != pytest.approx(shifted, abs=1e-6)
    # An eighth turn is exactly one electrode step. Rotating the ring
    # anticlockwise puts electrode i where electrode i+1 used to be, so the
    # weight pattern moves the other way along the index.
    rotated = base[1:] + base[:1]
    assert shifted == pytest.approx(rotated, abs=1e-9)
