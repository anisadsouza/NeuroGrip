import numpy as np
import pytest

from neurogrip.anatomy import (
    MUSCLES,
    ForearmGeometry,
    attenuation_matrix,
    electrode_positions,
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
