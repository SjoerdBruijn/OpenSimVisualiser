from pathlib import Path

import numpy as np

from OpenSimVisualiser.data import DEFAULT_BUNDLE, read_grf, read_storage, read_trc


def test_bundled_example_is_available() -> None:
    assert DEFAULT_BUNDLE.is_dir()
    assert (DEFAULT_BUNDLE / "subject_walk_scaled.osim").is_file()
    assert (DEFAULT_BUNDLE / "Geometry").is_dir()


def test_read_bundled_coordinates() -> None:
    table = read_storage(DEFAULT_BUNDLE / "coordinates.sto")

    assert table.times.ndim == 1
    assert table.values.shape == (table.times.size, len(table.labels))
    assert table.times.size > 1
    assert np.all(np.diff(table.times) >= 0)


def test_read_bundled_markers_in_metres() -> None:
    times, labels, positions = read_trc(DEFAULT_BUNDLE / "marker_trajectories.trc")

    assert positions.shape == (times.size, len(labels), 3)
    assert len(labels) > 0
    assert np.nanmax(np.abs(positions)) < 10.0


def test_read_bundled_ground_reactions() -> None:
    times, forces, points, torques = read_grf(DEFAULT_BUNDLE / "grf_walk.mot")

    expected_shape = (times.size, 2, 3)
    assert forces.shape == expected_shape
    assert points.shape == expected_shape
    assert torques.shape == expected_shape
    assert np.isfinite(forces).any()


def test_default_bundle_path_is_inside_package() -> None:
    package_directory = Path(__import__("OpenSimVisualiser").__file__).resolve().parent

    assert DEFAULT_BUNDLE.parent == package_directory
