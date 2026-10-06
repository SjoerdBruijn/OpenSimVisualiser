from pathlib import Path

import numpy as np
import pytest

from OpenSimVisualiser.data import DEFAULT_BUNDLE, _load_trial, read_grf, read_storage, read_trc


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


def test_load_markers_without_model() -> None:
    trial = _load_trial(marker_path=DEFAULT_BUNDLE / "marker_trajectories.trc")

    assert trial.model_path is None
    assert trial.measured_markers is not None
    assert trial.measured_markers.shape == (
        trial.frame_count,
        len(trial.measured_marker_labels),
        3,
    )
    assert trial.coordinates.shape == (trial.frame_count, 0)


def test_load_grf_without_model() -> None:
    trial = _load_trial(grf_path=DEFAULT_BUNDLE / "grf_walk.mot")

    assert trial.model_path is None
    assert trial.grf_forces is not None
    assert trial.grf_forces.shape == (trial.frame_count, 2, 3)
    assert trial.measured_markers is None


def test_load_markers_and_grf_without_model() -> None:
    marker_times, _labels, _positions = read_trc(DEFAULT_BUNDLE / "marker_trajectories.trc")
    trial = _load_trial(
        marker_path=DEFAULT_BUNDLE / "marker_trajectories.trc",
        grf_path=DEFAULT_BUNDLE / "grf_walk.mot",
    )

    assert np.array_equal(trial.times, marker_times)
    assert trial.measured_markers is not None
    assert trial.grf_forces is not None
    assert trial.grf_forces.shape[0] == trial.frame_count


def test_load_trial_requires_plottable_input() -> None:
    with pytest.raises(ValueError, match="Provide at least one"):
        _load_trial()
