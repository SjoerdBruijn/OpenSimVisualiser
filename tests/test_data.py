from pathlib import Path

import numpy as np
import pytest

from OpenSimVisualiser.data import DEFAULT_BUNDLE, _load_trial, read_grf, read_storage, read_trc


def _fake_c3d(*, markers: bool, force_platforms: bool) -> dict:
    point_count = 2 if markers else 0
    point_frames = 3 if markers else 0
    points = np.zeros((4, point_count, point_frames), dtype=float)
    if markers:
        points[:3, 0, :] = np.array([[1000.0], [2000.0], [3000.0]])
        points[:3, 1, :] = np.array([[4000.0], [5000.0], [6000.0]])

    residuals = np.ones((1, point_count, point_frames), dtype=float)
    if markers:
        residuals[0, 0, 1] = -1.0

    platforms = []
    if force_platforms:
        sample_count = 30
        force = np.zeros((3, sample_count), dtype=float)
        force[2, :] = 100.0
        center_of_pressure = np.zeros_like(force)
        center_of_pressure[0, :] = 1000.0
        center_of_pressure[1, :] = 2000.0
        moment = np.zeros_like(force)
        moment[0, :] = 1000.0
        platforms.append(
            {
                "force": force,
                "center_of_pressure": center_of_pressure,
                "moment": moment,
                "unit_force": "N",
                "unit_position": "mm",
                "unit_moment": "Nmm",
            }
        )

    return {
        "header": {
            "points": {"frame_rate": 100.0, "first_frame": 10},
            "analogs": {"frame_rate": 1000.0, "first_frame": 200},
        },
        "parameters": {
            "POINT": {
                "RATE": {"value": [100.0]},
                "UNITS": {"value": ["mm"]},
                "LABELS": {"value": ["LASI", "RASI"] if markers else []},
            },
            "ANALOG": {"RATE": {"value": [1000.0]}},
        },
        "data": {
            "points": points,
            "meta_points": {"residuals": residuals},
            "analogs": np.empty((1, 0, 0)),
            "platform": platforms,
        },
    }


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


def test_load_c3d_markers_and_force_platforms(monkeypatch: pytest.MonkeyPatch) -> None:
    import ezc3d

    monkeypatch.setattr(
        ezc3d,
        "c3d",
        lambda *_args, **_kwargs: _fake_c3d(markers=True, force_platforms=True),
    )
    trial = _load_trial(c3d_path="trial.c3d")

    assert trial.c3d_path == Path("trial.c3d")
    assert trial.model_path is None
    assert trial.measured_marker_labels == ["LASI", "RASI"]
    assert trial.measured_markers is not None
    assert np.allclose(trial.measured_markers[0, 0], [1.0, 3.0, -2.0])
    assert np.isnan(trial.measured_markers[1, 0]).all()
    assert trial.grf_forces is not None
    assert trial.grf_points is not None
    assert trial.grf_torques is not None
    assert trial.grf_forces.shape == (3, 1, 3)
    assert np.allclose(trial.grf_forces[0, 0], [0.0, 100.0, 0.0])
    assert np.allclose(trial.grf_points[0, 0], [1.0, 0.0, -2.0])
    assert np.allclose(trial.grf_torques[0, 0], [1.0, 0.0, 0.0])
    assert trial.grf_labels == ["platform 1"]


def test_load_marker_only_c3d(monkeypatch: pytest.MonkeyPatch) -> None:
    import ezc3d

    monkeypatch.setattr(
        ezc3d,
        "c3d",
        lambda *_args, **_kwargs: _fake_c3d(markers=True, force_platforms=False),
    )
    trial = _load_trial(c3d_path="markers.c3d")

    assert trial.measured_markers is not None
    assert trial.grf_forces is None
    assert trial.frame_count == 3


def test_load_force_platform_only_c3d(monkeypatch: pytest.MonkeyPatch) -> None:
    import ezc3d

    monkeypatch.setattr(
        ezc3d,
        "c3d",
        lambda *_args, **_kwargs: _fake_c3d(markers=False, force_platforms=True),
    )
    trial = _load_trial(c3d_path="forces.c3d")

    assert trial.measured_markers is None
    assert trial.grf_forces is not None
    assert trial.frame_count == 30
    assert trial.times[0] == pytest.approx(0.2)


def test_c3d_route_cannot_be_mixed_with_other_inputs() -> None:
    with pytest.raises(ValueError, match="separate input route"):
        _load_trial(c3d_path="trial.c3d", marker_path="markers.trc")
