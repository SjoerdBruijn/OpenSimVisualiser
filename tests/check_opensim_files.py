"""Regression checks for standard OpenSim trial files and output-folder models."""

from pathlib import Path

import numpy as np
import pytest

from OpenSimVisualiser.data import _geometry_mesh_path, _load_trial, read_grf, read_trc


def _write_marker_file(path: Path, units: str, scale: float) -> Path:
    path.write_text(
        "PathFileType\t4\t(X/Y/Z)\tmarkers.trc\n"
        "DataRate\tCameraRate\tNumFrames\tNumMarkers\tUnits\tOrigDataRate\n"
        f"100\t100\t2\t2\t{units}\t100\n"
        "Frame#\tTime\tbox - 1\t\t\tLeft hand\t\t\n"
        "\t\tX1\tY1\tZ1\tX2\tY2\tZ2\n"
        f"1\t0.0\t{scale}\t\t{3 * scale}\t{4 * scale}\t{5 * scale}\t{6 * scale}\n"
        f"2\t0.01\t{2 * scale}\t{3 * scale}\t{4 * scale}\t\t\t\n",
        encoding="utf-8",
    )
    return path


@pytest.mark.parametrize(("units", "scale"), [("m", 1), ("mm", 1000), ("cm", 100)])
def check_trc_preserves_names_gaps_and_units(tmp_path: Path, units: str, scale: float) -> None:
    times, labels, points = read_trc(_write_marker_file(tmp_path / "markers.trc", units, scale))
    assert labels == ["box - 1", "Left hand"]
    np.testing.assert_allclose(times, [0, 0.01])
    np.testing.assert_allclose(
        points, [[[1, np.nan, 3], [4, 5, 6]], [[2, 3, 4], [np.nan, np.nan, np.nan]]], equal_nan=True
    )


def check_trc_rejects_unknown_units(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Unsupported TRC length units"):
        read_trc(_write_marker_file(tmp_path / "markers.trc", "inches", 1))


def check_whitespace_trc_still_loads(tmp_path: Path) -> None:
    path = tmp_path / "markers.trc"
    path.write_text("DataRate Units\n100 mm\nFrame# Time hand\nX1 Y1 Z1\n1 0 1000 2000 3000\n")
    np.testing.assert_allclose(read_trc(path)[2], [[[1, 2, 3]]])


def _write_forces(path: Path, identifiers: list[str]) -> Path:
    columns = []
    values = []
    for index, identifier in enumerate(identifiers):
        suffix = f"{identifier}_" if identifier else ""
        columns += [f"ground_force_{suffix}{axis}" for axis in ("vx", "vy", "vz", "px", "py", "pz")]
        columns += [f"ground_torque_{suffix}{axis}" for axis in "xyz"]
        values += [10 + index, 700 + index, 30 + index, 0.1 + index, 0, 0.3 + index, 1, 2, 3]
    path.write_text(
        "inDegrees=no\nendheader\ntime\t"
        + "\t".join(columns)
        + "\n0\t"
        + "\t".join(map(str, values))
        + "\n"
    )
    return path


@pytest.mark.parametrize("identifiers", [["1"], ["1", "2", "3"], ["A", "B"], [""]])
def check_force_platform_count_and_values(tmp_path: Path, identifiers: list[str]) -> None:
    path = _write_forces(tmp_path / "forces.mot", identifiers)
    times, forces, points, torques = read_grf(path)
    assert forces.shape == points.shape == torques.shape == (1, len(identifiers), 3)
    np.testing.assert_allclose(forces[0, :, 1], np.arange(len(identifiers)) + 700)
    np.testing.assert_allclose(points[0, :, 0], np.arange(len(identifiers)) + 0.1)
    trial = _load_trial(grf_path=path)
    assert trial.grf_labels == [f"platform {name or 1}" for name in identifiers]


def check_named_force_sides_retain_right_left_order(tmp_path: Path) -> None:
    path = _write_forces(tmp_path / "forces.mot", ["l", "r"])
    trial = _load_trial(grf_path=path)
    assert trial.grf_labels == ["right", "left"]
    np.testing.assert_allclose(trial.grf_forces[0, :, 1], [701, 700])


def check_zero_force_cop_is_missing(tmp_path: Path) -> None:
    path = _write_forces(tmp_path / "forces.mot", ["1"])
    text = path.read_text().replace("10\t700\t30", "0\t0\t0")
    path.write_text(text)
    assert np.isnan(read_grf(path)[2]).all()


def check_unrecognised_forces_raise_instead_of_empty_overlay(tmp_path: Path) -> None:
    path = tmp_path / "forces.mot"
    path.write_text("endheader\ntime unrelated\n0 1\n")
    with pytest.raises(ValueError, match="No complete"):
        read_grf(path)


def check_geometry_finds_sibling_data_from_nested_output(tmp_path: Path, monkeypatch) -> None:
    model = tmp_path / "practical" / "output" / "scaled" / "model.osim"
    model.parent.mkdir(parents=True)
    mesh = tmp_path / "practical" / "data" / "Geometry" / "pelvis.vtp"
    mesh.parent.mkdir(parents=True)
    mesh.touch()
    monkeypatch.chdir(tmp_path.parent)
    assert _geometry_mesh_path(model, "pelvis.vtp") == mesh.resolve()
    assert _geometry_mesh_path(model, r"Geometry\pelvis.vtp") == mesh.resolve()


def check_geometry_prefers_model_local_mesh(tmp_path: Path) -> None:
    model = tmp_path / "output" / "model.osim"
    model.parent.mkdir()
    local = model.parent / "Geometry" / "pelvis.vtp"
    local.parent.mkdir()
    local.touch()
    other = tmp_path / "data" / "Geometry" / "pelvis.vtp"
    other.parent.mkdir(parents=True)
    other.touch()
    assert _geometry_mesh_path(model, "pelvis.vtp") == local.resolve()
    assert _geometry_mesh_path(model, str(other)) == other.resolve()


def check_additional_geometry_paths(tmp_path: Path) -> None:
    mesh = tmp_path / "shared meshes" / "pelvis.vtp"
    mesh.parent.mkdir()
    mesh.touch()
    assert (
        _geometry_mesh_path(
            tmp_path / "output" / "model.osim", "pelvis.vtp", geometry_paths=[mesh.parent]
        )
        == mesh.resolve()
    )


def check_c3d_rejects_model_geometry_options() -> None:
    with pytest.raises(ValueError, match="separate input route"):
        _load_trial(c3d_path="trial.c3d", geometry_paths=["meshes"])
