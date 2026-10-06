"""OpenSim trial loading without importing the optional GUI dependencies.

The default example is bundled under this package. The internal loader accepts
arbitrary model/trial paths supplied through the public visualiser function.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import re
from typing import Any, Iterable
import xml.etree.ElementTree as ET

import numpy as np


DEFAULT_BUNDLE = Path(__file__).resolve().parent / "example3DWalking"


@dataclass
class StorageTable:
    times: np.ndarray
    labels: list[str]
    values: np.ndarray


@dataclass
class GeometrySpec:
    name: str
    mesh_path: Path
    scale: np.ndarray
    transforms: np.ndarray
    color: tuple[float, float, float] = (0.75, 0.78, 0.84)
    opacity: float = 1.0


@dataclass
class OpenSimTrial:
    """All data needed by the visualiser, aligned to ``times``."""

    model_path: Path | None
    coordinate_path: Path | None
    marker_path: Path | None
    grf_path: Path | None
    activity_path: Path | None
    c3d_path: Path | None
    times: np.ndarray
    coordinate_labels: list[str]
    coordinates: np.ndarray
    measured_marker_labels: list[str] = field(default_factory=list)
    measured_markers: np.ndarray | None = None
    model_marker_labels: list[str] = field(default_factory=list)
    model_markers: np.ndarray | None = None
    grf_forces: np.ndarray | None = None
    grf_points: np.ndarray | None = None
    grf_torques: np.ndarray | None = None
    grf_labels: list[str] = field(default_factory=list)
    geometry: list[GeometrySpec] = field(default_factory=list)
    muscle_labels: list[str] = field(default_factory=list)
    muscle_paths: list[list[np.ndarray]] = field(default_factory=list)
    muscle_activity: np.ndarray | None = None
    opensim_available: bool = False
    opensim_error: str | None = None

    @property
    def frame_count(self) -> int:
        return int(self.times.size)

    @property
    def duration(self) -> float:
        return float(self.times[-1] - self.times[0]) if self.frame_count > 1 else 0.0


def _split_fields(line: str) -> list[str]:
    return line.strip().replace(",", " ").split()


def _find_endheader(lines: list[str]) -> int:
    for index, line in enumerate(lines):
        if line.strip().lower() == "endheader":
            return index
    raise ValueError("OpenSim storage file has no endheader line")


def read_storage(path: str | Path) -> StorageTable:
    """Read the tabular portion of an OpenSim ``.sto`` or ``.mot`` file."""

    path = Path(path)
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    endheader = _find_endheader(lines)
    header_index = endheader + 1
    while header_index < len(lines) and not lines[header_index].strip():
        header_index += 1
    if header_index >= len(lines):
        raise ValueError(f"No column header found in {path}")

    labels = _split_fields(lines[header_index])
    rows: list[list[float]] = []
    for line in lines[header_index + 1 :]:
        fields = _split_fields(line)
        if not fields:
            continue
        try:
            row = [float(value) for value in fields]
        except ValueError:
            continue
        if len(row) >= len(labels):
            rows.append(row[: len(labels)])
    if not rows:
        raise ValueError(f"No numeric data found in {path}")
    values = np.asarray(rows, dtype=float)
    return StorageTable(times=values[:, 0], labels=labels[1:], values=values[:, 1:])


def _model_default_coordinate_table(model_path: Path) -> StorageTable:
    """Build a one-frame coordinate table from defaults stored in a model."""

    labels: list[str] = []
    defaults: list[float] = []
    try:  # Prefer OpenSim so motion types and component paths are exact.
        import opensim as osim  # type: ignore

        model = osim.Model(str(model_path))
        model.initSystem()
        coordinates = model.getCoordinateSet()
        for index in range(coordinates.getSize()):
            coordinate = coordinates.get(index)
            try:
                label = f"{coordinate.getAbsolutePathString()}/value"
            except Exception:
                label = str(coordinate.getName())
            value = float(coordinate.getDefaultValue())
            try:
                # OpenSim stores rotational defaults in radians while its
                # kinematics storage files conventionally use degrees.
                if int(coordinate.getMotionType()) != 2:
                    value = float(np.rad2deg(value))
            except Exception:
                pass
            labels.append(str(label))
            defaults.append(value)
    except Exception:
        # Basic fallback for environments without the OpenSim API. This still
        # exposes coordinate names/defaults, although model geometry itself
        # also requires OpenSim and therefore cannot be evaluated there.
        try:
            root = ET.parse(model_path).getroot()
            for element in root.iter():
                if element.tag.rsplit("}", 1)[-1] != "Coordinate":
                    continue
                name = element.attrib.get("name")
                if not name:
                    continue
                value = 0.0
                for child in element:
                    if child.tag.rsplit("}", 1)[-1] == "default_value" and child.text:
                        value = float(child.text)
                        break
                labels.append(name)
                defaults.append(value)
        except Exception:
            labels = []
            defaults = []

    values = np.asarray(defaults, dtype=float).reshape(1, -1)
    return StorageTable(times=np.array([0.0], dtype=float), labels=labels, values=values)


def read_trc(path: str | Path) -> tuple[np.ndarray, list[str], np.ndarray]:
    """Read a TRC file and return time, marker names, and positions in metres."""

    path = Path(path)
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    frame_index = next(
        (i for i, line in enumerate(lines) if line.strip().lower().startswith("frame#")),
        None,
    )
    if frame_index is None:
        raise ValueError(f"No Frame# header found in {path}")

    labels = _split_fields(lines[frame_index])
    # TRC stores the marker names once in the first header row; the X/Y/Z
    # sublabels are on the following row and are not part of ``labels``.
    marker_names = labels[2:]
    rows: list[list[float]] = []
    for line in lines[frame_index + 2 :]:
        fields = _split_fields(line)
        if len(fields) < 2 + 3 * len(marker_names):
            continue
        try:
            rows.append([float(value) for value in fields[: 2 + 3 * len(marker_names)]])
        except ValueError:
            continue
    if not rows:
        raise ValueError(f"No marker data found in {path}")

    values = np.asarray(rows, dtype=float)
    positions = values[:, 2:].reshape(len(rows), len(marker_names), 3)
    units_line = next((line.lower() for line in lines[:frame_index] if "units" in line.lower()), "")
    # The bundled example is in millimetres.  Treat centimetres similarly;
    # metre-valued TRCs are left unchanged.
    if "mm" in units_line:
        positions /= 1000.0
    elif re.search(r"\bcm\b", units_line):
        positions /= 100.0
    return values[:, 1], marker_names, positions


def _column_group(labels: Iterable[str], prefix: str, suffixes: tuple[str, str, str]) -> list[int] | None:
    labels = list(labels)
    indexes: list[int] = []
    for suffix in suffixes:
        candidates = [f"{prefix}{suffix}", f"{prefix}_{suffix}"]
        match = next((labels.index(candidate) for candidate in candidates if candidate in labels), None)
        if match is None:
            return None
        indexes.append(match)
    return indexes


def read_grf(path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Read forces, CoP/point-of-application, and torques from an OpenSim MOT."""

    table = read_storage(path)
    forces = np.full((table.times.size, 2, 3), np.nan, dtype=float)
    points = np.full_like(forces, np.nan)
    torques = np.full_like(forces, np.nan)
    for side_index, side in enumerate(("ground_force_r_", "ground_force_l_")):
        force_columns = _column_group(table.labels, side, ("vx", "vy", "vz"))
        point_columns = _column_group(table.labels, side, ("px", "py", "pz"))
        torque_columns = _column_group(table.labels, f"ground_torque_{side[-2]}_", ("x", "y", "z"))
        if force_columns:
            forces[:, side_index, :] = table.values[:, force_columns]
        if point_columns:
            points[:, side_index, :] = table.values[:, point_columns]
        if torque_columns:
            torques[:, side_index, :] = table.values[:, torque_columns]
    # Active point-of-application values in this OpenSim MOT are metres. The
    # very large values present in zero-force rows are invalid placeholders,
    # so discard those rows instead of using them to infer a unit conversion.
    active = np.linalg.norm(forces, axis=2) > 1.0
    points[~active] = np.nan
    return table.times, forces, points, torques


def _interp_array(source_times: np.ndarray, source: np.ndarray, target_times: np.ndarray) -> np.ndarray:
    result = np.full(target_times.shape + source.shape[1:], np.nan, dtype=float)
    inside = (target_times >= source_times[0]) & (target_times <= source_times[-1])
    if not np.any(inside):
        return result
    for index in np.ndindex(source.shape[1:]):
        series = source[(slice(None),) + index]
        valid = np.isfinite(series)
        if not np.any(valid):
            continue
        result[(inside,) + index] = np.interp(
            target_times[inside], source_times[valid], series[valid]
        )
    return result


def _as_float_array(vec: Any) -> np.ndarray:
    return np.asarray([float(vec[i]) for i in range(3)], dtype=float)


def _geometry_mesh_path(model_path: Path, mesh_file: str) -> Path:
    candidate = Path(mesh_file)
    if candidate.is_absolute() and candidate.exists():
        return candidate
    candidates = [model_path.parent / candidate, model_path.parent / "Geometry" / candidate.name]
    for item in candidates:
        if item.exists():
            return item
    return candidates[-1]


def _geometry_xml(geometry: Any) -> ET.Element | None:
    try:
        dump = geometry.dump()
        return ET.fromstring(dump)
    except Exception:
        return None


def _load_opensim_content(
    model_path: Path,
    coordinate_labels: list[str],
    coordinate_values: np.ndarray,
    times: np.ndarray,
) -> tuple[
    list[str],
    np.ndarray | None,
    list[GeometrySpec],
    list[str],
    list[list[np.ndarray]],
    str | None,
]:
    """Evaluate OpenSim markers, meshes, and muscle paths for each frame."""

    try:
        import opensim as osim  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on local OpenSim installation
        return [], None, [], [], [], f"OpenSim Python API unavailable: {exc}"

    try:  # pragma: no cover - the API is supplied by the user's OpenSim install
        model = osim.Model(str(model_path))
        state = model.initSystem()
        coordinates = model.getCoordinateSet()
        coordinate_objects = [coordinates.get(i) for i in range(coordinates.getSize())]
        label_to_object: dict[str, Any] = {}
        for coordinate in coordinate_objects:
            label_to_object[str(coordinate.getName())] = coordinate
            try:
                label_to_object[str(coordinate.getAbsolutePathString())] = coordinate
            except Exception:
                pass

        # Coordinates.sto labels are normally absolute paths ending in /value.
        resolved_coordinates: list[Any | None] = []
        for label in coordinate_labels:
            key = label[:-6] if label.endswith("/value") else label
            obj = label_to_object.get(key) or label_to_object.get(key.rsplit("/", 1)[-1])
            resolved_coordinates.append(obj)

        marker_set = model.getMarkerSet()
        marker_labels = [marker_set.get(i).getName() for i in range(marker_set.getSize())]
        marker_objects = [marker_set.get(i) for i in range(marker_set.getSize())]
        model_markers = np.full((times.size, len(marker_objects), 3), np.nan, dtype=float)
        muscle_set = model.getMuscles()
        muscle_labels = [muscle_set.get(i).getName() for i in range(muscle_set.getSize())]
        muscle_objects = [muscle_set.get(i) for i in range(muscle_set.getSize())]
        muscle_paths: list[list[np.ndarray]] = [[] for _ in muscle_objects]
        geometry_records: list[tuple[Any, Path, np.ndarray, tuple[float, float, float], float]] = []

        body_set = model.getBodySet()
        for body_index in range(body_set.getSize()):
            body = body_set.get(body_index)
            geometry_index = 0
            while True:
                try:
                    geometry = body.get_attached_geometry(geometry_index)
                except Exception:
                    break
                geometry_index += 1
                xml = _geometry_xml(geometry)
                if xml is None:
                    continue
                mesh_element = xml.find(".//mesh_file")
                if mesh_element is None or not mesh_element.text:
                    continue
                scale_element = xml.find(".//scale_factors")
                scale = np.ones(3, dtype=float)
                if scale_element is not None and scale_element.text:
                    values = [float(item) for item in scale_element.text.split()]
                    if len(values) >= 3:
                        scale = np.asarray(values[:3], dtype=float)
                try:
                    color_vec = _as_float_array(geometry.getColor())
                    color = tuple(np.clip(color_vec, 0.0, 1.0).tolist())
                except Exception:
                    color = (0.75, 0.78, 0.84)
                try:
                    opacity = float(geometry.getOpacity())
                except Exception:
                    opacity = 1.0
                geometry_records.append(
                    (geometry, _geometry_mesh_path(model_path, mesh_element.text.strip()), scale, color, opacity)
                )

        transforms_by_geometry = [
            np.repeat(np.eye(4, dtype=float)[None, :, :], times.size, axis=0)
            for _record in geometry_records
        ]
        for frame_index, time in enumerate(times):
            state.setTime(float(time))
            for coordinate, value in zip(resolved_coordinates, coordinate_values[frame_index]):
                if coordinate is None or not np.isfinite(value):
                    continue
                motion_type = int(coordinate.getMotionType())
                # coordinates.sto declares inDegrees=yes; translations stay metres.
                value_for_model = np.deg2rad(value) if motion_type != 2 else value
                coordinate.setValue(state, float(value_for_model), False)
            model.realizePosition(state)
            for marker_index, marker in enumerate(marker_objects):
                model_markers[frame_index, marker_index, :] = _as_float_array(marker.getLocationInGround(state))
            for muscle_index, muscle in enumerate(muscle_objects):
                current_path = muscle.getGeometryPath().getCurrentPath(state)
                points = np.asarray(
                    [
                        _as_float_array(current_path.get(point_index).getLocationInGround(state))
                        for point_index in range(current_path.getSize())
                    ],
                    dtype=float,
                ).reshape(-1, 3)
                muscle_paths[muscle_index].append(points)
            for geometry_index, (geometry, _mesh_path, _scale, _color, _opacity) in enumerate(geometry_records):
                transform = geometry.getFrame().getTransformInGround(state)
                rotation = transform.R()
                transforms_by_geometry[geometry_index][frame_index, :3, :3] = [
                    [rotation.get(i, j) for j in range(3)] for i in range(3)
                ]
                transforms_by_geometry[geometry_index][frame_index, :3, 3] = _as_float_array(transform.p())

        geometry_specs: list[GeometrySpec] = []
        for record, transforms in zip(geometry_records, transforms_by_geometry):
            geometry, mesh_path, scale, color, opacity = record
            geometry_specs.append(
                GeometrySpec(
                    name=str(geometry.getName()),
                    mesh_path=mesh_path,
                    scale=scale,
                    transforms=transforms,
                    color=color,
                    opacity=opacity,
                )
            )
        return marker_labels, model_markers, geometry_specs, muscle_labels, muscle_paths, None
    except Exception as exc:
        return [], None, [], [], [], f"OpenSim model evaluation failed: {exc}"


def _activity_for_muscles(
    activity_path: Path,
    times: np.ndarray,
    muscle_labels: list[str],
) -> np.ndarray:
    """Map direct activation columns or grouped EMG channels onto muscles."""

    table = read_storage(activity_path)
    aligned = _interp_array(table.times, table.values, times)

    def canonical(label: str) -> str:
        parts = [part for part in label.strip("/").split("/") if part]
        if parts and parts[-1].lower() in {"activation", "value"}:
            parts = parts[:-1]
        return parts[-1].lower() if parts else label.lower()

    columns = {canonical(label): index for index, label in enumerate(table.labels)}
    grouped_channels = {
        "soleus": "soleus",
        "gaslat": "gastrocnemius",
        "gasmed": "gastrocnemius",
        "tibant": "tibialis_anterior",
        "semimem": "medial_hamstrings",
        "semiten": "medial_hamstrings",
        "bflh": "biceps_femoris",
        "bfsh": "biceps_femoris",
        "vaslat": "vastus_lateralis",
        "vasmed": "vastus_medius",
        "vasint": "vastus_medius",
        "recfem": "rectus_femoris",
        "glmax1": "gluteus_maximus",
        "glmax2": "gluteus_maximus",
        "glmax3": "gluteus_maximus",
        "glmed1": "gluteus_medius",
        "glmed2": "gluteus_medius",
        "glmed3": "gluteus_medius",
    }
    result = np.full((times.size, len(muscle_labels)), np.nan, dtype=float)
    for muscle_index, muscle_name in enumerate(muscle_labels):
        lower_name = muscle_name.lower()
        direct_index = columns.get(lower_name)
        base_name = lower_name.rsplit("_", 1)[0]
        channel_name = grouped_channels.get(base_name)
        column_index = direct_index if direct_index is not None else columns.get(channel_name or "")
        if column_index is not None:
            result[:, muscle_index] = np.clip(aligned[:, column_index], 0.0, 1.0)
    return result


def _c3d_scalar(value: Any, default: float = 0.0) -> float:
    """Return the first numeric value from an ezc3d field."""

    if value is None:
        return default
    values = np.asarray(value).reshape(-1)
    return float(values[0]) if values.size else default


def _c3d_text(value: Any, default: str = "") -> str:
    """Return the first text value from an ezc3d field."""

    if value is None:
        return default
    values = np.asarray(value, dtype=object).reshape(-1)
    return str(values[0]).strip() if values.size else default


def _c3d_parameter(c3d: Any, group: str, name: str, default: Any = None) -> Any:
    try:
        return c3d["parameters"][group][name]["value"]
    except (KeyError, TypeError):
        return default


def _c3d_rate(c3d: Any, section: str, parameter_group: str) -> float:
    try:
        header_rate = _c3d_scalar(c3d["header"][section].get("frame_rate"))
    except (KeyError, TypeError, AttributeError):
        header_rate = 0.0
    if header_rate > 0.0:
        return header_rate
    return _c3d_scalar(_c3d_parameter(c3d, parameter_group, "RATE"))


def _c3d_length_scale(unit: str) -> float:
    normalized = unit.strip().lower().replace(" ", "")
    return {
        "m": 1.0,
        "meter": 1.0,
        "metre": 1.0,
        "cm": 0.01,
        "centimeter": 0.01,
        "centimetre": 0.01,
        "mm": 0.001,
        "millimeter": 0.001,
        "millimetre": 0.001,
    }.get(normalized, 1.0)


def _c3d_force_scale(unit: str) -> float:
    normalized = unit.strip().lower().replace(" ", "")
    return {"n": 1.0, "kn": 1000.0}.get(normalized, 1.0)


def _c3d_moment_scale(unit: str) -> float:
    normalized = unit.strip().lower().replace(" ", "").replace("*", "")
    return {"nm": 1.0, "ncm": 0.01, "nmm": 0.001, "knm": 1000.0}.get(
        normalized, 1.0
    )


def _c3d_vector_series(value: Any, name: str) -> np.ndarray:
    """Return a C3D vector series as samples-by-XYZ."""

    array = np.asarray(value, dtype=float).squeeze()
    if array.ndim == 1 and array.size == 3:
        return array.reshape(1, 3)
    if array.ndim != 2:
        raise ValueError(f"C3D {name} data must be a two-dimensional vector series")
    if array.shape[0] == 3:
        return array.T
    if array.shape[1] == 3:
        return array
    raise ValueError(f"C3D {name} data must have three vector components")


def _c3d_z_up_to_y_up(values: np.ndarray) -> np.ndarray:
    """Rotate C3D's conventional Z-up coordinates into the viewer's Y-up frame."""

    return np.stack((values[..., 0], values[..., 2], -values[..., 1]), axis=-1)


def _load_c3d_trial(c3d_path: Path) -> OpenSimTrial:
    """Load markers and force-platform data from one standalone C3D file."""

    try:
        import ezc3d  # type: ignore
    except Exception as exc:  # pragma: no cover - dependency supplied by the environment
        raise RuntimeError("C3D loading requires the ezc3d package") from exc

    try:
        c3d = ezc3d.c3d(str(c3d_path), extract_forceplat_data=True)
    except Exception as exc:
        raise ValueError(f"Could not load C3D file {c3d_path}: {exc}") from exc

    data = c3d["data"]
    raw_points = np.asarray(data.get("points", np.empty((4, 0, 0))), dtype=float)
    point_unit = _c3d_text(_c3d_parameter(c3d, "POINT", "UNITS"), "m")
    marker_times: np.ndarray | None = None
    marker_labels: list[str] = []
    measured_markers: np.ndarray | None = None
    point_rate = _c3d_rate(c3d, "points", "POINT")
    point_start = 0.0
    if point_rate > 0.0:
        try:
            first_frame = _c3d_scalar(c3d["header"]["points"].get("first_frame"))
        except (KeyError, TypeError, AttributeError):
            first_frame = 0.0
        point_start = first_frame / point_rate

    if (
        raw_points.ndim == 3
        and raw_points.shape[0] >= 3
        and raw_points.shape[1] > 0
        and raw_points.shape[2] > 0
    ):
        if point_rate <= 0.0:
            raise ValueError("C3D marker data has no valid POINT:RATE")
        marker_count = raw_points.shape[1]
        frame_count = raw_points.shape[2]
        positions = raw_points[:3, :, :].transpose(2, 1, 0)
        positions *= _c3d_length_scale(point_unit)
        measured_markers = _c3d_z_up_to_y_up(positions)
        marker_times = point_start + np.arange(frame_count, dtype=float) / point_rate

        labels = list(_c3d_parameter(c3d, "POINT", "LABELS", []))
        marker_labels = [
            str(labels[index]).strip() if index < len(labels) and str(labels[index]).strip()
            else f"marker_{index + 1}"
            for index in range(marker_count)
        ]

        residuals = np.asarray(
            data.get("meta_points", {}).get("residuals", np.empty((0, 0, 0))),
            dtype=float,
        )
        if residuals.ndim == 3 and residuals.shape[1:] == (marker_count, frame_count):
            measured_markers[residuals[0].T < 0.0] = np.nan

    platforms = list(data.get("platform", []))
    platform_series: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for index, platform in enumerate(platforms):
        force = _c3d_vector_series(platform["force"], f"platform {index + 1} force")
        force *= _c3d_force_scale(_c3d_text(platform.get("unit_force"), "N"))
        force = _c3d_z_up_to_y_up(force)

        if "center_of_pressure" in platform:
            point = _c3d_vector_series(
                platform["center_of_pressure"], f"platform {index + 1} center of pressure"
            )
            point *= _c3d_length_scale(
                _c3d_text(platform.get("unit_position"), point_unit)
            )
            point = _c3d_z_up_to_y_up(point)
        else:
            point = np.full_like(force, np.nan)

        if "moment" in platform:
            torque = _c3d_vector_series(platform["moment"], f"platform {index + 1} moment")
            torque *= _c3d_moment_scale(_c3d_text(platform.get("unit_moment"), "Nm"))
            torque = _c3d_z_up_to_y_up(torque)
        else:
            torque = np.full_like(force, np.nan)
        platform_series.append((force, point, torque))

    force_times: np.ndarray | None = None
    raw_forces = raw_points_of_application = raw_torques = None
    if platform_series:
        analog_rate = _c3d_rate(c3d, "analogs", "ANALOG")
        if analog_rate <= 0.0:
            raise ValueError("C3D force-platform data has no valid ANALOG:RATE")
        force_start = point_start
        if marker_times is None:
            try:
                first_analog_frame = _c3d_scalar(
                    c3d["header"]["analogs"].get("first_frame")
                )
            except (KeyError, TypeError, AttributeError):
                first_analog_frame = 0.0
            force_start = first_analog_frame / analog_rate
        force_frame_count = max(series[0].shape[0] for series in platform_series)
        force_times = force_start + np.arange(force_frame_count, dtype=float) / analog_rate
        platform_count = len(platform_series)
        shape = (force_frame_count, platform_count, 3)
        raw_forces = np.full(shape, np.nan, dtype=float)
        raw_points_of_application = np.full(shape, np.nan, dtype=float)
        raw_torques = np.full(shape, np.nan, dtype=float)
        for index, (force, point, torque) in enumerate(platform_series):
            local_times = force_start + np.arange(force.shape[0], dtype=float) / analog_rate
            raw_forces[:, index, :] = _interp_array(local_times, force, force_times)
            if point.shape[0] == force.shape[0]:
                raw_points_of_application[:, index, :] = _interp_array(
                    local_times, point, force_times
                )
            if torque.shape[0] == force.shape[0]:
                raw_torques[:, index, :] = _interp_array(local_times, torque, force_times)

    if marker_times is None and force_times is None:
        raise ValueError("C3D file contains neither marker nor force-platform data")

    times = marker_times if marker_times is not None else force_times
    assert times is not None
    grf_forces = grf_points = grf_torques = None
    if force_times is not None:
        assert raw_forces is not None
        assert raw_points_of_application is not None
        assert raw_torques is not None
        grf_forces = _interp_array(force_times, raw_forces, times)
        grf_points = _interp_array(force_times, raw_points_of_application, times)
        grf_torques = _interp_array(force_times, raw_torques, times)
        grf_points[np.linalg.norm(grf_forces, axis=2) <= 1.0] = np.nan

    return OpenSimTrial(
        model_path=None,
        coordinate_path=None,
        marker_path=None,
        grf_path=None,
        activity_path=None,
        c3d_path=c3d_path,
        times=times,
        coordinate_labels=[],
        coordinates=np.empty((times.size, 0), dtype=float),
        measured_marker_labels=marker_labels,
        measured_markers=measured_markers,
        grf_forces=grf_forces,
        grf_points=grf_points,
        grf_torques=grf_torques,
        grf_labels=[f"platform {index + 1}" for index in range(len(platform_series))],
    )


def _load_trial(
    model_path: str | Path | None = None,
    coordinate_path: str | Path | None = None,
    marker_path: str | Path | None = None,
    grf_path: str | Path | None = None,
    activity_path: str | Path | None = None,
    c3d_path: str | Path | None = None,
) -> OpenSimTrial:
    """Load any supplied OpenSim model or trial files onto one timeline."""

    c3d_path = Path(c3d_path).expanduser() if c3d_path else None
    if c3d_path is not None:
        if any((model_path, coordinate_path, marker_path, grf_path, activity_path)):
            raise ValueError(
                "c3d_path is a separate input route and cannot be combined with "
                "model or trial files"
            )
        return _load_c3d_trial(c3d_path)

    model_path = Path(model_path).expanduser() if model_path else None
    coordinate_path = Path(coordinate_path).expanduser() if coordinate_path else None
    marker_path = Path(marker_path).expanduser() if marker_path else None
    grf_path = Path(grf_path).expanduser() if grf_path else None
    activity_path = Path(activity_path).expanduser() if activity_path else None

    measured_labels: list[str] = []
    marker_data: tuple[np.ndarray, list[str], np.ndarray] | None = None
    if marker_path:
        marker_data = read_trc(marker_path)
        measured_labels = marker_data[1]

    grf_data: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None = None
    if grf_path:
        grf_data = read_grf(grf_path)

    if coordinate_path is not None:
        coordinate_table = read_storage(coordinate_path)
        times = coordinate_table.times
    elif marker_data is not None:
        times = marker_data[0]
        coordinate_table = None
    elif grf_data is not None:
        times = grf_data[0]
        coordinate_table = None
    elif model_path is not None:
        coordinate_table = _model_default_coordinate_table(model_path)
        times = coordinate_table.times
    else:
        raise ValueError(
            "Provide at least one model, coordinate, marker, or ground-reaction-force file"
        )

    if coordinate_table is None:
        if model_path is not None:
            defaults = _model_default_coordinate_table(model_path)
            coordinate_labels = defaults.labels
            coordinates = np.repeat(defaults.values, times.size, axis=0)
        else:
            coordinate_labels = []
            coordinates = np.empty((times.size, 0), dtype=float)
    else:
        coordinate_labels = coordinate_table.labels
        coordinates = coordinate_table.values

    measured_markers: np.ndarray | None = None
    if marker_data is not None:
        marker_times, _marker_labels, marker_values = marker_data
        measured_markers = _interp_array(marker_times, marker_values, times)

    grf_forces = grf_points = grf_torques = None
    if grf_data is not None:
        grf_times, raw_forces, raw_points, raw_torques = grf_data
        grf_forces = _interp_array(grf_times, raw_forces, times)
        grf_points = _interp_array(grf_times, raw_points, times)
        grf_torques = _interp_array(grf_times, raw_torques, times)

    model_labels: list[str] = []
    model_markers: np.ndarray | None = None
    geometry: list[GeometrySpec] = []
    muscle_labels: list[str] = []
    muscle_paths: list[list[np.ndarray]] = []
    opensim_error: str | None = None
    if model_path is not None:
        (
            model_labels,
            model_markers,
            geometry,
            muscle_labels,
            muscle_paths,
            opensim_error,
        ) = _load_opensim_content(model_path, coordinate_labels, coordinates, times)

    muscle_activity = None
    if activity_path and muscle_labels:
        muscle_activity = _activity_for_muscles(activity_path, times, muscle_labels)
    return OpenSimTrial(
        model_path=model_path,
        coordinate_path=coordinate_path,
        marker_path=marker_path,
        grf_path=grf_path,
        activity_path=activity_path,
        c3d_path=None,
        times=times,
        coordinate_labels=coordinate_labels,
        coordinates=coordinates,
        measured_marker_labels=measured_labels,
        measured_markers=measured_markers,
        model_marker_labels=model_labels,
        model_markers=model_markers,
        grf_forces=grf_forces,
        grf_points=grf_points,
        grf_torques=grf_torques,
        grf_labels=["right", "left"] if grf_forces is not None else [],
        geometry=geometry,
        muscle_labels=muscle_labels,
        muscle_paths=muscle_paths,
        muscle_activity=muscle_activity,
        opensim_available=model_path is not None and opensim_error is None,
        opensim_error=opensim_error,
    )


def _load_default_trial(bundle: str | Path = DEFAULT_BUNDLE) -> OpenSimTrial:
    """Load the complete OpenSim 4.5 example3DWalking bundle."""

    bundle = Path(bundle).expanduser()
    return _load_trial(
        model_path=bundle / "subject_walk_scaled.osim",
        coordinate_path=bundle / "coordinates.sto",
        marker_path=bundle / "marker_trajectories.trc",
        grf_path=bundle / "grf_walk.mot",
        activity_path=bundle / "electromyography.sto",
    )
