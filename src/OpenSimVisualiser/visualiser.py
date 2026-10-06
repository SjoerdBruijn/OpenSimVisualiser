"""Native Qt/PyVista visualiser for :mod:`OpenSimVisualiser`."""

from __future__ import annotations

from pathlib import Path
import time
from typing import Any

import numpy as np

from .data import DEFAULT_BUNDLE, OpenSimTrial, _load_default_trial, _load_trial


def _gui_imports() -> tuple[Any, Any, Any, Any]:
    try:
        from PyQt5 import QtCore, QtWidgets  # type: ignore
        import pyqtgraph as pg  # type: ignore
        import pyvista as pv  # type: ignore
        from pyvistaqt import QtInteractor  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on the selected GUI environment
        raise RuntimeError(
            "The OpenSim visualiser needs PyQt5, pyqtgraph, pyvista, and pyvistaqt. "
            "Install OpenSimVisualiser in the environment used to launch it."
        ) from exc
    return QtCore, QtWidgets, pg, (pv, QtInteractor)


def _finite_points(points: np.ndarray | None, frame: int) -> np.ndarray:
    if points is None or points.size == 0:
        return np.zeros((0, 3), dtype=float)
    values = np.asarray(points[frame], dtype=float).copy()
    values[~np.isfinite(values)] = 0.0
    return values


def _transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    homogeneous = np.ones((points.shape[0], 4), dtype=float)
    homogeneous[:, :3] = points
    return (homogeneous @ transform.T)[:, :3]


def _activity_color(value: float) -> np.ndarray:
    """Return an RGB colour for normalized muscle activity."""

    if not np.isfinite(value):
        return np.array([145, 145, 145], dtype=np.uint8)
    value = float(np.clip(value, 0.0, 1.0))
    low = np.array([40.0, 95.0, 210.0])
    middle = np.array([250.0, 205.0, 45.0])
    high = np.array([220.0, 45.0, 35.0])
    if value <= 0.5:
        color = low + (middle - low) * (value / 0.5)
    else:
        color = middle + (high - middle) * ((value - 0.5) / 0.5)
    return np.asarray(np.round(color), dtype=np.uint8)


class OpenSimVisualizerWindow:
    """Interactive window with playback and independently switchable layers."""

    def __init__(self, trial: OpenSimTrial, *, title: str = "OpenSim Python Visualiser") -> None:
        QtCore, QtWidgets, pg, pyvista_parts = _gui_imports()
        self.QtCore = QtCore
        self.QtWidgets = QtWidgets
        self.pg = pg
        self.pv, self.QtInteractor = pyvista_parts
        self.trial = trial
        self.frame = 0
        self.speed = 1.0
        self._play_wall_start: float | None = None
        self._play_sim_start: float | None = None
        self._advancing_frame = False
        self._chart_cursor = None
        self._replacement_window: OpenSimVisualizerWindow | None = None
        self._geometry_entries: list[dict[str, Any]] = []
        self._muscle_poly = None
        self._muscle_actor = None
        self._timer = QtCore.QTimer()
        self._timer.timeout.connect(self._advance_frame)

        self.window = QtWidgets.QMainWindow()
        self.window.setWindowTitle(title)
        self.window.resize(1500, 920)
        self._build_ui()
        self._build_scene()
        self._set_frame(0)
        self._refresh_chart()

    def _build_ui(self) -> None:
        QtWidgets = self.QtWidgets
        central = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(central)
        layout.setContentsMargins(6, 6, 6, 6)
        self.window.setCentralWidget(central)

        controls = QtWidgets.QWidget()
        controls.setMinimumWidth(260)
        controls_layout = QtWidgets.QVBoxLayout(controls)
        layout.addWidget(controls)

        title = QtWidgets.QLabel("OpenSim visualiser")
        title.setStyleSheet("font-size: 18px; font-weight: bold;")
        controls_layout.addWidget(title)
        source_names = []
        if self.trial.c3d_path is not None:
            source_names.append(f"C3D: {self.trial.c3d_path.name}")
        if self.trial.model_path is not None:
            source_names.append(f"Model: {self.trial.model_path.name}")
        if self.trial.marker_path is not None:
            source_names.append(f"Markers: {self.trial.marker_path.name}")
        if self.trial.grf_path is not None:
            source_names.append(f"GRF: {self.trial.grf_path.name}")
        if self.trial.coordinate_path is not None:
            source_names.append(f"Coordinates: {self.trial.coordinate_path.name}")
        source = QtWidgets.QLabel("\n".join(source_names))
        source.setWordWrap(True)
        controls_layout.addWidget(source)

        controls_layout.addWidget(QtWidgets.QLabel("Data"))
        load_default_button = QtWidgets.QPushButton("Load bundled example")
        load_default_button.clicked.connect(self._load_bundled_example)
        controls_layout.addWidget(load_default_button)
        load_folder_button = QtWidgets.QPushButton("Load trial folder…")
        load_folder_button.clicked.connect(self._load_trial_folder)
        controls_layout.addWidget(load_folder_button)
        load_c3d_button = QtWidgets.QPushButton("Load C3D file…")
        load_c3d_button.clicked.connect(self._load_c3d_file)
        controls_layout.addWidget(load_c3d_button)

        controls_layout.addWidget(QtWidgets.QLabel("Layers"))
        self.layer_checks: dict[str, Any] = {}
        layer_definitions = [
            ("ground", "Ground plane", True),
            ("geometry", "Model geometry", bool(self.trial.geometry)),
            ("measured", "Measured markers", self.trial.measured_markers is not None),
            ("model_markers", "Model markers", self.trial.model_markers is not None),
            ("model_labels", "Model marker names", self.trial.model_markers is not None),
            ("muscles", "Muscles (activity colour)", bool(self.trial.muscle_paths)),
            ("grf", "Ground reaction forces", self.trial.grf_forces is not None),
        ]
        for key, label, enabled in layer_definitions:
            check = QtWidgets.QCheckBox(label)
            check.setChecked(enabled)
            check.stateChanged.connect(self._update_visibility)
            self.layer_checks[key] = check
            controls_layout.addWidget(check)
        muscle_legend = QtWidgets.QLabel(
            "Muscle activity:  <span style='color:#285fd2'>low</span> → "
            "<span style='color:#dcae16'>medium</span> → "
            "<span style='color:#dc2d23'>high</span><br>Grey = no matching activity channel"
        )
        muscle_legend.setStyleSheet("font-size: 10px; color: #666;")
        controls_layout.addWidget(muscle_legend)
        controls_layout.addWidget(QtWidgets.QLabel("Playback"))
        playback = QtWidgets.QHBoxLayout()
        self.play_button = QtWidgets.QPushButton("Play")
        self.play_button.clicked.connect(self.play)
        self.pause_button = QtWidgets.QPushButton("Pause")
        self.pause_button.clicked.connect(self.pause)
        self.stop_button = QtWidgets.QPushButton("Reset")
        self.stop_button.clicked.connect(lambda: self._set_frame(0))
        playback.addWidget(self.play_button)
        playback.addWidget(self.pause_button)
        playback.addWidget(self.stop_button)
        controls_layout.addLayout(playback)

        speed_row = QtWidgets.QHBoxLayout()
        speed_row.addWidget(QtWidgets.QLabel("Speed"))
        self.speed_combo = QtWidgets.QComboBox()
        for value in (0.25, 0.5, 1.0, 2.0, 4.0):
            self.speed_combo.addItem(f"{value:g}×", value)
        self.speed_combo.setCurrentIndex(2)
        self.speed_combo.currentIndexChanged.connect(self._set_speed)
        speed_row.addWidget(self.speed_combo)
        controls_layout.addLayout(speed_row)

        self.frame_label = QtWidgets.QLabel()
        controls_layout.addWidget(self.frame_label)
        self.frame_slider = QtWidgets.QSlider(self.QtCore.Qt.Horizontal)
        self.frame_slider.setRange(0, max(0, self.trial.frame_count - 1))
        self.frame_slider.valueChanged.connect(self._set_frame)
        controls_layout.addWidget(self.frame_slider)

        controls_layout.addWidget(QtWidgets.QLabel("Chart"))
        self.series_combo = QtWidgets.QComboBox()
        self._populate_series_combo()
        self.series_combo.currentIndexChanged.connect(self._refresh_chart)
        controls_layout.addWidget(self.series_combo)
        self.axis_checks: dict[str, Any] = {}
        axis_row = QtWidgets.QHBoxLayout()
        for axis in "XYZ":
            check = QtWidgets.QCheckBox(axis)
            check.setChecked(True)
            check.stateChanged.connect(self._refresh_chart)
            self.axis_checks[axis] = check
            axis_row.addWidget(check)
        controls_layout.addLayout(axis_row)

        self.status_label = QtWidgets.QLabel()
        self.status_label.setWordWrap(True)
        self.status_label.setStyleSheet("color: #555;")
        controls_layout.addWidget(self.status_label)
        controls_layout.addStretch(1)

        right = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(right, 1)
        self.plotter = self.QtInteractor(right)
        right_layout.addWidget(self.plotter, 4)

        self.chart = self.pg.PlotWidget()
        self.chart.setLabel("bottom", "Time", units="s")
        self.chart.setLabel("left", "Value")
        self.chart.showGrid(x=True, y=True, alpha=0.25)
        right_layout.addWidget(self.chart, 1)

        view_row = QtWidgets.QHBoxLayout()
        for label, view in (("XY", "xy"), ("XZ", "xz"), ("YZ", "yz"), ("3D", "iso")):
            button = QtWidgets.QPushButton(label)
            button.clicked.connect(lambda _checked=False, value=view: self._set_view(value))
            view_row.addWidget(button)
        right_layout.addLayout(view_row)
        camera_help = QtWidgets.QLabel(
            "Left drag: orbit (fixed up)   •   Shift+drag or middle drag: pan   •   Wheel: zoom"
        )
        camera_help.setAlignment(self.QtCore.Qt.AlignCenter)
        camera_help.setStyleSheet("color: #666; font-size: 11px;")
        right_layout.addWidget(camera_help)

        if self.trial.opensim_error:
            self.status_label.setText(self.trial.opensim_error)
        elif self.trial.model_path is None:
            grf_status = "GRFs loaded" if self.trial.grf_forces is not None else "no GRFs"
            self.status_label.setText(
                f"{len(self.trial.measured_marker_labels)} measured markers, {grf_status}; "
                "no model loaded"
            )
        else:
            mapped_muscles = 0
            if self.trial.muscle_activity is not None:
                mapped_muscles = int(np.any(np.isfinite(self.trial.muscle_activity), axis=0).sum())
            self.status_label.setText(
                f"{len(self.trial.geometry)} mesh objects, "
                f"{len(self.trial.model_marker_labels)} model markers, "
                f"{len(self.trial.muscle_labels)} muscles "
                f"({mapped_muscles} activity-mapped) loaded"
            )

    def _populate_series_combo(self) -> None:
        QtWidgets = self.QtWidgets
        if self.trial.coordinate_labels:
            for index, label in enumerate(self.trial.coordinate_labels):
                parts = [part for part in label.strip("/").split("/") if part]
                if parts and parts[-1] == "value":
                    parts = parts[:-1]
                display_name = "/".join(parts[-2:]) if len(parts) >= 2 else (parts[0] if parts else label)
                self.series_combo.addItem(f"Coordinate: {display_name}", ("coordinate", index))
        if self.trial.measured_marker_labels:
            for index, label in enumerate(self.trial.measured_marker_labels):
                self.series_combo.addItem(f"Measured: {label}", ("measured", index))
        if self.trial.model_marker_labels:
            for index, label in enumerate(self.trial.model_marker_labels):
                self.series_combo.addItem(f"Model: {label}", ("model", index))
        if self.trial.grf_forces is not None:
            labels = self.trial.grf_labels or [
                f"platform {index + 1}" for index in range(self.trial.grf_forces.shape[1])
            ]
            for index, label in enumerate(labels):
                self.series_combo.addItem(f"GRF: {label} magnitude", ("grf", index))
        if self.series_combo.count() == 0:
            self.series_combo.addItem("No chart series", None)

    def _load_bundled_example(self) -> None:
        try:
            trial = _load_default_trial()
            self._replace_trial(trial)
        except Exception as exc:
            self.QtWidgets.QMessageBox.critical(self.window, "Could not load example", str(exc))

    def _load_c3d_file(self) -> None:
        source_path = self.trial.c3d_path or Path.cwd()
        file_name, _selected_filter = self.QtWidgets.QFileDialog.getOpenFileName(
            self.window,
            "Select a standalone C3D trial",
            str(source_path if source_path.is_dir() else source_path.parent),
            "C3D files (*.c3d);;All files (*)",
        )
        if not file_name:
            return
        try:
            self._replace_trial(_load_trial(c3d_path=file_name))
        except Exception as exc:
            self.QtWidgets.QMessageBox.critical(self.window, "Could not load C3D", str(exc))

    def _load_trial_folder(self) -> None:
        source_path = next(
            (
                path
                for path in (
                    self.trial.model_path,
                    self.trial.marker_path,
                    self.trial.grf_path,
                    self.trial.coordinate_path,
                    self.trial.activity_path,
                    self.trial.c3d_path,
                )
                if path is not None
            ),
            Path.cwd(),
        )
        folder_name = self.QtWidgets.QFileDialog.getExistingDirectory(
            self.window,
            "Select an OpenSim trial folder",
            str(source_path if source_path.is_dir() else source_path.parent),
        )
        if not folder_name:
            return
        folder = Path(folder_name)
        selected = self._select_trial_files(folder)
        if selected is None:
            return
        model, coordinates, markers, grf, activity = selected
        if model is None and coordinates is None and markers is None and grf is None:
            self.QtWidgets.QMessageBox.warning(
                self.window,
                "No plottable data selected",
                "Select at least a model, kinematics, measured-marker, or ground-reaction file.",
            )
            return
        try:
            trial = _load_trial(
                model_path=model,
                coordinate_path=coordinates,
                marker_path=markers,
                grf_path=grf,
                activity_path=activity,
            )
            self._replace_trial(trial)
        except Exception as exc:
            self.QtWidgets.QMessageBox.critical(self.window, "Could not load trial", str(exc))

    def _select_trial_files(
        self, folder: Path
    ) -> tuple[Path | None, Path | None, Path | None, Path | None, Path | None] | None:
        """Let the user select a consistent model/trial file set."""

        QtWidgets = self.QtWidgets
        models = sorted(folder.glob("*.osim"))
        kinematics = sorted(folder.glob("*.sto")) + sorted(folder.glob("*.mot"))
        markers = sorted(folder.glob("*.trc"))
        grfs = sorted(folder.glob("*.mot"))
        activities = sorted(folder.glob("*.sto")) + sorted(folder.glob("*.mot"))
        dialog = QtWidgets.QDialog(self.window)
        dialog.setWindowTitle("Select OpenSim trial files")
        dialog.setMinimumWidth(620)
        layout = QtWidgets.QFormLayout(dialog)
        description = QtWidgets.QLabel(
            "Choose the files belonging to one trial. A model is optional when plotting "
            "measured markers or ground-reaction forces."
        )
        description.setWordWrap(True)
        layout.addRow(description)

        def add_selector(
            label: str,
            paths: list[Path],
            preferred: tuple[str, ...],
            optional: bool = False,
            select_first: bool = False,
        ) -> QtWidgets.QComboBox:
            combo = QtWidgets.QComboBox()
            if optional:
                combo.addItem("— none —", None)
            for path in paths:
                combo.addItem(path.name, str(path))
            preferred_index = self._preferred_path_index(paths, preferred)
            if preferred_index is None and select_first and paths:
                preferred_index = 0
            if preferred_index is not None:
                combo.setCurrentIndex(preferred_index + (1 if optional else 0))
            layout.addRow(label, combo)
            return combo

        model_combo = add_selector(
            "Model (.osim)",
            models,
            ("subject_walk_scaled.osim",),
            optional=True,
            select_first=True,
        )
        coordinate_combo = add_selector(
            "Kinematics (.sto/.mot)",
            kinematics,
            ("coordinates.sto",),
            optional=True,
        )
        marker_combo = add_selector(
            "Measured markers (.trc)",
            markers,
            ("marker_trajectories.trc",),
            optional=True,
            select_first=True,
        )
        grf_combo = add_selector(
            "Ground reactions (.mot)",
            grfs,
            ("grf_walk.mot",),
            optional=True,
        )
        activity_combo = add_selector(
            "Muscle activity / EMG (.sto/.mot)",
            activities,
            ("electromyography.sto", "activations.sto"),
            optional=True,
        )

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addRow(buttons)
        if dialog.exec_() != QtWidgets.QDialog.Accepted:
            return None

        def selected_path(combo: QtWidgets.QComboBox) -> Path | None:
            value = combo.currentData()
            return Path(value) if value else None

        return (
            selected_path(model_combo),
            selected_path(coordinate_combo),
            selected_path(marker_combo),
            selected_path(grf_combo),
            selected_path(activity_combo),
        )

    @staticmethod
    def _preferred_path_index(paths: list[Path], preferred: tuple[str, ...]) -> int | None:
        lowered = {path.name.lower(): index for index, path in enumerate(paths)}
        for name in preferred:
            if name.lower() in lowered:
                return lowered[name.lower()]
        return None

    def _replace_trial(self, trial: OpenSimTrial) -> None:
        self.pause()
        display_path = next(
            (
                path
                for path in (
                    trial.model_path,
                    trial.marker_path,
                    trial.grf_path,
                    trial.coordinate_path,
                    trial.c3d_path,
                )
                if path is not None
            ),
            None,
        )
        display_name = display_path.name if display_path is not None else "trial"
        self._replacement_window = OpenSimVisualizerWindow(
            trial,
            title=f"OpenSim Python Visualiser — {display_name}",
        )
        self._replacement_window.show()
        self.window.close()

    def _build_scene(self) -> None:
        pv = self.pv
        self.plotter.set_background("#f4f5f7")
        self.plotter.add_axes()
        self._marker_measured = None
        self._marker_model = None
        self._model_label_actor = None
        self._model_label_poly = None
        self._force_actor = None
        self._cop_actor = None
        grf_count = self.trial.grf_forces.shape[1] if self.trial.grf_forces is not None else 0
        self._force_arrow_actors: list[Any | None] = [None] * grf_count
        self._force_arrow_polys: list[Any | None] = [None] * grf_count
        self._cop_poly = None
        self._ground_actor = None

        for spec in self.trial.geometry:
            if not spec.mesh_path.exists():
                continue
            try:
                mesh = pv.read(str(spec.mesh_path)).extract_surface().triangulate().copy(deep=True)
            except Exception as exc:
                self.status_label.setText(f"Could not read {spec.mesh_path.name}: {exc}")
                continue
            local_points = np.asarray(mesh.points, dtype=float) * spec.scale[None, :]
            mesh.points = local_points
            actor = self.plotter.add_mesh(
                mesh,
                color=spec.color,
                opacity=spec.opacity,
                smooth_shading=True,
                name=f"geometry:{spec.name}",
            )
            self._geometry_entries.append({"spec": spec, "mesh": mesh, "local": local_points, "actor": actor})

        if self.trial.measured_markers is not None:
            measured = pv.PolyData(_finite_points(self.trial.measured_markers, 0))
            self._marker_measured = self.plotter.add_mesh(
                measured, color="#e34a33", point_size=9, render_points_as_spheres=True, name="measured-markers"
            )
            self._measured_poly = measured
        else:
            self._measured_poly = None

        if self.trial.model_markers is not None:
            model = pv.PolyData(_finite_points(self.trial.model_markers, 0))
            self._marker_model = self.plotter.add_mesh(
                model, color="#2171b5", point_size=7, render_points_as_spheres=True, name="model-markers"
            )
            self._model_poly = model
        else:
            self._model_poly = None

        self._create_muscle_actor()

        bounds = self._bounds()
        low = np.asarray(bounds[::2], dtype=float)
        high = np.asarray(bounds[1::2], dtype=float)
        # Centre and size the floor from the measured marker cloud when it is
        # available. This keeps the walking data centred even if mesh bounds
        # or model-only markers extend farther in one horizontal direction.
        ground_low = low.copy()
        ground_high = high.copy()
        if self.trial.measured_markers is not None:
            measured = self.trial.measured_markers
            finite_measured = measured[np.isfinite(measured).all(axis=2)]
            if finite_measured.size:
                ground_low = np.min(finite_measured, axis=0)
                ground_high = np.max(finite_measured, axis=0)
        ground_size_x = max(float(ground_high[0] - ground_low[0]) * 10.0, 2.0)
        ground_size_z = max(float(ground_high[2] - ground_low[2]) * 10.0, 2.0)
        ground = pv.Plane(
            center=(
                (ground_low[0] + ground_high[0]) / 2.0,
                0.0,
                (ground_low[2] + ground_high[2]) / 2.0,
            ),
            direction=(0.0, 1.0, 0.0),
            # For a Y-normal PyVista plane, the local i/j directions map to
            # Z/X respectively; provide the horizontal extents accordingly.
            i_size=ground_size_z,
            j_size=ground_size_x,
            i_resolution=12,
            j_resolution=12,
        )
        self._ground_actor = self.plotter.add_mesh(
            ground,
            color="#cbd5df",
            opacity=0.42,
            show_edges=True,
            edge_color="#aeb7c2",
            name="ground-plane",
        )
        self.plotter.reset_camera(bounds=bounds)
        self._set_view("iso")
        # Terrain-style navigation behaves like MATLAB rotate3d/Three.js
        # OrbitControls: azimuth/elevation around the focal point without
        # free trackball roll. The initial camera uses OpenSim's Y-up world.
        self.plotter.enable_terrain_style(mouse_wheel_zooms=1.08, shift_pans=True)
        self._update_visibility()

    def _bounds(self) -> tuple[float, float, float, float, float, float]:
        chunks: list[np.ndarray] = []
        for entry in self._geometry_entries:
            transforms = entry["spec"].transforms
            for index in (0, min(self.trial.frame_count - 1, transforms.shape[0] - 1)):
                chunks.append(_transform_points(entry["local"], transforms[index]))
        for values in (self.trial.measured_markers, self.trial.model_markers):
            if values is not None:
                finite = values[np.isfinite(values).all(axis=2)]
                if finite.size:
                    chunks.append(finite)
        if self.trial.grf_points is not None:
            finite_grf_points = self.trial.grf_points[
                np.isfinite(self.trial.grf_points).all(axis=2)
            ]
            if finite_grf_points.size:
                chunks.append(finite_grf_points)
        if not chunks:
            return (-1, 1, -1, 1, -1, 1)
        points = np.vstack(chunks)
        low = np.nanmin(points, axis=0)
        high = np.nanmax(points, axis=0)
        span = np.maximum(high - low, 0.5)
        pad = span * 0.08
        return (
            float(low[0] - pad[0]),
            float(high[0] + pad[0]),
            float(low[1] - pad[1]),
            float(high[1] + pad[1]),
            float(low[2] - pad[2]),
            float(high[2] + pad[2]),
        )

    def _set_speed(self, _index: int) -> None:
        self.speed = float(self.speed_combo.currentData())
        if self._timer.isActive():
            self._play_wall_start = time.perf_counter()
            self._play_sim_start = float(self.trial.times[self.frame])
            self._start_timer()

    def _start_timer(self) -> None:
        if self.trial.frame_count < 2:
            return
        # The timer is only a refresh opportunity. The simulation frame is
        # selected from wall-clock time in _advance_frame, so slow renders
        # automatically skip source frames instead of falling behind.
        self._timer.start(16)

    def play(self) -> None:
        if self._timer.isActive():
            return
        self._play_wall_start = time.perf_counter()
        self._play_sim_start = float(self.trial.times[self.frame])
        self._start_timer()

    def pause(self) -> None:
        self._timer.stop()
        self._play_wall_start = None
        self._play_sim_start = None

    def _advance_frame(self) -> None:
        if self._play_wall_start is None or self._play_sim_start is None:
            self.play()
            return
        elapsed = time.perf_counter() - self._play_wall_start
        target_time = self._play_sim_start + elapsed * self.speed
        first_time = float(self.trial.times[0])
        last_time = float(self.trial.times[-1])
        duration = last_time - first_time
        if duration > 0.0:
            target_time = first_time + ((target_time - first_time) % duration)
        next_frame = int(np.searchsorted(self.trial.times, target_time, side="right") - 1)
        next_frame = int(np.clip(next_frame, 0, self.trial.frame_count - 1))
        if next_frame == self.frame:
            return
        self._advancing_frame = True
        try:
            self._set_frame(next_frame)
        finally:
            self._advancing_frame = False

    def _set_frame(self, frame: int) -> None:
        if self.trial.frame_count == 0:
            return
        self.frame = int(np.clip(frame, 0, self.trial.frame_count - 1))
        if self._timer.isActive() and not self._advancing_frame:
            self._play_wall_start = time.perf_counter()
            self._play_sim_start = float(self.trial.times[self.frame])
        if self.frame_slider.value() != self.frame:
            self.frame_slider.blockSignals(True)
            self.frame_slider.setValue(self.frame)
            self.frame_slider.blockSignals(False)
        self.frame_label.setText(
            f"Frame {self.frame + 1}/{self.trial.frame_count}  |  t = {self.trial.times[self.frame]:.3f} s"
        )
        for entry in self._geometry_entries:
            entry["mesh"].points = _transform_points(entry["local"], entry["spec"].transforms[self.frame])
            if hasattr(entry["mesh"], "Modified"):
                entry["mesh"].Modified()
        if self._measured_poly is not None:
            self._measured_poly.points = _finite_points(self.trial.measured_markers, self.frame)
            if hasattr(self._measured_poly, "Modified"):
                self._measured_poly.Modified()
        if self._model_poly is not None:
            self._model_poly.points = _finite_points(self.trial.model_markers, self.frame)
            if hasattr(self._model_poly, "Modified"):
                self._model_poly.Modified()
        self._update_model_labels()
        self._update_muscle_actor()
        self._update_force_actor()
        self._update_chart_cursor()
        self.plotter.render()

    def _update_force_actor(self) -> None:
        if self.trial.grf_forces is None or self.trial.grf_points is None:
            return
        if self._cop_poly is None:
            self._cop_poly = self.pv.PolyData(np.zeros((1, 3), dtype=float))
            self._cop_actor = self.plotter.add_mesh(
                self._cop_poly,
                color="#d95f02",
                point_size=12,
                render_points_as_spheres=True,
                name="grf-cop",
            )
        if not self.layer_checks["grf"].isChecked():
            for actor in self._force_arrow_actors:
                if actor is not None:
                    actor.SetVisibility(False)
            self._cop_actor.SetVisibility(False)
            return
        origins = self._grf_origins()
        forces = self.trial.grf_forces[self.frame]
        cop_points: list[np.ndarray] = []
        for platform_index, (origin, force) in enumerate(zip(origins, forces)):
            arrow_actor = self._force_arrow_actors[platform_index]
            if (
                not np.isfinite(origin).all()
                or not np.isfinite(force).all()
                or np.linalg.norm(force) <= 5.0
            ):
                if arrow_actor is not None:
                    arrow_actor.SetVisibility(False)
                continue
            cop_points.append(origin)
            force_length = float(np.linalg.norm(force) * 0.001)
            arrow = self.pv.Arrow(
                start=origin,
                direction=force,
                tip_length=0.18,
                tip_radius=0.035,
                shaft_radius=0.014,
                tip_resolution=16,
                shaft_resolution=16,
                scale=force_length,
            )
            if self._force_arrow_polys[platform_index] is None:
                self._force_arrow_polys[platform_index] = arrow.copy(deep=True)
                self._force_arrow_actors[platform_index] = self.plotter.add_mesh(
                    self._force_arrow_polys[platform_index],
                    color="#31a354",
                    name=f"grf-arrow-{platform_index}",
                )
            else:
                self._force_arrow_polys[platform_index].deep_copy(arrow)
                if hasattr(self._force_arrow_polys[platform_index], "Modified"):
                    self._force_arrow_polys[platform_index].Modified()
            self._force_arrow_actors[platform_index].SetVisibility(True)
        if cop_points:
            cop_array = np.vstack(cop_points)
            visible = True
        else:
            cop_array = np.zeros((1, 3), dtype=float)
            visible = False
        self._cop_poly.points = cop_array
        if hasattr(self._cop_poly, "Modified"):
            self._cop_poly.Modified()
        self._cop_actor.SetVisibility(visible)

    def _update_model_labels(self) -> None:
        if self.trial.model_markers is None:
            return
        show_labels = self.layer_checks["model_labels"].isChecked()
        if self._model_label_poly is None:
            self._model_label_poly = self.pv.PolyData(_finite_points(self.trial.model_markers, self.frame))
            self._model_label_poly["labels"] = np.asarray(self.trial.model_marker_labels)
            self._model_label_actor = self.plotter.add_point_labels(
                self._model_label_poly,
                "labels",
                font_size=10,
                text_color="#202020",
                shape="rounded_rect",
                shape_color="#ffffff",
                shape_opacity=0.72,
                margin=2,
                show_points=False,
                always_visible=True,
                name="model-marker-labels",
            )
        else:
            self._model_label_poly.points = _finite_points(self.trial.model_markers, self.frame)
            if hasattr(self._model_label_poly, "Modified"):
                self._model_label_poly.Modified()
        if self._model_label_actor is not None:
            self._model_label_actor.SetVisibility(show_labels)

    def _muscle_polydata_for_frame(self, frame: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Pack all muscle paths into one line mesh for fast animation."""

        point_chunks: list[np.ndarray] = []
        line_chunks: list[np.ndarray] = []
        colors: list[np.ndarray] = []
        point_offset = 0
        for muscle_index, paths in enumerate(self.trial.muscle_paths):
            if frame >= len(paths):
                continue
            points = np.asarray(paths[frame], dtype=float).reshape(-1, 3)
            if points.shape[0] < 2 or not np.isfinite(points).all():
                continue
            point_chunks.append(points)
            line_chunks.append(
                np.r_[points.shape[0], np.arange(point_offset, point_offset + points.shape[0])]
            )
            point_offset += points.shape[0]
            activity = np.nan
            if self.trial.muscle_activity is not None:
                activity = float(self.trial.muscle_activity[frame, muscle_index])
            colors.append(_activity_color(activity))
        if not point_chunks:
            return (
                np.zeros((0, 3), dtype=float),
                np.zeros(0, dtype=np.int64),
                np.zeros((0, 3), dtype=np.uint8),
            )
        return (
            np.vstack(point_chunks),
            np.concatenate(line_chunks).astype(np.int64),
            np.vstack(colors),
        )

    def _create_muscle_actor(self) -> None:
        if not self.trial.muscle_paths:
            return
        points, lines, colors = self._muscle_polydata_for_frame(0)
        if not lines.size:
            return
        self._muscle_poly = self.pv.PolyData(points, lines=lines)
        self._muscle_poly.cell_data["activity_rgb"] = colors
        self._muscle_actor = self.plotter.add_mesh(
            self._muscle_poly,
            scalars="activity_rgb",
            rgb=True,
            preference="cell",
            line_width=6,
            render_lines_as_tubes=True,
            show_scalar_bar=False,
            name="muscles",
        )

    def _update_muscle_actor(self) -> None:
        if self._muscle_actor is None or self._muscle_poly is None:
            return
        show_muscles = self.layer_checks["muscles"].isChecked()
        self._muscle_actor.SetVisibility(show_muscles)
        if not show_muscles:
            return
        points, lines, colors = self._muscle_polydata_for_frame(self.frame)
        self._muscle_poly.clear_data()
        self._muscle_poly.points = points
        self._muscle_poly.lines = lines
        self._muscle_poly.cell_data["activity_rgb"] = colors
        if hasattr(self._muscle_poly, "Modified"):
            self._muscle_poly.Modified()

    def _grf_origins(self) -> np.ndarray:
        """Return the ground-frame CoP/point-of-application positions."""

        return np.asarray(self.trial.grf_points[self.frame], dtype=float).copy()

    def _update_visibility(self) -> None:
        if self._ground_actor is not None:
            self._ground_actor.SetVisibility(self.layer_checks["ground"].isChecked())
        show_geometry = self.layer_checks["geometry"].isChecked()
        for entry in self._geometry_entries:
            entry["actor"].SetVisibility(show_geometry)
        if self._marker_measured is not None:
            self._marker_measured.SetVisibility(self.layer_checks["measured"].isChecked())
        if self._marker_model is not None:
            self._marker_model.SetVisibility(self.layer_checks["model_markers"].isChecked())
        self._update_model_labels()
        self._update_muscle_actor()
        self._update_force_actor()
        self.plotter.render()

    def _refresh_chart(self) -> None:
        self.chart.clear()
        self._chart_cursor = None
        if self.series_combo.currentData() is None:
            return
        kind, index = self.series_combo.currentData()
        if kind == "measured" and self.trial.measured_markers is not None:
            values = self.trial.measured_markers[:, index, :]
            labels = ("X", "Y", "Z")
        elif kind == "model" and self.trial.model_markers is not None:
            values = self.trial.model_markers[:, index, :]
            labels = ("X", "Y", "Z")
        elif kind == "coordinate":
            values = self.trial.coordinates[:, index, None]
            labels = ("value",)
        elif kind == "grf" and self.trial.grf_forces is not None:
            values = np.linalg.norm(self.trial.grf_forces[:, index, :], axis=1)[:, None]
            labels = ("magnitude",)
        else:
            return
        colors = ("#e34a33", "#2171b5", "#31a354")
        for component, label in enumerate(labels):
            if label in self.axis_checks and not self.axis_checks[label].isChecked():
                continue
            self.chart.plot(self.trial.times, values[:, component], pen=colors[component], name=label)
        self._chart_cursor = self.chart.addLine(
            x=float(self.trial.times[self.frame]),
            pen=self.pg.mkPen("#555", style=self.QtCore.Qt.DashLine),
        )

    def _update_chart_cursor(self) -> None:
        if self._chart_cursor is not None:
            self._chart_cursor.setValue(float(self.trial.times[self.frame]))

    def _set_view(self, view: str) -> None:
        bounds = self._bounds()
        low = np.asarray(bounds[::2], dtype=float)
        high = np.asarray(bounds[1::2], dtype=float)
        center = (low + high) / 2.0
        distance = max(float(np.linalg.norm(high - low) * 1.7), 2.0)
        if view == "xy":
            direction, up = np.array([0.0, 0.0, 1.0]), np.array([0.0, 1.0, 0.0])
        elif view == "xz":
            direction, up = np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0])
        elif view == "yz":
            direction, up = np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])
        else:
            direction = np.array([1.0, 1.0, 1.0]) / np.sqrt(3.0)
            up = np.array([0.0, 1.0, 0.0])
        self.plotter.camera_position = [center + direction * distance, center, up]
        self.plotter.reset_camera_clipping_range()

    def show(self) -> None:
        self.window.show()


def _launch(
    trial: OpenSimTrial | None = None,
    *,
    bundle: str | Path = DEFAULT_BUNDLE,
) -> OpenSimVisualizerWindow:
    """Launch the visualiser and return its window object."""

    QtCore, QtWidgets, _pg, _parts = _gui_imports()
    trial = trial or _load_default_trial(bundle)
    app = QtWidgets.QApplication.instance()
    owns_app = app is None
    if app is None:
        app = QtWidgets.QApplication([])
    window = OpenSimVisualizerWindow(trial)
    window.show()
    if owns_app:
        app.exec_()
    return window
