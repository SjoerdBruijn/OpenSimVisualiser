# OpenSimVisualiser

OpenSimVisualiser is a desktop Python application for exploring an OpenSim model together with motion-capture and trial data. It ships with OpenSim's `example3DWalking` data, so you can launch a working example immediately after installation.

![OpenSimVisualiser displaying the bundled walking example with model geometry, markers, muscles, and a time-series chart](docs/assets/opensim-visualiser.png)

## Features

- interactive 3D model geometry with standard camera controls;
- animated kinematics, model markers, and measured markers;
- ground-reaction force vectors and centres of pressure;
- marker and ground-reaction-force plotting without an OpenSim model;
- standalone C3D loading for marker data, force-platform data, or both;
- muscle paths coloured by activation or EMG level;
- playback speed, frame scrubbing, layer controls, and time-series charts;
- a Python API for opening your own OpenSim trial.

## Requirements

- Conda or Miniforge;
- Git.

The supplied Conda environment installs Python 3.12, OpenSim 4.5.2, ezc3d, Spyder, and all visualization dependencies. OpenSim comes from its official `opensim-org` channel.

## Install

Clone the repository and create the complete environment from `environment.yml`:

```bash
git clone https://github.com/SjoerdBruijn/OpenSimVisualiser.git
cd OpenSimVisualiser
conda env create --file environment.yml
conda activate opensim-visualiser
```

The environment includes OpenSim, Spyder, all visualization dependencies, and an editable installation of OpenSimVisualiser. Start Spyder from the activated environment so its console uses the correct interpreter:

```bash
spyder
```

You can use the Python API from Spyder or launch the bundled example from the same activated Conda environment:

```bash
opensim-visualiser
```

To refresh an existing environment after `environment.yml` changes:

```bash
conda env update --file environment.yml --prune
```

Confirm that the bindings are available with:

```bash
python -c "import opensim; print(opensim.__version__)"
```

## Run the bundled example

After installation, either command opens the application with the included walking trial:

```bash
opensim-visualiser
# or
python -m OpenSimVisualiser
```

The application has two separate input routes:

1. Use **Load trial folder…** for OpenSim-style inputs. Models (`.osim`), kinematics (`.sto` or `.mot`), markers (`.trc`), ground reactions (`.mot`), and activity/EMG (`.sto` or `.mot`) are optional, but at least one plottable file must be selected. Marker and ground-reaction data can be viewed without a model.
2. Use **Load C3D file…** for one standalone `.c3d` file. The visualiser loads its marker data, force-platform data, or both with ezc3d. This route does not use an OpenSim model or any of the separate trial-file inputs.

## Python API

```python
from OpenSimVisualiser import OpenSimVisualiser

window = OpenSimVisualiser(
    model_path="path/to/model.osim",              # optional
    coordinate_path="path/to/coordinates.sto",  # optional
    marker_path="path/to/markers.trc",           # optional
    grf_path="path/to/forces.mot",               # optional
    activity_path="path/to/activations.sto",     # optional
)
```

To inspect measured markers, ground-reaction forces, or both without a model:

```python
from OpenSimVisualiser import OpenSimVisualiser

window = OpenSimVisualiser(
    marker_path="path/to/markers.trc",  # either marker_path or grf_path may be omitted
    grf_path="path/to/forces.mot",
)
```

Provide at least one model, kinematics, marker, or ground-reaction file. A model is only needed for model geometry, evaluated model markers, and muscle paths. When a model is supplied without kinematics, its default pose is used; marker or GRF timestamps drive playback when either data source is present.

### Standalone C3D route

To load marker and/or force-platform data directly from a C3D file:

```python
from OpenSimVisualiser import OpenSimVisualiser

window = OpenSimVisualiser(c3d_path="path/to/trial.c3d")
```

The C3D route is separate from the OpenSim-style route above: pass only `c3d_path`. Do not combine it with `model_path`, `coordinate_path`, `marker_path`, `grf_path`, or `activity_path`. Markers are plotted when point data is present; ground-reaction forces are plotted when the C3D contains valid force-platform data. C3D coordinates are converted from the conventional Z-up frame used by ezc3d's force-platform calculations to the visualiser's Y-up frame.

## Development

The project uses a `src` layout, with the installable package in `src/OpenSimVisualiser` and tests in `tests`.

```bash
python -m pip install -e ".[dev]"
python -m pytest
python -m build
```

## Bundled example data

The included `example3DWalking` bundle originates from the OpenSim 4.5 Python/Moco examples. It contains the model, geometry meshes, kinematics, marker trajectories, ground-reaction forces, and electromyography data used by the default demo.

## License

OpenSimVisualiser is available under the [MIT License](LICENSE).

This project was vibecoded using Codex GPT-5.6 Sol.
