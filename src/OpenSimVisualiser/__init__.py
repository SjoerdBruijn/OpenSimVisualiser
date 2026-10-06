"""Public entry point for the native Python OpenSim visualiser."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .visualiser import OpenSimVisualizerWindow

__all__ = ["OpenSimVisualiser"]


def OpenSimVisualiser(
    model_path: str | Path,
    coordinate_path: str | Path | None = None,
    marker_path: str | Path | None = None,
    grf_path: str | Path | None = None,
    activity_path: str | Path | None = None,
) -> OpenSimVisualizerWindow:
    """Load an OpenSim model and optional trial data, then open the visualiser."""

    from .data import _load_trial
    from .visualiser import _launch

    trial = _load_trial(
        model_path=model_path,
        coordinate_path=coordinate_path,
        marker_path=marker_path,
        grf_path=grf_path,
        activity_path=activity_path,
    )
    return _launch(trial)
