from __future__ import annotations

import cv2
import numpy as np
from typing import Optional

from src.state import ProgrammaticAnalysis, State


def analyze_color(state: State) -> State:
    if not state.frame_paths:
        raise ValueError("frame_paths are required before color analysis")

    saturations: list[float] = []
    values: list[float] = []
    for path in state.frame_paths:
        frame = cv2.imread(path)
        if frame is None:
            continue
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        saturations.append(float(np.mean(hsv[:, :, 1]) / 255.0))
        values.append(float(np.mean(hsv[:, :, 2]) / 255.0))

    if not saturations:
        raise ValueError("No readable frames for color analysis")

    mean_sat = float(np.mean(saturations))
    score = score_color_intensity(mean_sat)

    programmatic = state.programmatic or ProgrammaticAnalysis()
    programmatic.color = {
        "mean_saturation": round(mean_sat, 4),
        "max_saturation": round(float(np.max(saturations)), 4),
        "mean_value": round(float(np.mean(values)), 4),
        "score": score,
    }
    state.programmatic = programmatic
    _update_sensory_load(state)
    return state


def score_color_intensity(mean_saturation: float) -> int:
    if mean_saturation < 0.25:
        return 100
    if mean_saturation < 0.35:
        return 80
    if mean_saturation < 0.50:
        return 60
    if mean_saturation < 0.65:
        return 50
    if mean_saturation <= 0.80:
        return 25
    return 0


def _update_sensory_load(state: State) -> None:
    p = state.programmatic
    if not p:
        return
    pacing = (p.pacing or {}).get("score")
    audio = (p.audio or {}).get("score")
    color = (p.color or {}).get("score")
    if pacing is None or audio is None or color is None:
        return
    duration_score = _duration_score(state.metadata.duration_seconds if state.metadata else None)
    p.sensory_load = round(
        0.35 * pacing + 0.30 * audio + 0.20 * color + 0.15 * duration_score,
        2,
    )


def _duration_score(duration_seconds: Optional[int]) -> int:
    if not duration_seconds:
        return 70
    minutes = duration_seconds / 60.0
    if 4 <= minutes <= 15:
        return 100
    if 2 <= minutes < 4 or 15 < minutes <= 25:
        return 70
    if minutes <= 30:
        return 45
    return 20
