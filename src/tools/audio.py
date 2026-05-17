from __future__ import annotations

import librosa
import numpy as np
from typing import Optional

from src.state import ProgrammaticAnalysis, State


def analyze_audio_energy(state: State) -> State:
    if not state.audio_path:
        raise ValueError("audio_path is required before audio analysis")

    y, sr = librosa.load(state.audio_path, sr=None, mono=True)
    if y.size == 0:
        raise ValueError("Audio file is empty")

    rms = librosa.feature.rms(y=y)[0]
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    mean_rms = float(np.mean(rms))
    peak_rms = float(np.max(rms))
    std_rms = float(np.std(rms))
    brightness = float(np.mean(centroid) / max(sr / 2.0, 1.0))

    intensity = (
        min(mean_rms / 0.12, 1.0) * 0.35
        + min(peak_rms / 0.35, 1.0) * 0.25
        + min(std_rms / 0.08, 1.0) * 0.20
        + min(brightness / 0.55, 1.0) * 0.20
    )
    score = int(round(max(0.0, min(100.0, 100.0 - intensity * 100.0))))

    programmatic = state.programmatic or ProgrammaticAnalysis()
    programmatic.audio = {
        "mean_rms": round(mean_rms, 5),
        "peak_rms": round(peak_rms, 5),
        "std_rms": round(std_rms, 5),
        "brightness": round(brightness, 5),
        "score": score,
    }
    state.programmatic = programmatic
    _update_sensory_load(state)
    return state


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
