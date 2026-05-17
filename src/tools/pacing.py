from __future__ import annotations

from scenedetect import ContentDetector, SceneManager, open_video

from src.state import ProgrammaticAnalysis, State


def analyze_pacing(state: State) -> State:
    if not state.video_path:
        raise ValueError("video_path is required before pacing analysis")

    video = open_video(state.video_path)
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector())
    scene_manager.detect_scenes(video)
    scenes = scene_manager.get_scene_list()

    duration_seconds = _duration_seconds(state)
    cuts = max(len(scenes) - 1, 0)
    cuts_per_minute = cuts / max(duration_seconds / 60.0, 0.01)
    score = score_visual_pacing(cuts_per_minute)

    programmatic = state.programmatic or ProgrammaticAnalysis()
    programmatic.pacing = {
        "scene_count": len(scenes),
        "cut_count": cuts,
        "cuts_per_minute": round(cuts_per_minute, 2),
        "score": score,
    }
    state.programmatic = programmatic
    return state


def score_visual_pacing(cuts_per_minute: float) -> int:
    if cuts_per_minute < 3:
        return 100
    if cuts_per_minute < 5:
        return 70
    if cuts_per_minute < 7:
        return 50
    if cuts_per_minute <= 12:
        return 30
    return 0


def _duration_seconds(state: State) -> float:
    if state.metadata and state.metadata.duration_seconds:
        return float(state.metadata.duration_seconds)
    return 1.0
