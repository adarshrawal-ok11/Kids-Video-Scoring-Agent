from __future__ import annotations

from datetime import datetime
from typing import Callable

from src.agents.critic import run_critic
from src.agents.narrator import run_narrator
from src.agents.scorer import run_scorer
from src.agents.veto import run_veto_agent
from src.state import State
from src.tools.audio import analyze_audio_energy
from src.tools.cache import ensure_data_dirs
from src.tools.color import analyze_color
from src.tools.download import download_video
from src.tools.embeddings import compute_video_embedding
from src.tools.extract import extract_audio, extract_frames
from src.tools.gold_retrieval import retrieve_similar_gold
from src.tools.metadata import fetch_metadata
from src.tools.pacing import analyze_pacing
from src.tools.persist import save_to_sqlite
from src.tools.transcript import get_transcript
from src.utils import get_config, log, log_error, setup_logging
from src.verdict import compute_final_verdict


CRITICAL_ACQUISITION_STEPS = {"download_video", "extract_frames"}


def safely_run(step_name: str, func: Callable[[State], State], state: State) -> State:
    try:
        log(f"Starting step: {step_name}")
        result_state = func(state)
        log(f"Completed step: {step_name}")
        return result_state
    except Exception as exc:
        log_error(step_name, exc)
        state.errors.append({
            "step": step_name,
            "error": str(exc),
            "error_type": type(exc).__name__,
            "timestamp": datetime.now().isoformat(),
        })
        return state


def analyze_video(url: str, persist: bool = True) -> State:
    setup_logging(get_config().get("logging", {}).get("level", "INFO"))
    ensure_data_dirs()
    state = State(
        url=url,
        started_at=datetime.now(),
        prompt_version=get_config().get("prompts", {}).get("version", "v2.0"),
    )

    state = safely_run("fetch_metadata", fetch_metadata, state)
    state = safely_run("download_video", download_video, state)
    if _has_failed(state, "download_video") or not state.video_path:
        return _finish_error(
            state,
            persist,
            "Video could not be downloaded; analysis cannot continue.",
        )

    state = safely_run("extract_frames", extract_frames, state)
    if _has_failed(state, "extract_frames") or not state.frame_paths:
        return _finish_error(
            state,
            persist,
            "Frames could not be extracted; visual analysis cannot continue.",
        )

    state = safely_run("extract_audio", extract_audio, state)
    state = safely_run("get_transcript", get_transcript, state)

    state = safely_run("analyze_pacing", analyze_pacing, state)
    state = safely_run("analyze_color", analyze_color, state)
    state = safely_run("analyze_audio_energy", analyze_audio_energy, state)

    state = safely_run("run_veto", run_veto_agent, state)
    state = safely_run("run_narrator", run_narrator, state)

    state = safely_run("compute_embedding", compute_video_embedding, state)
    state = safely_run("retrieve_gold", retrieve_similar_gold, state)

    state = safely_run("scorer_gemini", lambda s: run_scorer(s, "gemini"), state)
    state = safely_run("scorer_claude", lambda s: run_scorer(s, "claude"), state)

    state = safely_run("run_critic", run_critic, state)
    state = safely_run("compute_verdict", compute_final_verdict, state)

    state.completed_at = datetime.now()
    if persist:
        state = safely_run("save_to_db", save_to_sqlite, state)
    return state


def _has_failed(state: State, step_name: str) -> bool:
    return any(error.get("step") == step_name for error in state.errors)


def _finish_error(state: State, persist: bool, reason: str) -> State:
    from src.state import FinalVerdict

    state.final_verdict = FinalVerdict(
        verdict="ERROR",
        overall_score=0.0,
        confidence="LOW",
        reasoning=reason,
        key_decision_drivers=["critical_acquisition_failure"],
        error_reason=reason,
    )
    state.completed_at = datetime.now()
    if persist:
        state = safely_run("save_to_db", save_to_sqlite, state)
    return state
