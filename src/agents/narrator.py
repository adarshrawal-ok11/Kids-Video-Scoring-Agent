from __future__ import annotations

from src.agents.common import prompt_text, validate_model
from src.llm_clients import call_gemini_video
from src.state import NarratorOutput, State
from src.utils import get_config


def run_narrator(state: State) -> State:
    if not state.video_path:
        raise ValueError("video_path is required before narrator")
    config = get_config()
    base_prompt = prompt_text("narrator_prompt.txt").format(
        interval_seconds=config.get("narrator", {}).get("interval_seconds", 10),
        max_chunks=config.get("narrator", {}).get("max_chunks", 60),
    )
    prompt = base_prompt
    for attempt in range(2):
        payload, cost, _ = call_gemini_video(
            config["models"]["narrator"],
            prompt,
            state.video_path,
            context="narrator",
        )
        state.cost.narrator += cost
        state.cost.total += cost
        try:
            state.narrator = validate_model(NarratorOutput, payload, "narrator")
            return state
        except ValueError as exc:
            if attempt == 1:
                raise
            prompt = f"{base_prompt}\n\nYour last response had JSON schema errors: {exc}. Reformat as valid JSON only."
    return state
