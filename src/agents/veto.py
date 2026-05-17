from __future__ import annotations

from src.agents.common import compact_state_context, prompt_text, validate_model
from src.llm_clients import call_gemini_text
from src.state import State, VetoResult
from src.utils import get_config


def run_veto_agent(state: State) -> State:
    frames = state.frame_paths[: int(get_config().get("veto", {}).get("n_frames", 5))]
    base_prompt = prompt_text("veto_prompt.txt").format(
        video_context=compact_state_context(state),
    )
    prompt = base_prompt
    for attempt in range(2):
        payload, cost, thinking = call_gemini_text(
            get_config()["models"]["veto"],
            prompt,
            images=frames,
            context="veto",
        )
        state.cost.veto += cost
        state.cost.total += cost
        try:
            state.veto = validate_model(VetoResult, payload, "veto")
            state.veto.thinking = thinking
            return state
        except ValueError as exc:
            if attempt == 1:
                raise
            prompt = f"{base_prompt}\n\nYour last response had JSON schema errors: {exc}. Reformat as valid JSON only."
    return state
