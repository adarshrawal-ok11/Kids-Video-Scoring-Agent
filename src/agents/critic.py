from __future__ import annotations

from src.agents.common import compact_state_context, prompt_text, validate_model
from src.llm_clients import call_gemini_text
from src.state import CriticOutput, State
from src.utils import get_config


def run_critic(state: State) -> State:
    config = get_config()
    base_prompt = prompt_text("critic_prompt.txt").format(
        video_context=compact_state_context(state),
    )
    prompt = base_prompt
    for attempt in range(2):
        payload, cost, thinking = call_gemini_text(
            config["models"]["critic"],
            prompt,
            context="critic",
        )
        state.cost.critic += cost
        state.cost.total += cost
        try:
            state.critic = validate_model(CriticOutput, payload, "critic")
            state.critic.thinking = thinking
            return state
        except ValueError as exc:
            if attempt == 1:
                raise
            prompt = f"{base_prompt}\n\nYour last response had JSON schema errors: {exc}. Reformat as valid JSON only."
    return state
