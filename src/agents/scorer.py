from __future__ import annotations

import json

from src.agents.common import compact_state_context, prompt_text, validate_model
from src.llm_clients import call_claude_text, call_gemini_text
from src.state import ScorerOutput, State
from src.tools.gold_retrieval import load_anchor_videos
from src.utils import get_config


def run_scorer(state: State, provider: str) -> State:
    provider = provider.lower()
    if provider not in {"gemini", "claude"}:
        raise ValueError("provider must be 'gemini' or 'claude'")

    config = get_config()
    anchors = load_anchor_videos()
    base_prompt = prompt_text("scorer_prompt.txt").format(
        provider=provider,
        anchor_examples_block=_format_anchors(anchors),
        video_context=compact_state_context(state),
    )
    prompt = base_prompt

    for attempt in range(2):
        if provider == "gemini":
            payload, cost, thinking = call_gemini_text(
                config["models"]["scorer_gemini"],
                prompt,
                images=state.frame_paths,
                context="scorer_gemini",
            )
            state.cost.scorer_gemini += cost
        else:
            payload, cost, thinking = call_claude_text(
                config["models"]["scorer_claude"],
                prompt,
                images=state.frame_paths,
                context="scorer_claude",
            )
            state.cost.scorer_claude += cost

        state.cost.total += cost
        try:
            output = validate_model(ScorerOutput, payload, f"scorer_{provider}")
            output.thinking = thinking
            if provider == "gemini":
                state.scorer_gemini = output
            else:
                state.scorer_claude = output
            return state
        except ValueError as exc:
            if attempt == 1:
                raise
            prompt = f"{base_prompt}\n\nYour last response had JSON schema errors: {exc}. Reformat as valid JSON only."
    return state


def _format_anchors(anchors: list[dict]) -> str:
    if not anchors:
        return "(No calibration anchors available yet. Use the scoring rubric and scale anchors.)"
    safe_anchors = []
    for item in anchors[:5]:
        safe_anchors.append({
            "title": item.get("title"),
            "channel": item.get("channel"),
            "human_scores": item.get("human_scores") or item.get("scores"),
            "rater_notes": item.get("rater_notes") or item.get("notes"),
        })
    return json.dumps(safe_anchors, ensure_ascii=True, indent=2)
