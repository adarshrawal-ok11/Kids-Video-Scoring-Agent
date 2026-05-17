from __future__ import annotations

import json
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from src.state import State

T = TypeVar("T", bound=BaseModel)

CONTENT_SAFETY_SUBVARS = [
    "violence_aggression",
    "scary_fear_inducing",
    "inappropriate_language",
    "bullying_exclusion",
    "stereotyping",
    "fantasy_level",
    "negative_behavior_modeling",
]

STIMULATION_SUBVARS = ["visual_pacing", "audio_energy", "color_intensity", "sensory_load"]
EDUCATION_SUBVARS = ["concepts_taught", "participation", "social_modeling", "repetition", "language_quality"]
CHANNEL_LEVEL_SUBVARS = ["content_consistency", "channel_reputation"]


def prompt_text(name: str) -> str:
    return Path("prompts", name).read_text(encoding="utf-8")


def compact_state_context(state: State, include_frames: bool = False) -> str:
    metadata = state.metadata
    payload = {
        "url": state.url,
        "video_id": state.video_id,
        "metadata": metadata.model_dump() if metadata else {},
        "transcript": state.transcript.model_dump() if state.transcript else {},
        "programmatic": state.programmatic.model_dump() if state.programmatic else {},
        "veto": state.veto.model_dump() if state.veto else {},
        "narrator": state.narrator.model_dump() if state.narrator else {},
        "similar_gold": [g.model_dump() for g in state.similar_gold],
        "scorer_gemini": state.scorer_gemini.model_dump() if state.scorer_gemini else {},
        "scorer_claude": state.scorer_claude.model_dump() if state.scorer_claude else {},
        "frame_paths": state.frame_paths if include_frames else [],
    }
    return json.dumps(payload, ensure_ascii=True, indent=2)


def validate_model(model_cls: type[T], payload: dict, context: str) -> T:
    try:
        return model_cls(**payload)
    except ValidationError as exc:
        raise ValueError(f"{context} response failed schema validation: {exc}") from exc


def average(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0
