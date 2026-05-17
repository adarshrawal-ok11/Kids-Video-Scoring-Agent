from __future__ import annotations

from functools import lru_cache

from sentence_transformers import SentenceTransformer

from src.state import State
from src.utils import get_config


@lru_cache(maxsize=1)
def _embedding_model() -> SentenceTransformer:
    model_name = get_config()["embedding"]["model"]
    return SentenceTransformer(model_name)


def compute_video_embedding(state: State) -> State:
    text = build_embedding_text(state)
    if not text.strip():
        raise ValueError("Cannot compute embedding without metadata, transcript, or narrator text")
    embedding = _embedding_model().encode(
        text,
        normalize_embeddings=bool(get_config()["embedding"].get("normalize", True)),
    )
    state.embedding = [float(v) for v in embedding.tolist()]
    return state


def build_embedding_text(state: State) -> str:
    metadata = state.metadata
    narrator_lines = []
    if state.narrator:
        narrator_lines = [
            f"{chunk.timestamp}: {chunk.what_happens} Tone: {chunk.tone} Concerns: {chunk.concerns}"
            for chunk in state.narrator.chunks
        ]
    return "\n".join(
        part
        for part in [
            f"Title: {metadata.title}" if metadata and metadata.title else "",
            f"Channel: {metadata.channel}" if metadata and metadata.channel else "",
            f"Description: {metadata.description}" if metadata and metadata.description else "",
            f"Tags: {', '.join(metadata.tags)}" if metadata and metadata.tags else "",
            f"Transcript: {state.transcript.text}" if state.transcript and state.transcript.text else "",
            "Narrator:\n" + "\n".join(narrator_lines) if narrator_lines else "",
        ]
        if part
    )
