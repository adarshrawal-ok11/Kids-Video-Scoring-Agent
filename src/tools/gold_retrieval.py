from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.state import GoldReference, State
from src.utils import get_config

GOLD_PATH = Path("data/gold_set_enriched.json")


def retrieve_similar_gold(state: State) -> State:
    videos = load_gold_videos()
    if not videos:
        state.similar_gold = []
        return state
    if not state.embedding:
        raise ValueError("state.embedding is required before gold retrieval")

    query = np.array(state.embedding, dtype=np.float32)
    scored = []
    for item in videos:
        emb = item.get("embedding")
        if not emb:
            continue
        sim = _cosine(query, np.array(emb, dtype=np.float32))
        scored.append((sim, item))

    cfg = get_config().get("retrieval", {})
    k = int(cfg.get("k_similar_gold", 3))
    min_sim = float(cfg.get("min_similarity_threshold", 0.4))

    state.similar_gold = [
        GoldReference(
            id=str(item.get("id") or item.get("video_id") or item.get("url") or ""),
            title=str(item.get("title") or ""),
            similarity=round(float(sim), 4),
            human_scores=item.get("human_scores") or item.get("scores") or {},
            rater_notes=item.get("rater_notes") or item.get("notes"),
        )
        for sim, item in sorted(scored, key=lambda pair: pair[0], reverse=True)[:k]
        if sim >= min_sim
    ]
    return state


def load_gold_videos() -> list[dict]:
    if not GOLD_PATH.exists():
        return []
    payload = json.loads(GOLD_PATH.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return payload
    return payload.get("videos") or []


def load_anchor_videos() -> list[dict]:
    return [item for item in load_gold_videos() if item.get("is_anchor")]


def gold_size() -> int:
    return len(load_gold_videos())


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)
