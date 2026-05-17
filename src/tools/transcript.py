from __future__ import annotations

import json
import re
import urllib.request
from pathlib import Path
from typing import Optional

import yt_dlp

from src.llm_clients import call_whisper_api
from src.state import State, TranscriptResult
from src.tools.cache import YDL_BASE_OPTS, is_cache_fresh, transcript_cache_path


def get_transcript(state: State) -> State:
    video_id = state.video_id or "unknown"
    target = transcript_cache_path(video_id)
    target.parent.mkdir(parents=True, exist_ok=True)

    if is_cache_fresh(target):
        state.transcript = TranscriptResult(**json.loads(target.read_text(encoding="utf-8")))
        state.cache_hits.append("transcript")
        return state

    transcript = _try_youtube_captions(state.url)
    if transcript is None and state.audio_path:
        text, language, cost = call_whisper_api(state.audio_path)
        state.cost.whisper += cost
        state.cost.total += cost
        transcript = TranscriptResult(
            text=text,
            language=language,
            source="whisper_api",
            word_count=len(text.split()),
        )

    if transcript is None:
        transcript = TranscriptResult(
            text="",
            language="unknown",
            source="unavailable",
            word_count=0,
        )

    target.write_text(transcript.model_dump_json(indent=2), encoding="utf-8")
    state.transcript = transcript
    return state


def _try_youtube_captions(url: str) -> Optional[TranscriptResult]:
    opts = {**YDL_BASE_OPTS, "skip_download": True}
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=False)

    caption_sets = [info.get("subtitles") or {}, info.get("automatic_captions") or {}]
    preferred_langs = ["en", "en-US", "en-GB", "hi", "hinglish"]
    for captions in caption_sets:
        lang = _pick_caption_language(captions, preferred_langs)
        if not lang:
            continue
        caption = _pick_caption_format(captions.get(lang) or [])
        if not caption:
            continue
        raw = _fetch_caption(caption["url"])
        text = _caption_to_text(raw, caption.get("ext", ""))
        if text:
            return TranscriptResult(
                text=text,
                language=lang,
                source="youtube_captions",
                word_count=len(text.split()),
            )
    return None


def _pick_caption_language(captions: dict, preferred: list[str]) -> Optional[str]:
    for lang in preferred:
        if lang in captions:
            return lang
    return next(iter(captions.keys()), None)


def _pick_caption_format(formats: list[dict]) -> Optional[dict]:
    if not formats:
        return None
    for ext in ["vtt", "srv3", "json3"]:
        for item in formats:
            if item.get("ext") == ext and item.get("url"):
                return item
    return next((item for item in formats if item.get("url")), None)


def _fetch_caption(url: str) -> str:
    with urllib.request.urlopen(url, timeout=20) as response:
        return response.read().decode("utf-8", errors="ignore")


def _caption_to_text(raw: str, ext: str) -> str:
    if ext == "json3" or raw.lstrip().startswith("{"):
        try:
            payload = json.loads(raw)
            lines = []
            for event in payload.get("events", []):
                segs = event.get("segs") or []
                line = "".join(seg.get("utf8", "") for seg in segs).strip()
                if line:
                    lines.append(line)
            return _clean_caption_text(" ".join(lines))
        except json.JSONDecodeError:
            pass

    lines = []
    for line in raw.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("WEBVTT") or stripped.startswith("Kind:"):
            continue
        if "-->" in stripped or stripped.isdigit():
            continue
        lines.append(stripped)
    return _clean_caption_text(" ".join(lines))


def _clean_caption_text(text: str) -> str:
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\[[^\]]+\]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()
