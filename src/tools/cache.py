from __future__ import annotations

import hashlib
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Union
from urllib.parse import parse_qs, urlparse

from src.utils import get_config

# Shared yt-dlp base options required for YouTube JS challenge solving.
# js_runtimes needs Node.js installed; remote_components fetches the EJS solver once.
YDL_BASE_OPTS: dict = {
    "quiet": True,
    "no_warnings": True,
    "noplaylist": True,
    "js_runtimes": {"node": {}},
    "remote_components": ["ejs:github"],
}


def ensure_data_dirs() -> None:
    for path in [
        "data/cache/videos",
        "data/cache/frames",
        "data/cache/audio",
        "data/cache/transcripts",
        "data/logs",
        "data/outputs",
    ]:
        Path(path).mkdir(parents=True, exist_ok=True)


def video_id_from_url(url: str) -> str:
    parsed = urlparse(url)
    if parsed.hostname and "youtu" in parsed.hostname:
        if parsed.hostname.endswith("youtu.be"):
            candidate = parsed.path.strip("/").split("/")[0]
            if candidate:
                return sanitize_id(candidate)
        query_id = parse_qs(parsed.query).get("v", [None])[0]
        if query_id:
            return sanitize_id(query_id)
        shorts = re.search(r"/shorts/([^/?#]+)", parsed.path)
        if shorts:
            return sanitize_id(shorts.group(1))
    return hashlib.sha1(url.encode("utf-8")).hexdigest()[:16]


def sanitize_id(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]", "_", value)


def is_cache_fresh(path: Union[str, Path]) -> bool:
    p = Path(path)
    if not p.exists():
        return False
    ttl_days = get_config().get("cache", {}).get("ttl_days", 30)
    cutoff = datetime.now() - timedelta(days=ttl_days)
    modified = datetime.fromtimestamp(p.stat().st_mtime)
    return modified >= cutoff


def video_cache_path(video_id: str) -> Path:
    return Path("data/cache/videos") / f"{video_id}.mp4"


def audio_cache_path(video_id: str) -> Path:
    return Path("data/cache/audio") / f"{video_id}.wav"


def frames_cache_dir(video_id: str) -> Path:
    return Path("data/cache/frames") / video_id


def transcript_cache_path(video_id: str) -> Path:
    return Path("data/cache/transcripts") / f"{video_id}.json"
