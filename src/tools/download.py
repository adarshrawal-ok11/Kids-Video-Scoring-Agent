from __future__ import annotations

from pathlib import Path

import yt_dlp

from src.state import State
from src.tools.cache import YDL_BASE_OPTS, ensure_data_dirs, is_cache_fresh, video_cache_path, video_id_from_url


def download_video(state: State) -> State:
    ensure_data_dirs()
    video_id = state.video_id or video_id_from_url(state.url)
    state.video_id = video_id
    target = video_cache_path(video_id)

    if is_cache_fresh(target):
        state.video_path = str(target)
        state.cache_hits.append("video")
        return state

    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_template = str(target.with_suffix(".%(ext)s"))
    opts = {
        **YDL_BASE_OPTS,
        "format": "bestvideo[ext=mp4][height<=720]+bestaudio[ext=m4a]/best[ext=mp4][height<=720]/best",
        "outtmpl": tmp_template,
        "merge_output_format": "mp4",
    }
    with yt_dlp.YoutubeDL(opts) as ydl:
        ydl.download([state.url])

    if not target.exists():
        candidates = sorted(target.parent.glob(f"{video_id}.*"))
        mp4_candidates = [p for p in candidates if p.suffix.lower() == ".mp4"]
        if mp4_candidates:
            mp4_candidates[0].replace(target)

    if not target.exists():
        raise FileNotFoundError(f"yt-dlp did not produce expected video file: {target}")

    state.video_path = str(Path(target))
    return state
