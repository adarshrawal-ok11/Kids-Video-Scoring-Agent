from __future__ import annotations

import yt_dlp

from src.state import State, VideoMetadata
from src.tools.cache import YDL_BASE_OPTS, video_id_from_url


def fetch_metadata(state: State) -> State:
    opts = {**YDL_BASE_OPTS, "skip_download": True}
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(state.url, download=False)

    state.video_id = info.get("id") or video_id_from_url(state.url)
    state.metadata = VideoMetadata(
        title=info.get("title"),
        channel=info.get("channel") or info.get("uploader"),
        channel_id=info.get("channel_id") or info.get("uploader_id"),
        duration_seconds=info.get("duration"),
        language=info.get("language"),
        view_count=info.get("view_count"),
        upload_date=info.get("upload_date"),
        description=info.get("description"),
        tags=info.get("tags") or [],
    )
    return state
