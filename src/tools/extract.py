from __future__ import annotations

from pathlib import Path

import cv2
import ffmpeg

from src.state import State
from src.tools.cache import audio_cache_path, frames_cache_dir, is_cache_fresh
from src.utils import get_config


def extract_frames(state: State) -> State:
    if not state.video_path:
        raise ValueError("video_path is required before frame extraction")

    n_frames = int(get_config().get("scoring", {}).get("n_frames", 15))
    out_dir = frames_cache_dir(state.video_id or "unknown")
    out_dir.mkdir(parents=True, exist_ok=True)

    existing = sorted(out_dir.glob("frame_*.jpg"))
    if len(existing) >= n_frames and all(is_cache_fresh(p) for p in existing[:n_frames]):
        state.frame_paths = [str(p) for p in existing[:n_frames]]
        state.cache_hits.append("frames")
        return state

    cap = cv2.VideoCapture(state.video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video for frame extraction: {state.video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total_frames <= 0:
        cap.release()
        raise ValueError("Video frame count is unavailable")

    indices = _sample_frame_indices(total_frames, n_frames)
    paths: list[str] = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            continue
        frame_path = out_dir / f"frame_{idx:06d}.jpg"
        cv2.imwrite(str(frame_path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        paths.append(str(frame_path))
    cap.release()

    if not paths:
        raise ValueError("No frames could be extracted")
    state.frame_paths = paths
    return state


def extract_audio(state: State) -> State:
    if not state.video_path:
        raise ValueError("video_path is required before audio extraction")

    target = audio_cache_path(state.video_id or "unknown")
    target.parent.mkdir(parents=True, exist_ok=True)
    if is_cache_fresh(target):
        state.audio_path = str(target)
        state.cache_hits.append("audio")
        return state

    (
        ffmpeg
        .input(state.video_path)
        .output(str(target), ac=1, ar="16000", format="wav", loglevel="error")
        .overwrite_output()
        .run(capture_stdout=True, capture_stderr=True)
    )
    if not target.exists():
        raise FileNotFoundError(f"Audio extraction failed: {target}")
    state.audio_path = str(Path(target))
    return state


def _sample_frame_indices(total_frames: int, n_frames: int) -> list[int]:
    if n_frames <= 0:
        return []
    if total_frames <= n_frames:
        return list(range(total_frames))
    indices = {0, total_frames // 2, total_frames - 1}
    remaining = max(n_frames - len(indices), 0)
    step = max(total_frames // (remaining + 1), 1)
    for i in range(remaining):
        indices.add(min(step * (i + 1), total_frames - 1))
    return sorted(indices)[:n_frames]
