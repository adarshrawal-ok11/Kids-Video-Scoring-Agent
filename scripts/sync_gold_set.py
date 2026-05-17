from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime as dt
from pathlib import Path

import pandas as pd
import yt_dlp
from sentence_transformers import SentenceTransformer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.tools.cache import YDL_BASE_OPTS
from src.utils import get_config

DEFAULT_INPUT = Path("data/gold_set_human.xlsx")
DEFAULT_OUTPUT = Path("data/gold_set_enriched.json")

SCORE_COLUMNS = [
    "violence_aggression", "scary_fear_inducing", "inappropriate_language",
    "bullying_exclusion", "stereotyping", "fantasy_level", "negative_behavior_modeling",
    "visual_pacing", "audio_energy", "color_intensity", "concepts_taught",
    "participation", "social_modeling", "repetition", "language_quality",
    "content_consistency", "channel_reputation",
]

YDL_OPTS = {**YDL_BASE_OPTS, "skip_download": True}


def _fetch_video_data(url: str) -> dict:
    """Fetch title, channel, description, and transcript from YouTube."""
    result: dict = {}
    try:
        with yt_dlp.YoutubeDL(YDL_OPTS) as ydl:
            info = ydl.extract_info(url, download=False)
        result["title"] = info.get("title")
        result["channel"] = info.get("channel") or info.get("uploader")
        result["duration_sec"] = info.get("duration")
        result["description"] = (info.get("description") or "")[:500]
        result["tags"] = info.get("tags") or []

        # Try to get transcript from YouTube captions
        transcript = _extract_caption_text(info)
        if transcript:
            result["transcript"] = transcript[:3000]
    except Exception as exc:
        print(f"    Warning: could not fetch data for {url}: {exc}")
    return result


def _extract_caption_text(info: dict) -> str:
    """Extract plain text from YouTube caption data."""
    import re
    import urllib.request

    caption_sets = [info.get("subtitles") or {}, info.get("automatic_captions") or {}]
    preferred_langs = ["en", "en-US", "en-GB", "hi"]
    for captions in caption_sets:
        for lang in preferred_langs:
            if lang not in captions:
                continue
            formats = captions[lang]
            fmt = next((f for f in formats if f.get("ext") in ("vtt", "srv3") and f.get("url")), None)
            if not fmt:
                continue
            try:
                with urllib.request.urlopen(fmt["url"], timeout=15) as r:
                    raw = r.read().decode("utf-8", errors="ignore")
                lines = [
                    l.strip() for l in raw.splitlines()
                    if l.strip() and "-->" not in l and not l.strip().isdigit()
                    and not l.startswith("WEBVTT") and not l.startswith("Kind:")
                ]
                text = re.sub(r"<[^>]+>", " ", " ".join(lines))
                text = re.sub(r"\s+", " ", text).strip()
                if text:
                    return text
            except Exception:
                continue
    return ""


def main() -> None:
    parser = argparse.ArgumentParser(description="Sync gold-set: auto-fill metadata, build embeddings, write JSON.")
    parser.add_argument("--input", default=str(DEFAULT_INPUT), help="Human gold-set Excel path")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Enriched JSON output path")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Gold-set Excel not found: {input_path}")

    df_full = pd.read_excel(input_path)

    # Work only on real data rows (skip template header row and placeholder URLs)
    mask_valid = (
        df_full["url"].notna()
        & ~df_full["url"].astype(str).str.contains("REPLACE", case=False)
        & ~df_full["id"].astype(str).str.startswith("Unique ID")
    )
    df = df_full[mask_valid].copy()

    # Auto-fill auto_title and auto_channel from yt-dlp where missing
    changed = False
    for idx, row in df.iterrows():
        needs_title = pd.isna(row.get("auto_title")) or not str(row.get("auto_title", "")).strip()
        needs_channel = pd.isna(row.get("auto_channel")) or not str(row.get("auto_channel", "")).strip()
        if needs_title or needs_channel:
            print(f"  Fetching metadata for {row['id']} ({row['url']}) ...")
            meta = _fetch_video_data(str(row["url"]))
            if meta.get("title") and needs_title:
                df_full.at[idx, "auto_title"] = meta["title"]
                df.at[idx, "auto_title"] = meta["title"]
                changed = True
            if meta.get("channel") and needs_channel:
                df_full.at[idx, "auto_channel"] = meta["channel"]
                df.at[idx, "auto_channel"] = meta["channel"]
                changed = True
            if meta.get("duration_sec") and pd.isna(row.get("auto_duration_sec")):
                df_full.at[idx, "auto_duration_sec"] = str(int(meta["duration_sec"]))
                changed = True
            # Store transcript and description for embedding (not in Excel, kept in memory)
            df.at[idx, "_transcript"] = meta.get("transcript", "")
            df.at[idx, "_description"] = meta.get("description", "")
            df.at[idx, "_tags"] = ", ".join(meta.get("tags") or [])

    if changed:
        df_full.to_excel(input_path, index=False)
        print(f"  Updated Excel with auto-filled metadata: {input_path}")

    # Build embeddings and enriched JSON
    cfg = get_config()
    model = SentenceTransformer(cfg["embedding"]["model"])
    videos = []
    for _, row in df.iterrows():
        item = _row_to_video(row)
        text = _embedding_text(item)
        if not text.strip():
            print(f"  Skipping {item['id']} — no text to embed")
            continue
        embedding = model.encode(text, normalize_embeddings=bool(cfg["embedding"].get("normalize", True)))
        item["embedding"] = [float(v) for v in embedding.tolist()]
        videos.append(item)
        label = (item['title'] or item['url']).encode('ascii', errors='replace').decode('ascii')
        print(f"  Embedded: {item['id']} | {label}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps({"videos": videos}, indent=2, default=_json_default),
        encoding="utf-8",
    )
    print(f"Wrote {len(videos)} enriched gold videos to {output_path}")
    anchors = sum(1 for v in videos if v.get("is_anchor"))
    print(f"  Anchors (scorer calibration): {anchors}")
    print(f"  Holdout (critic calibration): {len(videos) - anchors}")


def _row_to_video(row: pd.Series) -> dict:
    data = {str(k).strip(): _clean_value(v) for k, v in row.items()}
    scores = {col: int(data[col]) for col in SCORE_COLUMNS if data.get(col) is not None}
    return {
        "id": str(data.get("id") or data.get("url") or ""),
        "url": data.get("url"),
        "title": data.get("auto_title") or data.get("title"),
        "channel": data.get("auto_channel") or data.get("channel"),
        "is_anchor": bool(data.get("is_anchor", False)),
        "verdict": data.get("verdict"),
        "overall_estimate": data.get("overall_estimate"),
        "human_scores": scores,
        "rater_notes": data.get("notes") or data.get("rater_notes"),
        "transcript": data.get("_transcript") or "",
        "description": data.get("_description") or "",
        "tags": data.get("_tags") or "",
        "raw": data,
    }


def _embedding_text(item: dict) -> str:
    """Mirror build_embedding_text() from src/tools/embeddings.py for consistency."""
    parts = []
    if item.get("title"):
        parts.append(f"Title: {item['title']}")
    if item.get("channel"):
        parts.append(f"Channel: {item['channel']}")
    if item.get("description"):
        parts.append(f"Description: {item['description']}")
    if item.get("tags"):
        parts.append(f"Tags: {item['tags']}")
    if item.get("transcript"):
        parts.append(f"Transcript: {item['transcript']}")
    if item.get("rater_notes"):
        parts.append(f"Notes: {item['rater_notes']}")
    return "\n".join(parts)


def _json_default(obj):
    if isinstance(obj, (dt, date)):
        return obj.isoformat()
    raise TypeError(f"Not serializable: {type(obj)}")


def _clean_value(value):
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


if __name__ == "__main__":
    main()
