from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional, Union

import pandas as pd

from src.state import State

DB_PATH = Path("data/tara_results.db")


def save_to_sqlite(state: State) -> State:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    try:
        with sqlite3.connect(DB_PATH) as conn:
            _ensure_schema(conn)
            conn.execute(
                """
                INSERT INTO analyses (
                    url, video_id, title, channel, channel_id, duration_seconds, language,
                    started_at, completed_at, total_duration_seconds, final_verdict,
                    overall_score, confidence, error_reason, content_safety_score,
                    stimulation_score, education_score, channel_level_score, vetoed,
                    veto_concerns, full_state_json, prompt_version, total_cost_usd,
                    bootstrap_mode, has_errors, error_count
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                _analysis_row(state),
            )
            _upsert_channel(conn, state)
    except Exception:
        fallback = Path("data/outputs") / f"sqlite_fallback_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        fallback.parent.mkdir(parents=True, exist_ok=True)
        fallback.write_text(state.model_dump_json(indent=2), encoding="utf-8")
        raise
    return state


def export_excel(states: Iterable[State], output_path: Optional[Union[str, Path]] = None) -> Path:
    state_list = list(states)
    out_dir = Path("data/outputs")
    out_dir.mkdir(parents=True, exist_ok=True)
    path = Path(output_path) if output_path else out_dir / f"tara_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"

    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        pd.DataFrame([_summary_row(s) for s in state_list]).to_excel(writer, sheet_name="Summary", index=False)
        pd.DataFrame(_dimension_rows(state_list, "content_safety")).to_excel(writer, sheet_name="Content Safety detail", index=False)
        pd.DataFrame(_dimension_rows(state_list, "education")).to_excel(writer, sheet_name="Education detail", index=False)
        pd.DataFrame(_stimulation_rows(state_list)).to_excel(writer, sheet_name="Stimulation detail", index=False)
        pd.DataFrame(_dimension_rows(state_list, "channel_level")).to_excel(writer, sheet_name="Channel Level detail", index=False)
        pd.DataFrame(_narrator_rows(state_list)).to_excel(writer, sheet_name="Narrator Timeline", index=False)
        pd.DataFrame([_critic_row(s) for s in state_list]).to_excel(writer, sheet_name="Critic Findings", index=False)
        pd.DataFrame(_disagreement_rows(state_list)).to_excel(writer, sheet_name="Disagreements", index=False)
        pd.DataFrame(_thinking_rows(state_list)).to_excel(writer, sheet_name="Thinking Logs", index=False)
    return path


def _ensure_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS analyses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            url TEXT NOT NULL,
            video_id TEXT,
            title TEXT,
            channel TEXT,
            channel_id TEXT,
            duration_seconds INTEGER,
            language TEXT,
            started_at DATETIME,
            completed_at DATETIME,
            total_duration_seconds REAL,
            final_verdict TEXT,
            overall_score REAL,
            confidence TEXT,
            error_reason TEXT,
            content_safety_score REAL,
            stimulation_score REAL,
            education_score REAL,
            channel_level_score REAL,
            vetoed BOOLEAN,
            veto_concerns TEXT,
            full_state_json TEXT,
            prompt_version TEXT,
            total_cost_usd REAL,
            bootstrap_mode BOOLEAN,
            has_errors BOOLEAN,
            error_count INTEGER
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_analyses_url ON analyses(url)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_analyses_channel ON analyses(channel)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_analyses_verdict ON analyses(final_verdict)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_analyses_timestamp ON analyses(started_at)")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS channels (
            channel_id TEXT PRIMARY KEY,
            channel_name TEXT,
            n_videos_analyzed INTEGER DEFAULT 0,
            avg_content_safety REAL,
            avg_stimulation REAL,
            avg_education REAL,
            avg_channel_level REAL,
            avg_overall REAL,
            last_analyzed DATETIME,
            first_analyzed DATETIME
        )
        """
    )


def _analysis_row(state: State) -> tuple:
    metadata = state.metadata
    final = state.final_verdict
    dims = state.critic.dimension_scores if state.critic else {}
    completed = state.completed_at or datetime.now()
    duration = (completed - state.started_at).total_seconds()
    return (
        state.url,
        state.video_id,
        metadata.title if metadata else None,
        metadata.channel if metadata else None,
        metadata.channel_id if metadata else None,
        metadata.duration_seconds if metadata else None,
        metadata.language if metadata else None,
        state.started_at.isoformat(),
        completed.isoformat(),
        duration,
        final.verdict if final else None,
        final.overall_score if final else None,
        final.confidence if final else None,
        final.error_reason if final else None,
        dims.get("content_safety"),
        dims.get("stimulation"),
        dims.get("education"),
        dims.get("channel_level"),
        bool(state.veto.vetoed) if state.veto else None,
        json.dumps(state.veto.concerns if state.veto else []),
        state.model_dump_json(),
        state.prompt_version,
        state.cost.total,
        bool(state.critic.bootstrap_mode) if state.critic else True,
        bool(state.errors),
        len(state.errors),
    )


def _upsert_channel(conn: sqlite3.Connection, state: State) -> None:
    metadata = state.metadata
    if not metadata or not metadata.channel_id:
        return
    dims = state.critic.dimension_scores if state.critic else {}
    overall = state.final_verdict.overall_score if state.final_verdict else None
    conn.execute(
        """
        INSERT INTO channels (
            channel_id, channel_name, n_videos_analyzed, avg_content_safety,
            avg_stimulation, avg_education, avg_channel_level, avg_overall,
            last_analyzed, first_analyzed
        ) VALUES (?, ?, 1, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(channel_id) DO UPDATE SET
            channel_name=excluded.channel_name,
            n_videos_analyzed=n_videos_analyzed + 1,
            avg_content_safety=excluded.avg_content_safety,
            avg_stimulation=excluded.avg_stimulation,
            avg_education=excluded.avg_education,
            avg_channel_level=excluded.avg_channel_level,
            avg_overall=excluded.avg_overall,
            last_analyzed=excluded.last_analyzed
        """,
        (
            metadata.channel_id,
            metadata.channel,
            dims.get("content_safety"),
            dims.get("stimulation"),
            dims.get("education"),
            dims.get("channel_level"),
            overall,
            datetime.now().isoformat(),
            datetime.now().isoformat(),
        ),
    )


def _summary_row(state: State) -> dict:
    metadata = state.metadata
    final = state.final_verdict
    critic = state.critic
    dims = critic.dimension_scores if critic else {}
    return {
        "URL": state.url,
        "Title": metadata.title if metadata else None,
        "Channel": metadata.channel if metadata else None,
        "Duration (min)": round((metadata.duration_seconds or 0) / 60, 2) if metadata else None,
        "Language": metadata.language if metadata else None,
        "Verdict": final.verdict if final else None,
        "Overall Score": final.overall_score if final else None,
        "Critic Confidence": final.confidence if final else None,
        "CS Dimension": dims.get("content_safety"),
        "SL Dimension": dims.get("stimulation"),
        "EV Dimension": dims.get("education"),
        "CL Dimension": dims.get("channel_level"),
        "Vetoed": state.veto.vetoed if state.veto else None,
        "Veto Concerns": _join_list(state.veto.concerns) if state.veto else "",
        "Adjustments Made by Critic": (critic.scorer_comparison or {}).get("adjustments_made") if critic else None,
        "Bootstrap Mode": critic.bootstrap_mode if critic else True,
        "Has Errors": bool(state.errors),
        "Cost (USD)": round(state.cost.total, 4),
        "Duration (sec)": round(((state.completed_at or datetime.now()) - state.started_at).total_seconds(), 2),
        "Started At": state.started_at.isoformat(),
        "Notes": final.reasoning if final else "",
    }


def _dimension_rows(states: list[State], dimension: str) -> list[dict]:
    rows = []
    for state in states:
        metadata = state.metadata
        reconciled = (state.critic.reconciled_scores.get(dimension, {}) if state.critic else {})
        for subvar, data in reconciled.items():
            rows.append({
                "URL": state.url,
                "Title": metadata.title if metadata else None,
                "Channel": metadata.channel if metadata else None,
                "Sub-variable": subvar,
                "Gemini Score": _subvar_score(state.scorer_gemini, dimension, subvar),
                "Claude Score": _subvar_score(state.scorer_claude, dimension, subvar),
                "Critic Reconciled": data.get("reconciled_score"),
                "Critic Reasoning": data.get("reasoning"),
                "Confidence": data.get("confidence"),
                "Calibration Flag": data.get("calibration_flag"),
                "Flags": "; ".join(data.get("flags", [])) if isinstance(data.get("flags"), list) else data.get("flags"),
            })
    return rows


def _stimulation_rows(states: list[State]) -> list[dict]:
    rows = []
    for state in states:
        metadata = state.metadata
        p = state.programmatic
        rows.append({
            "URL": state.url,
            "Title": metadata.title if metadata else None,
            "Pacing - Code cuts/min": (p.pacing or {}).get("cuts_per_minute") if p else None,
            "Pacing - Code Score": (p.pacing or {}).get("score") if p else None,
            "Audio - Code Score": (p.audio or {}).get("score") if p else None,
            "Color - Code Score": (p.color or {}).get("score") if p else None,
            "Sensory Load Composite": p.sensory_load if p else None,
            "SL Dimension Score": state.critic.dimension_scores.get("stimulation") if state.critic else None,
        })
    return rows


def _narrator_rows(states: list[State]) -> list[dict]:
    rows = []
    for state in states:
        metadata = state.metadata
        if not state.narrator:
            continue
        for idx, chunk in enumerate(state.narrator.chunks):
            rows.append({
                "URL": state.url,
                "Title": metadata.title if metadata else None,
                "Chunk Index": idx,
                "Timestamp": chunk.timestamp,
                "What Happens": chunk.what_happens,
                "Tone": chunk.tone,
                "Concerns": chunk.concerns,
            })
    return rows


def _critic_row(state: State) -> dict:
    metadata = state.metadata
    critic = state.critic
    overall = critic.overall if critic else {}
    review = critic.qualitative_review if critic else {}
    return {
        "URL": state.url,
        "Title": metadata.title if metadata else None,
        "N Sub-vars Adjusted": overall.get("n_subvars_adjusted"),
        "N Unchanged": overall.get("n_subvars_unchanged"),
        "Total Adjustment Magnitude": overall.get("total_adjustment_magnitude"),
        "Missed Concerns": _join_list(review.get("missed_concerns", [])),
        "Weak Reasoning Flags": _join_list(review.get("weak_reasoning_flags", [])),
        "Cultural Notes": review.get("cultural_notes"),
        "Production Quality Notes": review.get("production_quality_notes"),
        "Calibration Flags": _join_list(overall.get("calibration_flags", [])),
        "Critic Confidence": overall.get("critic_confidence"),
        "Key Decision Drivers": _join_list(overall.get("key_decision_drivers", [])),
    }


def _disagreement_rows(states: list[State]) -> list[dict]:
    rows = []
    for state in states:
        metadata = state.metadata
        for dimension in ["content_safety", "stimulation", "education", "channel_level"]:
            keys = set()
            if state.scorer_gemini:
                keys.update(getattr(state.scorer_gemini, dimension, {}).keys())
            if state.scorer_claude:
                keys.update(getattr(state.scorer_claude, dimension, {}).keys())
            for subvar in keys:
                gemini = _subvar_score(state.scorer_gemini, dimension, subvar)
                claude = _subvar_score(state.scorer_claude, dimension, subvar)
                if gemini is None or claude is None or abs(gemini - claude) <= 15:
                    continue
                rows.append({
                    "URL": state.url,
                    "Title": metadata.title if metadata else None,
                    "Sub-variable": subvar,
                    "Gemini Score": gemini,
                    "Claude Score": claude,
                    "Divergence": abs(gemini - claude),
                    "Critic Reconciled": _critic_subvar_score(state, dimension, subvar),
                })
    return rows


def _join_list(value) -> str:
    if not isinstance(value, list):
        return str(value) if value else ""
    return "; ".join(item if isinstance(item, str) else str(item) for item in value)


def _subvar_score(scorer, dimension: str, subvar: str) -> Optional[int]:
    if not scorer:
        return None
    bucket = getattr(scorer, dimension, {})
    value = bucket.get(subvar)
    return value.score if value else None


def _critic_subvar_score(state: State, dimension: str, subvar: str) -> Optional[int]:
    if not state.critic:
        return None
    data = state.critic.reconciled_scores.get(dimension, {}).get(subvar, {})
    return data.get("reconciled_score")


def _thinking_rows(states: list[State]) -> list[dict]:
    rows = []
    for state in states:
        metadata = state.metadata
        title = metadata.title if metadata else None
        agents = [
            ("Veto", state.veto.thinking if state.veto else None),
            ("Scorer Gemini", state.scorer_gemini.thinking if state.scorer_gemini else None),
            ("Scorer Claude", state.scorer_claude.thinking if state.scorer_claude else None),
            ("Critic", state.critic.thinking if state.critic else None),
        ]
        for agent_name, thinking in agents:
            if thinking:
                rows.append({
                    "URL": state.url,
                    "Title": title,
                    "Agent": agent_name,
                    "Thinking Text": thinking,
                })
    return rows
