from datetime import datetime
from typing import Optional, Literal
from pydantic import BaseModel, Field


class VideoMetadata(BaseModel):
    title: Optional[str] = None
    channel: Optional[str] = None
    channel_id: Optional[str] = None
    duration_seconds: Optional[int] = None
    language: Optional[str] = None
    view_count: Optional[int] = None
    upload_date: Optional[str] = None
    description: Optional[str] = None
    tags: list[str] = Field(default_factory=list)


class TranscriptResult(BaseModel):
    text: str
    language: str
    source: Literal["youtube_captions", "whisper_api", "unavailable"]
    word_count: int


class ProgrammaticAnalysis(BaseModel):
    pacing: Optional[dict] = None
    audio: Optional[dict] = None
    color: Optional[dict] = None
    sensory_load: Optional[float] = None


class VetoResult(BaseModel):
    vetoed: bool
    concerns: list[str] = Field(default_factory=list)
    confidence: Literal["HIGH", "MEDIUM", "LOW"]
    reasoning: str
    thinking: Optional[str] = None


class NarratorChunk(BaseModel):
    timestamp: str
    what_happens: str
    tone: str
    concerns: str


class NarratorOutput(BaseModel):
    chunks: list[NarratorChunk]
    total_chunks: int
    interval_seconds: int


class SubVariableScore(BaseModel):
    score: int = Field(ge=0, le=100)
    reasoning: str
    flags: list[str] = Field(default_factory=list)
    confidence: Literal["HIGH", "MEDIUM", "LOW"] = "HIGH"


class ScorerOutput(BaseModel):
    model_used: str
    content_safety: dict[str, SubVariableScore]
    stimulation: dict[str, SubVariableScore]
    education: dict[str, SubVariableScore]
    channel_level: dict[str, SubVariableScore]
    overall_score: float
    overall_confidence: Literal["HIGH", "MEDIUM", "LOW"]
    notes: Optional[str] = None
    thinking: Optional[str] = None


class GoldReference(BaseModel):
    id: str
    title: str
    similarity: float
    human_scores: dict[str, int]
    rater_notes: Optional[str] = None


class CriticOutput(BaseModel):
    critic_version: str
    model_used: str
    reconciled_at: datetime
    reconciled_scores: dict
    dimension_scores: dict[str, float]
    qualitative_review: dict
    scorer_comparison: dict
    overall: dict
    metadata: dict
    bootstrap_mode: bool = False
    thinking: Optional[str] = None


class FinalVerdict(BaseModel):
    verdict: Literal["APPROVED", "REVIEW", "REJECTED", "ERROR"]
    overall_score: float
    confidence: Literal["HIGH", "MEDIUM", "LOW"]
    reasoning: str
    key_decision_drivers: list[str]
    error_reason: Optional[str] = None


class CostBreakdown(BaseModel):
    veto: float = 0.0
    narrator: float = 0.0
    scorer_gemini: float = 0.0
    scorer_claude: float = 0.0
    critic: float = 0.0
    whisper: float = 0.0
    total: float = 0.0


class State(BaseModel):
    url: str
    video_id: Optional[str] = None
    started_at: datetime
    completed_at: Optional[datetime] = None

    metadata: Optional[VideoMetadata] = None
    video_path: Optional[str] = None
    audio_path: Optional[str] = None
    frame_paths: list[str] = Field(default_factory=list)
    transcript: Optional[TranscriptResult] = None

    programmatic: Optional[ProgrammaticAnalysis] = None

    veto: Optional[VetoResult] = None
    narrator: Optional[NarratorOutput] = None

    embedding: Optional[list[float]] = None
    similar_gold: list[GoldReference] = Field(default_factory=list)

    scorer_gemini: Optional[ScorerOutput] = None
    scorer_claude: Optional[ScorerOutput] = None

    critic: Optional[CriticOutput] = None
    final_verdict: Optional[FinalVerdict] = None

    errors: list[dict] = Field(default_factory=list)
    cost: CostBreakdown = Field(default_factory=CostBreakdown)
    cache_hits: list[str] = Field(default_factory=list)
    prompt_version: str = "v2.0"

    class Config:
        arbitrary_types_allowed = True
