# TARA — Trustworthy AI Review Agent for Kids' Videos

TARA is an AI-powered pipeline that automatically scores YouTube videos for safety and suitability for Indian children aged **3–5 years**. It combines multiple vision-language models, programmatic signal analysis, and a human-calibrated gold set to produce a structured, auditable verdict on every video.

---

## Why TARA?

Parents and platforms struggle to verify whether a children's video is truly safe — content moderation at scale misses cultural nuance, stimulation levels, and educational quality. TARA brings together:

- **Multi-model scoring** — Gemini and Claude independently score every video
- **A senior critic model** — Gemini 2.5 Pro reconciles scores against human-rated references
- **Chain-of-thought reasoning** — logged for every decision, fully auditable
- **Cultural calibration** — tuned for Indian kids: Hindi/Hinglish content, Indian cultural norms, appropriate fantasy levels

---

## Pipeline Architecture

```
YouTube URL
    │
    ▼
┌─────────────┐
│  Download   │  yt-dlp — video, audio, transcript
└──────┬──────┘
       │
    ┌──▼──────────────────┐
    │  Programmatic       │  Scene pacing · Audio energy · Colour intensity
    └──┬──────────────────┘
       │
    ┌──▼──────────────────┐
    │  Veto Agent         │  Gemini 2.5 Flash — hard-fail on explicit content
    └──┬──────────────────┘
       │
    ┌──▼──────────────────┐
    │  Narrator Agent     │  Gemini 2.5 Flash — timestamped play-by-play of full video
    └──┬──────────────────┘
       │
    ┌──▼──────────────────┐
    │  Gold Retrieval     │  Cosine similarity against human-rated gold set
    └──┬──────────────────┘
       │
    ┌──▼──────────┐  ┌──▼──────────┐
    │Scorer Gemini│  │Scorer Claude│  Independent scoring — chain-of-thought
    │  3 Flash    │  │  Sonnet 4.6 │  reasoning logged for every sub-variable
    └──────┬──────┘  └──────┬──────┘
           └────────┬───────┘
                    │
            ┌───────▼───────┐
            │  Critic Agent │  Gemini 2.5 Pro — reconciles both scorers,
            │               │  calibrates against similar gold-set videos
            └───────┬───────┘
                    │
            ┌───────▼───────┐
            │    Verdict    │  APPROVED / REVIEW / REJECTED
            └───────────────┘
                    │
            ┌───────▼───────┐
            │ Excel + SQLite│  9-sheet report · Full audit trail
            └───────────────┘
```

---

## Scoring Dimensions

Each video is scored across **18 sub-variables** in 4 dimensions:

| Dimension | Weight | Sub-variables |
|---|---|---|
| **Content Safety** | 25% | Violence, scary content, inappropriate language, bullying, stereotyping, fantasy level, negative behaviour modelling |
| **Stimulation** | 20% | Visual pacing, audio energy, colour intensity, sensory load |
| **Educational Value** | 20% | Concepts taught, viewer participation, social modelling, repetition, language quality |
| **Channel Level** | 10% | Content consistency, channel reputation |
| **Consistency Bonus** | 25% | Applied when all dimensions score > 60 |

Scores are 0–100. Verdict thresholds:

| Verdict | Condition |
|---|---|
| **APPROVED** | Overall ≥ 70, Content Safety ≥ 60, all CS sub-vars ≥ 40 |
| **REJECTED** | Overall < 50 or any CS sub-var < 20 |
| **REVIEW** | Everything else — human review recommended |

---

## Models Used

| Agent | Model | Role |
|---|---|---|
| Veto | Gemini 2.5 Flash | Hard-fail check on explicit content |
| Narrator | Gemini 2.5 Flash | Full video understanding — timestamped |
| Scorer Gemini | Gemini 3 Flash Preview | Primary scorer with chain-of-thought |
| Scorer Claude | Claude Sonnet 4.6 | Independent scorer with extended thinking |
| Critic | Gemini 2.5 Pro | Score reconciliation + gold calibration |
| Transcription | OpenAI Whisper | Fallback when captions unavailable |

---

## Output

Every run produces:

- **Excel workbook** (9 sheets): Summary · Content Safety · Education · Stimulation · Channel Level · Narrator Timeline · Critic Findings · Disagreements · Thinking Logs
- **SQLite database**: Full audit trail of every analysis with channel-level aggregation
- **JSON summary** printed to console

---

## Setup

### Prerequisites

- Python 3.12+
- [uv](https://github.com/astral-sh/uv) — fast Python package manager
- Node.js v20+ — required by yt-dlp for YouTube access
- ffmpeg on your system PATH

### Installation

```bash
git clone https://github.com/your-username/tara-kids-video-scoring.git
cd tara-kids-video-scoring

# Create virtual environment and install dependencies
uv venv
uv pip install -r requirements.txt
```

### API Keys

Create a `.env` file in the project root:

```env
GEMINI_API_KEY=your_gemini_api_key
ANTHROPIC_API_KEY=your_anthropic_api_key
OPENAI_API_KEY=your_openai_api_key   # optional — only needed if YouTube captions unavailable
```

Get your keys from:
- Gemini: [Google AI Studio](https://aistudio.google.com)
- Anthropic: [Anthropic Console](https://console.anthropic.com)
- OpenAI: [OpenAI Platform](https://platform.openai.com) *(optional)*

---

## Running TARA

### Analyse a single video

```bash
# Windows
.venv\Scripts\python.exe scripts/run_single.py "https://www.youtube.com/watch?v=VIDEO_ID" --excel

# Mac/Linux
.venv/bin/python scripts/run_single.py "https://www.youtube.com/watch?v=VIDEO_ID" --excel
```

Example output:
```json
{
  "url": "https://www.youtube.com/watch?v=e_04ZrNroTo",
  "title": "Wheels on the Bus | CoComelon Nursery Rhymes",
  "verdict": "APPROVED",
  "overall_score": 86.0,
  "errors": 0,
  "cost_usd": 0.17
}
```

### Analyse a batch

```bash
.venv\Scripts\python.exe scripts/run_batch.py urls.txt --excel
```

---

## Human Gold Set (Calibration)

TARA uses a human-rated gold set to calibrate AI scoring. You maintain this as a simple Excel file at `data/gold_set_human.xlsx`.

### How to add videos

1. Open `data/gold_set_human.xlsx`
2. Add a YouTube URL and your human ratings for each sub-variable (0–100 scale)
3. Set `verdict` to APPROVED / REVIEW / REJECTED
4. Add any notes explaining your rating
5. Mark your best-calibrated videos as `is_anchor = True` (up to 5)
6. Run the sync script:

```bash
.venv\Scripts\python.exe scripts/sync_gold_set.py
```

The sync script automatically:
- Fetches video title, channel, and transcript from YouTube
- Writes them back into the Excel file
- Builds semantic embeddings for similarity matching at runtime

### How calibration works at runtime

- **Anchor videos** (`is_anchor = True`) → embedded directly in scorer prompts as calibration examples. Scorers see: *"a human rated this similar video 25/100 for education — calibrate accordingly."*
- **Non-anchor videos** → retrieved by cosine similarity and passed to the critic as reference context.

The more videos you add to the gold set, the more accurate and culturally calibrated the scores become.

---

## Project Structure

```
├── config.yaml                 # Model selection, weights, thresholds
├── data/
│   ├── gold_set_human.xlsx     # Your human-rated calibration videos
│   ├── gold_set_enriched.json  # Auto-generated — embeddings + metadata
│   ├── cache/                  # Downloaded videos, frames, audio, transcripts
│   ├── outputs/                # Excel reports per run
│   └── tara_results.db         # SQLite audit trail
├── prompts/                    # LLM prompt templates (veto, narrator, scorer, critic)
├── scripts/
│   ├── run_single.py           # Analyse one video
│   ├── run_batch.py            # Analyse many videos from a text file
│   └── sync_gold_set.py        # Rebuild gold set JSON from Excel
└── src/
    ├── agents/                 # Veto · Narrator · Scorer · Critic
    ├── tools/                  # Download · Frames · Audio · Embeddings · Persist
    ├── state.py                # Pydantic state model flowing through pipeline
    ├── main.py                 # Pipeline orchestration
    └── verdict.py              # Threshold-based verdict logic
```

---

## Cost Estimates

Approximate cost per video on paid API tier:

| Component | Typical cost |
|---|---|
| Gemini 3 Flash — scoring | ~$0.02 |
| Claude Sonnet 4.6 — scoring | ~$0.08 |
| Gemini 2.5 Pro — critic | ~$0.06 |
| Gemini 2.5 Flash — veto + narrator | ~$0.01 |
| **Total per video** | **~$0.15–0.20** |

Videos are cached locally after first download — re-running the same URL costs only the LLM calls (~$0.17).

---

## Key Design Decisions

- **Two independent scorers** prevent single-model bias; divergence > 15 points on any sub-variable is flagged in the Excel Disagreements sheet
- **Narrator agent** watches the full video so downstream agents have a complete temporal picture, not just sampled frames
- **Never crashes** — every pipeline step is wrapped in error handling; partial results are saved even if one agent fails
- **Fully auditable** — every agent's chain-of-thought reasoning is stored in the Thinking Logs sheet

---

## Contributing

This project is in active development. If you are evaluating TARA for a children's platform or research context, feel free to open an issue or reach out.
