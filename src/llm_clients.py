import json
import logging
import time
from pathlib import Path
from typing import Optional

import anthropic
from google import genai
from google.genai import types as gtypes
from openai import OpenAI
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from src.utils import compute_cost, get_config, get_env

logger = logging.getLogger("tara.llm")

_gemini_client: Optional[genai.Client] = None
_anthropic_client: Optional[anthropic.Anthropic] = None
_openai_client: Optional[OpenAI] = None


def _get_gemini() -> genai.Client:
    global _gemini_client
    if _gemini_client is None:
        _gemini_client = genai.Client(api_key=get_env("GEMINI_API_KEY"))
    return _gemini_client


def _get_anthropic() -> anthropic.Anthropic:
    global _anthropic_client
    if _anthropic_client is None:
        _anthropic_client = anthropic.Anthropic(api_key=get_env("ANTHROPIC_API_KEY"))
    return _anthropic_client


def _get_openai() -> OpenAI:
    global _openai_client
    if _openai_client is None:
        _openai_client = OpenAI(api_key=get_env("OPENAI_API_KEY"))
    return _openai_client


def _parse_json_response(text: str, context: str = "") -> dict:
    """Strip markdown fences and parse JSON. Raises ValueError on failure."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        cleaned = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise ValueError(f"JSON parse failed{' in ' + context if context else ''}: {e}\nRaw: {text[:500]}")


def _extract_gemini_thinking(response) -> tuple[str, str]:
    """Split Gemini response into (response_text, thinking_text). Works with or without thinking enabled."""
    thinking_parts: list[str] = []
    response_parts: list[str] = []
    try:
        for part in response.candidates[0].content.parts:
            if getattr(part, "thought", False):
                thinking_parts.append(part.text or "")
            else:
                response_parts.append(part.text or "")
    except Exception:
        pass

    text = "".join(response_parts) or (response.text or "")
    thinking = "".join(thinking_parts)
    return text, thinking


_GEMINI_THINKING_CONFIG = gtypes.GenerateContentConfig(
    thinking_config=gtypes.ThinkingConfig(
        include_thoughts=True,
        thinking_budget=8192,
    )
)


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=2, min=2, max=60),
    retry=retry_if_exception_type(Exception),
    reraise=True,
)
def call_gemini_text(
    model: str,
    prompt: str,
    images: Optional[list[str]] = None,
    context: str = "",
) -> tuple[dict, float, str]:
    """Call Gemini with text + optional image paths. Returns (parsed_json, cost_usd, thinking_text)."""
    client = _get_gemini()

    parts: list = []
    if images:
        for img_path in images:
            image_data = Path(img_path).read_bytes()
            parts.append(gtypes.Part.from_bytes(data=image_data, mime_type="image/jpeg"))
    parts.append(prompt)

    response = client.models.generate_content(
        model=model,
        contents=parts,
        config=_GEMINI_THINKING_CONFIG,
    )
    text, thinking = _extract_gemini_thinking(response)

    try:
        input_tokens = response.usage_metadata.prompt_token_count or 0
        output_tokens = response.usage_metadata.candidates_token_count or 0
    except Exception:
        input_tokens = len(prompt) // 4
        output_tokens = len(text) // 4

    cost = compute_cost(model, input_tokens, output_tokens)
    logger.debug(f"Gemini {context}: {input_tokens}in/{output_tokens}out tokens, ${cost:.4f}, thinking={len(thinking)} chars")

    parsed = _parse_json_response(text, context)
    return parsed, cost, thinking


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=2, min=2, max=60),
    retry=retry_if_exception_type(Exception),
    reraise=True,
)
def call_gemini_video(
    model: str,
    prompt: str,
    video_path: str,
    context: str = "",
) -> tuple[dict, float, str]:
    """Call Gemini with a video file (for narrator). Returns (parsed_json, cost_usd, thinking_text)."""
    client = _get_gemini()

    video_file = client.files.upload(
        file=video_path,
        config=gtypes.UploadFileConfig(mime_type="video/mp4"),
    )
    logger.debug(f"Uploaded video to Gemini Files API: {video_file.name}")

    while video_file.state.name == "PROCESSING":
        time.sleep(5)
        video_file = client.files.get(name=video_file.name)

    if video_file.state.name == "FAILED":
        raise RuntimeError(f"Gemini video processing failed for {video_path}")

    response = client.models.generate_content(
        model=model,
        contents=[video_file, prompt],
        config=_GEMINI_THINKING_CONFIG,
    )
    text, thinking = _extract_gemini_thinking(response)

    try:
        input_tokens = response.usage_metadata.prompt_token_count or 0
        output_tokens = response.usage_metadata.candidates_token_count or 0
    except Exception:
        input_tokens = len(prompt) // 4
        output_tokens = len(text) // 4

    cost = compute_cost(model, input_tokens, output_tokens)
    logger.debug(f"Gemini video {context}: {input_tokens}in/{output_tokens}out tokens, ${cost:.4f}, thinking={len(thinking)} chars")

    parsed = _parse_json_response(text, context)
    return parsed, cost, thinking


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=2, min=2, max=60),
    retry=retry_if_exception_type(Exception),
    reraise=True,
)
def call_claude_text(
    model: str,
    prompt: str,
    images: Optional[list[str]] = None,
    context: str = "",
) -> tuple[dict, float, str]:
    """Call Claude with text + optional image paths. Returns (parsed_json, cost_usd, thinking_text)."""
    import base64

    client = _get_anthropic()

    content: list = []
    if images:
        for img_path in images:
            img_b64 = base64.standard_b64encode(Path(img_path).read_bytes()).decode("utf-8")
            content.append({
                "type": "image",
                "source": {"type": "base64", "media_type": "image/jpeg", "data": img_b64},
            })
    content.append({"type": "text", "text": prompt})

    response = client.messages.create(
        model=model,
        max_tokens=16000,
        thinking={"type": "enabled", "budget_tokens": 8000},
        messages=[{"role": "user", "content": content}],
    )

    thinking_text = ""
    response_text = ""
    for block in response.content:
        if block.type == "thinking":
            thinking_text += (block.thinking or "")
        elif block.type == "text":
            response_text += (block.text or "")

    input_tokens = response.usage.input_tokens
    output_tokens = response.usage.output_tokens
    cost = compute_cost(model, input_tokens, output_tokens)
    logger.debug(f"Claude {context}: {input_tokens}in/{output_tokens}out tokens, ${cost:.4f}, thinking={len(thinking_text)} chars")

    parsed = _parse_json_response(response_text, context)
    return parsed, cost, thinking_text


def call_whisper_api(audio_path: str) -> tuple[str, str, float]:
    """Transcribe audio via OpenAI Whisper API. Returns (text, language, cost_usd)."""
    client = _get_openai()
    model = get_config()["models"]["whisper"]

    with open(audio_path, "rb") as f:
        response = client.audio.transcriptions.create(
            model=model,
            file=f,
            response_format="verbose_json",
        )

    duration_seconds = getattr(response, "duration", 0) or 0
    cost = compute_cost(model, audio_minutes=duration_seconds / 60.0)
    text = response.text or ""
    language = getattr(response, "language", "unknown") or "unknown"
    logger.debug(f"Whisper: {len(text.split())} words, ${cost:.4f}")
    return text, language, cost
