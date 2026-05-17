import logging
import os
from pathlib import Path
from datetime import datetime

import yaml
from dotenv import load_dotenv

load_dotenv()

_config: dict = {}
_logger: logging.Logger = logging.getLogger("tara")


def load_config(config_path: str = "config.yaml") -> dict:
    global _config
    if not _config:
        with open(config_path) as f:
            _config = yaml.safe_load(f)
    return _config


def get_config() -> dict:
    if not _config:
        return load_config()
    return _config


def setup_logging(level: str = "INFO") -> None:
    log_dir = Path("data/logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"tara_{datetime.now().strftime('%Y%m%d')}.log"

    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(),
        ],
    )


def log(message: str) -> None:
    _logger.info(message)


def log_error(step: str, exc: Exception) -> None:
    _logger.error(f"Step '{step}' failed: {type(exc).__name__}: {exc}", exc_info=True)


def get_env(key: str, required: bool = True) -> str:
    val = os.getenv(key)
    if required and not val:
        raise EnvironmentError(f"Missing required environment variable: {key}")
    return val or ""


PRICING = {
    "gemini-3-flash-preview": {"input": 0.50 / 1_000_000, "output": 3.0 / 1_000_000},
    "gemini-2.0-flash": {"input": 0.10 / 1_000_000, "output": 0.40 / 1_000_000},
    "gemini-2.5-flash": {"input": 0.075 / 1_000_000, "output": 0.30 / 1_000_000},
    "gemini-2.5-pro": {"input": 1.25 / 1_000_000, "output": 10.0 / 1_000_000},
    "claude-sonnet-4-6": {"input": 3.0 / 1_000_000, "output": 15.0 / 1_000_000},
    "claude-opus-4-7": {"input": 15.0 / 1_000_000, "output": 75.0 / 1_000_000},
    "whisper-1": {"per_minute": 0.006},
}


def compute_cost(model: str, input_tokens: int = 0, output_tokens: int = 0, audio_minutes: float = 0.0) -> float:
    p = PRICING.get(model)
    if not p:
        return 0.0
    if "per_minute" in p:
        return audio_minutes * p["per_minute"]
    return input_tokens * p["input"] + output_tokens * p["output"]
