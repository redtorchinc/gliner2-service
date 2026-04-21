"""Service configuration loaded from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# Load .env from the repo root if present
_repo_root = Path(__file__).resolve().parent.parent
_env_file = _repo_root / ".env"
if _env_file.is_file():
    from dotenv import load_dotenv
    load_dotenv(_env_file)


def _bool(val: str) -> bool:
    return val.strip().lower() in ("1", "true", "yes")


@dataclass(frozen=True)
class Settings:
    model: str = field(default_factory=lambda: os.getenv("GLINER2_MODEL", "fastino/gliner2-base-v1"))
    device: str = field(default_factory=lambda: os.getenv("GLINER2_DEVICE", "auto"))
    quantize: bool = field(default_factory=lambda: _bool(os.getenv("GLINER2_QUANTIZE", "false")))
    compile: bool = field(default_factory=lambda: _bool(os.getenv("GLINER2_COMPILE", "false")))
    host: str = field(default_factory=lambda: os.getenv("GLINER2_HOST", "127.0.0.1"))
    port: int = field(default_factory=lambda: int(os.getenv("GLINER2_PORT", "8077")))
    max_text_chars: int = field(default_factory=lambda: int(os.getenv("GLINER2_MAX_TEXT_CHARS", "200000")))
    max_batch_size: int = field(default_factory=lambda: int(os.getenv("GLINER2_MAX_BATCH_SIZE", "64")))
    api_key: str | None = field(default_factory=lambda: os.getenv("GLINER2_API_KEY"))
    log_level: str = field(default_factory=lambda: os.getenv("GLINER2_LOG_LEVEL", "INFO"))


settings = Settings()
