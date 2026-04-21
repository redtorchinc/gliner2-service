"""Logging configuration for the service."""

from __future__ import annotations

import logging

from service.config import settings


def setup_logging() -> None:
    fmt = "%(asctime)s %(levelname)s %(name)s %(message)s"
    logging.basicConfig(format=fmt, level=settings.log_level.upper(), force=True)
    # Let uvicorn inherit our level
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        logging.getLogger(name).setLevel(settings.log_level.upper())
