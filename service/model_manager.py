"""Thread-safe lazy singleton for the GLiNER2 model."""

from __future__ import annotations

import logging
import threading
from datetime import datetime, timezone
from typing import Optional

from service.config import Settings

logger = logging.getLogger(__name__)


class ModelManager:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._lock = threading.Lock()
        self._model = None
        self._loaded_at: Optional[str] = None
        self._error: Optional[Exception] = None
        self._device: Optional[str] = None

    def _resolve_device(self) -> str:
        device = self._settings.device
        if device == "auto":
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device

    def get(self):
        """Return the loaded GLiNER2 model, loading on first call."""
        if self._model is not None:
            return self._model

        with self._lock:
            # Double-check after acquiring lock
            if self._model is not None:
                return self._model

            self._load()

        if self._error is not None:
            raise self._error
        return self._model

    def _load(self) -> None:
        from gliner2 import GLiNER2

        device = self._resolve_device()
        model_name = self._settings.model
        logger.info("Loading model %s on %s (quantize=%s, compile=%s) ...",
                     model_name, device, self._settings.quantize, self._settings.compile)
        try:
            self._model = GLiNER2.from_pretrained(
                model_name,
                quantize=self._settings.quantize,
                compile=self._settings.compile,
                map_location=device,
            )
            self._device = device
            self._loaded_at = datetime.now(timezone.utc).isoformat()
            self._error = None
            logger.info("Model loaded successfully on %s", device)
        except Exception as exc:
            self._model = None
            self._error = exc
            logger.exception("Failed to load model: %s", exc)
            raise

    def status(self) -> dict:
        return {
            "loaded": self._model is not None,
            "device": self._device,
            "model": self._settings.model,
            "loaded_at": self._loaded_at,
            "error": str(self._error) if self._error else None,
        }

    def reload(self) -> None:
        with self._lock:
            logger.info("Reloading model ...")
            self._model = None
            self._loaded_at = None
            self._error = None
            self._device = None
            self._load()
