"""FastAPI application for the GLiNER2 service."""

from __future__ import annotations

import hmac
import logging
import time
import traceback
from typing import Any, Callable, Dict

import torch
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse

import gliner2 as _gliner2_pkg
from service.config import settings
from service.logging_conf import setup_logging
from service.model_manager import ModelManager
from service.schemas import (
    BatchClassifyRequest,
    BatchEntitiesRequest,
    BatchExtractJsonRequest,
    BatchExtractRequest,
    BatchRelationsRequest,
    ClassifyRequest,
    EntitiesRequest,
    ExtractJsonRequest,
    ExtractRequest,
    ExtractionResponse,
    RelationsRequest,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level singletons
# ---------------------------------------------------------------------------

_manager = ModelManager(settings)


# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------

def _check_auth(request: Request) -> None:
    if settings.api_key is None:
        return
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing Bearer token")
    token = auth[7:]
    if not hmac.compare_digest(token, settings.api_key):
        raise HTTPException(status_code=401, detail="Invalid API key")


async def _get_model():
    # Must be async so FastAPI does NOT run this in a threadpool.
    # huggingface_hub's httpx client breaks when used from a worker thread.
    # Blocks the event loop only on first call (model download); instant after.
    return _manager.get()


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def _validate_text(text: str) -> None:
    if len(text) > settings.max_text_chars:
        raise HTTPException(
            status_code=413,
            detail=f"Text length {len(text)} exceeds maximum of {settings.max_text_chars} characters",
        )


def _validate_texts(texts: list[str]) -> None:
    if len(texts) > settings.max_batch_size:
        raise HTTPException(
            status_code=413,
            detail=f"Batch size {len(texts)} exceeds maximum of {settings.max_batch_size}",
        )
    for i, t in enumerate(texts):
        if len(t) > settings.max_text_chars:
            raise HTTPException(
                status_code=413,
                detail=f"texts[{i}] length {len(t)} exceeds maximum of {settings.max_text_chars} characters",
            )


def _validate_batch_size(batch_size: int) -> None:
    if batch_size > settings.max_batch_size:
        raise HTTPException(
            status_code=413,
            detail=f"batch_size {batch_size} exceeds maximum of {settings.max_batch_size}",
        )


def _cuda_cache_clear() -> None:
    """Free cached CUDA memory after each inference call."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


async def _run_inference(fn: Callable, *args, **kwargs) -> Any:
    """Run a model call in the threadpool, catching OOM and clearing CUDA cache."""
    try:
        result = await run_in_threadpool(fn, *args, **kwargs)
    except torch.cuda.OutOfMemoryError:
        _cuda_cache_clear()
        raise HTTPException(
            status_code=413,
            detail=(
                "CUDA out of memory — input is too large for available GPU memory. "
                "Try shorter text, a smaller batch_size, or set max_len. "
                "Hint: PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True may help."
            ),
        )
    _cuda_cache_clear()
    return result


def _response(result: Any, elapsed: float) -> ExtractionResponse:
    return ExtractionResponse(result=result, elapsed_ms=elapsed * 1000, model=settings.model)


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

def create_app() -> FastAPI:
    setup_logging()

    application = FastAPI(title="GLiNER2 Service", version=_gliner2_pkg.__version__)

    # --- Exception handler ---------------------------------------------------

    @application.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception) -> JSONResponse:
        logger.error("%s %s — %s\n%s", request.method, request.url.path, exc, traceback.format_exc())
        return JSONResponse(status_code=500, content={"detail": "Internal server error"})

    # --- Health (no auth) ----------------------------------------------------

    @application.get("/health")
    async def health() -> Dict[str, Any]:
        return {"status": "ok", "model_loaded": _manager.status()["loaded"]}

    # --- Info ----------------------------------------------------------------

    @application.get("/v1/info", dependencies=[Depends(_check_auth)])
    async def info() -> Dict[str, Any]:
        s = _manager.status()
        return {
            "model": s["model"],
            "device": s["device"],
            "loaded": s["loaded"],
            "loaded_at": s["loaded_at"],
            "version": _gliner2_pkg.__version__,
            "host": settings.host,
            "port": settings.port,
            "quantize": settings.quantize,
            "compile": settings.compile,
        }

    # --- Entity extraction ---------------------------------------------------

    @application.post("/v1/extract/entities", response_model=ExtractionResponse, dependencies=[Depends(_check_auth)])
    async def extract_entities(req: EntitiesRequest, model=Depends(_get_model)):
        _validate_text(req.text)
        t0 = time.perf_counter()
        result = await _run_inference(
            model.extract_entities,
            text=req.text,
            entity_types=req.entity_types,
            threshold=req.threshold,
            format_results=req.format_results,
            include_confidence=req.include_confidence,
            include_spans=req.include_spans,
            max_len=req.max_len,
        )
        return _response(result, time.perf_counter() - t0)

    @application.post("/v1/extract/entities/batch", response_model=ExtractionResponse, dependencies=[Depends(_check_auth)])
    async def batch_extract_entities(req: BatchEntitiesRequest, model=Depends(_get_model)):
        _validate_texts(req.texts)
        _validate_batch_size(req.batch_size)
        t0 = time.perf_counter()
        result = await _run_inference(
            model.batch_extract_entities,
            texts=req.texts,
            entity_types=req.entity_types,
            batch_size=req.batch_size,
            threshold=req.threshold,
            format_results=req.format_results,
            include_confidence=req.include_confidence,
            include_spans=req.include_spans,
            max_len=req.max_len,
        )
        return _response(result, time.perf_counter() - t0)

    # --- Classification ------------------------------------------------------

    @application.post("/v1/classify", response_model=ExtractionResponse, dependencies=[Depends(_check_auth)])
    async def classify_text(req: ClassifyRequest, model=Depends(_get_model)):
        _validate_text(req.text)
        t0 = time.perf_counter()
        result = await _run_inference(
            model.classify_text,
            text=req.text,
            tasks=req.tasks,
            threshold=req.threshold,
            format_results=req.format_results,
            include_confidence=req.include_confidence,
            include_spans=req.include_spans,
            max_len=req.max_len,
        )
        return _response(result, time.perf_counter() - t0)

    @application.post("/v1/classify/batch", response_model=ExtractionResponse, dependencies=[Depends(_check_auth)])
    async def batch_classify_text(req: BatchClassifyRequest, model=Depends(_get_model)):
        _validate_texts(req.texts)
        _validate_batch_size(req.batch_size)
        t0 = time.perf_counter()
        result = await _run_inference(
            model.batch_classify_text,
            texts=req.texts,
            tasks=req.tasks,
            batch_size=req.batch_size,
            threshold=req.threshold,
            format_results=req.format_results,
            include_confidence=req.include_confidence,
            include_spans=req.include_spans,
            max_len=req.max_len,
        )
        return _response(result, time.perf_counter() - t0)

    # --- JSON / Structure extraction -----------------------------------------

    @application.post("/v1/extract/json", response_model=ExtractionResponse, dependencies=[Depends(_check_auth)])
    async def extract_json(req: ExtractJsonRequest, model=Depends(_get_model)):
        _validate_text(req.text)
        t0 = time.perf_counter()
        result = await _run_inference(
            model.extract_json,
            text=req.text,
            structures=req.structures,
            threshold=req.threshold,
            format_results=req.format_results,
            include_confidence=req.include_confidence,
            include_spans=req.include_spans,
            max_len=req.max_len,
        )
        return _response(result, time.perf_counter() - t0)

    @application.post("/v1/extract/json/batch", response_model=ExtractionResponse, dependencies=[Depends(_check_auth)])
    async def batch_extract_json(req: BatchExtractJsonRequest, model=Depends(_get_model)):
        _validate_texts(req.texts)
        _validate_batch_size(req.batch_size)
        t0 = time.perf_counter()
        result = await _run_inference(
            model.batch_extract_json,
            texts=req.texts,
            structures=req.structures,
            batch_size=req.batch_size,
            threshold=req.threshold,
            format_results=req.format_results,
            include_confidence=req.include_confidence,
            include_spans=req.include_spans,
            max_len=req.max_len,
        )
        return _response(result, time.perf_counter() - t0)

    # --- Relation extraction -------------------------------------------------

    @application.post("/v1/extract/relations", response_model=ExtractionResponse, dependencies=[Depends(_check_auth)])
    async def extract_relations(req: RelationsRequest, model=Depends(_get_model)):
        _validate_text(req.text)
        t0 = time.perf_counter()
        result = await _run_inference(
            model.extract_relations,
            text=req.text,
            relation_types=req.relation_types,
            threshold=req.threshold,
            format_results=req.format_results,
            include_confidence=req.include_confidence,
            include_spans=req.include_spans,
            max_len=req.max_len,
        )
        return _response(result, time.perf_counter() - t0)

    @application.post("/v1/extract/relations/batch", response_model=ExtractionResponse, dependencies=[Depends(_check_auth)])
    async def batch_extract_relations(req: BatchRelationsRequest, model=Depends(_get_model)):
        _validate_texts(req.texts)
        _validate_batch_size(req.batch_size)
        t0 = time.perf_counter()
        result = await _run_inference(
            model.batch_extract_relations,
            texts=req.texts,
            relation_types=req.relation_types,
            batch_size=req.batch_size,
            threshold=req.threshold,
            format_results=req.format_results,
            include_confidence=req.include_confidence,
            include_spans=req.include_spans,
            max_len=req.max_len,
        )
        return _response(result, time.perf_counter() - t0)

    # --- Full combined extract -----------------------------------------------

    @application.post("/v1/extract", response_model=ExtractionResponse, dependencies=[Depends(_check_auth)])
    async def extract(req: ExtractRequest, model=Depends(_get_model)):
        _validate_text(req.text)
        from gliner2.inference.engine import Schema
        try:
            schema = Schema.from_dict(req.schema_def)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        t0 = time.perf_counter()
        result = await _run_inference(
            model.extract,
            text=req.text,
            schema=schema,
            threshold=req.threshold,
            format_results=req.format_results,
            include_confidence=req.include_confidence,
            include_spans=req.include_spans,
            max_len=req.max_len,
        )
        return _response(result, time.perf_counter() - t0)

    @application.post("/v1/extract/batch", response_model=ExtractionResponse, dependencies=[Depends(_check_auth)])
    async def batch_extract(req: BatchExtractRequest, model=Depends(_get_model)):
        _validate_texts(req.texts)
        _validate_batch_size(req.batch_size)
        from gliner2.inference.engine import Schema
        try:
            schema = Schema.from_dict(req.schema_def)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        t0 = time.perf_counter()
        result = await _run_inference(
            model.batch_extract,
            texts=req.texts,
            schemas=schema,
            batch_size=req.batch_size,
            threshold=req.threshold,
            format_results=req.format_results,
            include_confidence=req.include_confidence,
            include_spans=req.include_spans,
            max_len=req.max_len,
        )
        return _response(result, time.perf_counter() - t0)

    # --- Admin ---------------------------------------------------------------

    @application.post("/v1/admin/reload", dependencies=[Depends(_check_auth)])
    async def admin_reload():
        await run_in_threadpool(_manager.reload)
        return {"status": "ok", **_manager.status()}

    return application


# Module-level app for uvicorn
app = create_app()
