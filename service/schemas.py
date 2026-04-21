"""Pydantic v2 request/response models for the GLiNER2 service."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Shared options (mirrors GLiNER2 method defaults)
# ---------------------------------------------------------------------------

class CommonOptions(BaseModel):
    threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    format_results: bool = True
    include_confidence: bool = False
    include_spans: bool = False
    max_len: Optional[int] = None


# ---------------------------------------------------------------------------
# Entities
# ---------------------------------------------------------------------------

class EntitiesRequest(CommonOptions):
    text: str = Field(..., min_length=1)
    entity_types: Union[List[str], Dict[str, str]]


class BatchEntitiesRequest(CommonOptions):
    texts: List[str] = Field(..., min_length=1)
    entity_types: Union[List[str], Dict[str, str]]
    batch_size: int = Field(default=8, ge=1)


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

class ClassifyRequest(CommonOptions):
    text: str = Field(..., min_length=1)
    tasks: Dict[str, Union[List[str], Dict[str, Any]]]


class BatchClassifyRequest(CommonOptions):
    texts: List[str] = Field(..., min_length=1)
    tasks: Dict[str, Union[List[str], Dict[str, Any]]]
    batch_size: int = Field(default=8, ge=1)


# ---------------------------------------------------------------------------
# JSON / Structure
# ---------------------------------------------------------------------------

class ExtractJsonRequest(CommonOptions):
    text: str = Field(..., min_length=1)
    structures: Dict[str, list]


class BatchExtractJsonRequest(CommonOptions):
    texts: List[str] = Field(..., min_length=1)
    structures: Dict[str, list]
    batch_size: int = Field(default=8, ge=1)


# ---------------------------------------------------------------------------
# Relations
# ---------------------------------------------------------------------------

class RelationsRequest(CommonOptions):
    text: str = Field(..., min_length=1)
    relation_types: Union[List[str], Dict[str, str]]


class BatchRelationsRequest(CommonOptions):
    texts: List[str] = Field(..., min_length=1)
    relation_types: Union[List[str], Dict[str, str]]
    batch_size: int = Field(default=8, ge=1)


# ---------------------------------------------------------------------------
# Full combined extract (schema builder via dict)
# ---------------------------------------------------------------------------

class ExtractRequest(CommonOptions):
    text: str = Field(..., min_length=1)
    schema_def: Dict[str, Any]


class BatchExtractRequest(CommonOptions):
    texts: List[str] = Field(..., min_length=1)
    schema_def: Dict[str, Any]
    batch_size: int = Field(default=8, ge=1)


# ---------------------------------------------------------------------------
# Response
# ---------------------------------------------------------------------------

class ExtractionResponse(BaseModel):
    result: Union[Dict[str, Any], List[Any]]
    elapsed_ms: float
    model: str
