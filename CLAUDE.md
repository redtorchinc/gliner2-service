# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

GLiNER2 is a Python library for unified schema-based information extraction built on PyTorch and Hugging Face Transformers. It supports entity extraction (NER), text classification, structured JSON extraction, and relation extraction. Version is in `gliner2/__init__.py` (currently 1.3.0).

## Common Commands

```bash
# Install locally (editable)
pip install -e .

# Run all tests
pytest tests/

# Run a single test
pytest tests/test_entity_extraction.py

# Run benchmarks
python benchmark_statistical.py --tag baseline --n 300
python benchmarks/benchmark_batching.py

# Build for PyPI release
pip install build twine
rm -rf dist/ build/ *.egg-info/
python -m build
twine upload dist/*

# --- FastAPI service ---
# Install & start systemd service (Linux only)
./scripts/install.sh

# Run service in dev mode (foreground with auto-reload)
./scripts/run_dev.sh

# Uninstall systemd service
./scripts/uninstall.sh
```

## Architecture

### Core Modules (`gliner2/`)

- **`inference/engine.py`** — `GLiNER2` class: main entry point for all extraction tasks. Handles model loading (`from_pretrained`, `from_api`), batch processing via DataLoader, fp16 quantization, and torch.compile support. All four extraction modes (entities, classification, structures, relations) flow through here.

- **`model.py`** — `Extractor` (PyTorch `PreTrainedModel`) and `ExtractorConfig`: the neural network architecture using transformer encoders (DeBERTa/BERT) with a span representation layer. Handles forward pass for both training and inference.

- **`processor.py`** — `SchemaTransformer`: preprocessing pipeline that converts raw text + schema into GPU-ready batches (`PreprocessedBatch`). Handles tokenization, token mapping/alignment, and DataLoader-compatible collation.

- **`layers.py`** — Custom neural network layers: `CompileSafeGRU`, `CountLSTM` variants, MLP factory. Used for span counting/scoring within the model.

- **`api_client.py`** — `GLiNER2API`: remote API wrapper with retry logic. Mirrors the local `GLiNER2` interface. Authenticates via `PIONEER_API_KEY` environment variable.

### Training (`gliner2/training/`)

- **`trainer.py`** — `GLiNER2Trainer`: orchestrates training with AdamW optimizer, linear warmup, early stopping, validation, and checkpointing. Accepts JSONL, dicts, or `InputExample` objects.

- **`data.py`** — `InputExample`, `TrainingDataset`, label classes (`Classification`, `Structure`, `Relation`). Handles data loading, validation, and format detection.

- **`lora.py`** — LoRA (parameter-efficient fine-tuning): `LoRAConfig`, `LoRALayer`, and utilities for saving/loading/merging adapters.

### Schema Validation (`gliner2/inference/schema_model.py`)

Pydantic 2.0+ models for runtime validation of extraction schemas: `SchemaInput`, `FieldInput`, `StructureInput`, `ClassificationInput`.

### FastAPI Service (`service/`)

HTTP wrapper around GLiNER2 inference methods, deployed via systemd on Linux.

- **`config.py`** — `Settings` dataclass loaded from env vars (prefixed `GLINER2_`). Reads `.env` from repo root.
- **`model_manager.py`** — `ModelManager`: thread-safe lazy singleton. Model loads on first request, not at startup. `/health` works before model is hot.
- **`schemas.py`** — Pydantic v2 request/response models mirroring GLiNER2's method signatures 1:1.
- **`main.py`** — FastAPI app factory with all routes under `/v1`. Model calls run in `run_in_threadpool`. Module-level `app = create_app()` for uvicorn.
- **`logging_conf.py`** — Plain Python logging; uvicorn inherits the configured level.

Scripts: `scripts/install.sh` (idempotent systemd installer), `scripts/run_dev.sh` (foreground with `--reload`), `scripts/uninstall.sh`.

## Key Design Patterns

- **Dual mode**: Models can run locally (`GLiNER2.from_pretrained("fastino/gliner2-base-v1")`) or via API (`GLiNER2.from_api()`), sharing the same interface.
- **Dependencies**: Core deps are `gliner`, `pydantic>=2.0.0`. PyTorch and Transformers are transitive via `gliner`.
- **Models are loaded from Hugging Face Hub** (e.g., `fastino/gliner2-base-v1`, `fastino/gliner2-large-v1`).
- **Fork relationship**: This repo (RedTorch Inc) is forked from `fastino-ai/GLiNER2` upstream.
- **Do not modify** `gliner2/`, `tests/`, `tutorial/`, `benchmarks/`, or `pyproject.toml` — the service lives entirely in `service/` and `scripts/`.
