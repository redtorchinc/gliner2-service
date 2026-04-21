# GLiNER2 Service — Operator Guide

A self-contained FastAPI HTTP service wrapping the `gliner2` library, managed by systemd.

## Quickstart

```bash
git clone <repo-url> && cd gliner2-service
./scripts/install.sh
```

The installer creates a virtual environment, installs all dependencies, and sets up a systemd service on port 8077.

## Environment Variables

Copy `.env.example` to `.env` and edit as needed. The service reads `.env` from the repo root.

| Variable | Default | Description |
|---|---|---|
| `GLINER2_MODEL` | `fastino/gliner2-base-v1` | HuggingFace model ID or local path |
| `GLINER2_DEVICE` | `auto` | `auto`, `cpu`, or `cuda` |
| `GLINER2_QUANTIZE` | `false` | Convert model to fp16 on load |
| `GLINER2_COMPILE` | `false` | Use `torch.compile` for faster GPU inference |
| `GLINER2_HOST` | `127.0.0.1` | Bind address |
| `GLINER2_PORT` | `8077` | Bind port |
| `GLINER2_MAX_TEXT_CHARS` | `200000` | Maximum characters per text input |
| `GLINER2_MAX_BATCH_SIZE` | `64` | Maximum texts in a batch request |
| `GLINER2_LOG_LEVEL` | `INFO` | Python log level |
| `GLINER2_API_KEY` | *(unset)* | If set, all endpoints (except `/health`) require `Authorization: Bearer <key>` |

## First-Request Latency

The model is **not** loaded at startup. The first inference request triggers a download (~500 MB) and model load. This can take 30–90 seconds. Subsequent requests are fast. `/health` responds immediately regardless of model state.

## Endpoints

All POST endpoints accept `Content-Type: application/json`.

### Health

```bash
curl http://localhost:8077/health
# {"status":"ok","model_loaded":false}
```

### Info

```bash
curl http://localhost:8077/v1/info
```

### Entity Extraction

```bash
curl -X POST http://localhost:8077/v1/extract/entities \
  -H 'Content-Type: application/json' \
  -d '{
    "text": "Apple CEO Tim Cook announced iPhone 15 in Cupertino.",
    "entity_types": ["company", "person", "product", "location"]
  }'
```

### Batch Entity Extraction

```bash
curl -X POST http://localhost:8077/v1/extract/entities/batch \
  -H 'Content-Type: application/json' \
  -d '{
    "texts": [
      "Apple CEO Tim Cook announced iPhone 15 in Cupertino.",
      "Google hired 100 engineers in New York."
    ],
    "entity_types": ["company", "person", "location"],
    "batch_size": 2
  }'
```

### Classification

```bash
curl -X POST http://localhost:8077/v1/classify \
  -H 'Content-Type: application/json' \
  -d '{
    "text": "The new MacBook Pro has an amazing display and incredible battery life.",
    "tasks": {
      "sentiment": ["positive", "negative", "neutral"],
      "topic": ["technology", "sports", "politics"]
    }
  }'
```

### Batch Classification

```bash
curl -X POST http://localhost:8077/v1/classify/batch \
  -H 'Content-Type: application/json' \
  -d '{
    "texts": [
      "I love this product!",
      "Terrible experience, would not recommend."
    ],
    "tasks": {"sentiment": ["positive", "negative", "neutral"]}
  }'
```

### JSON / Structure Extraction

```bash
curl -X POST http://localhost:8077/v1/extract/json \
  -H 'Content-Type: application/json' \
  -d '{
    "text": "John Smith is a 32 year old software engineer at Google.",
    "structures": {
      "person": ["name::str", "age::str", "occupation::str", "employer::str"]
    }
  }'
```

### Batch JSON Extraction

```bash
curl -X POST http://localhost:8077/v1/extract/json/batch \
  -H 'Content-Type: application/json' \
  -d '{
    "texts": [
      "John Smith is a 32 year old software engineer at Google.",
      "Jane Doe, 28, is a data scientist at Meta."
    ],
    "structures": {
      "person": ["name::str", "age::str", "occupation::str"]
    }
  }'
```

### Relation Extraction

```bash
curl -X POST http://localhost:8077/v1/extract/relations \
  -H 'Content-Type: application/json' \
  -d '{
    "text": "Tim Cook is the CEO of Apple Inc.",
    "relation_types": ["works_for", "founded_by"]
  }'
```

### Batch Relation Extraction

```bash
curl -X POST http://localhost:8077/v1/extract/relations/batch \
  -H 'Content-Type: application/json' \
  -d '{
    "texts": [
      "Tim Cook is the CEO of Apple Inc.",
      "Elon Musk founded SpaceX."
    ],
    "relation_types": ["works_for", "founded_by"]
  }'
```

### Full Combined Extract

```bash
curl -X POST http://localhost:8077/v1/extract \
  -H 'Content-Type: application/json' \
  -d '{
    "text": "Apple CEO Tim Cook announced iPhone 15 in Cupertino.",
    "schema_def": {
      "entities": ["company", "person", "product", "location"],
      "classifications": [
        {"task": "topic", "labels": ["technology", "sports", "politics"]}
      ]
    }
  }'
```

### Batch Combined Extract

```bash
curl -X POST http://localhost:8077/v1/extract/batch \
  -H 'Content-Type: application/json' \
  -d '{
    "texts": [
      "Apple CEO Tim Cook announced iPhone 15.",
      "Google launched Gemini AI."
    ],
    "schema_def": {
      "entities": ["company", "person", "product"]
    }
  }'
```

### Admin: Reload Model

```bash
curl -X POST http://localhost:8077/v1/admin/reload
```

Or simply restart the service:

```bash
sudo systemctl restart gliner2-service
```

## Common Options

All extraction endpoints accept these optional fields:

| Field | Default | Description |
|---|---|---|
| `threshold` | `0.5` | Confidence threshold (0.0 – 1.0) |
| `format_results` | `true` | Format output into structured form |
| `include_confidence` | `false` | Include confidence scores per extraction |
| `include_spans` | `false` | Include character-level start/end positions |
| `max_len` | `null` | Max word tokens per text (null = no limit) |

Batch endpoints additionally accept `batch_size` (default `8`).

## Logs

```bash
journalctl -u gliner2-service -f
```

## Uninstall

```bash
./scripts/uninstall.sh
```

This stops the service, removes the systemd unit, and optionally deletes the virtual environment. The repository is not deleted.

## Scaling

The service runs a single uvicorn worker because the model is held in-process memory and is not fork-safe. Do **not** increase `--workers`. For more throughput, run multiple instances on different ports behind a reverse proxy (e.g. nginx).

## GPU Support

If you want CUDA inference:

1. Install a matching PyTorch build into `.venv-service`:
   ```bash
   source .venv-service/bin/activate
   pip install torch --index-url https://download.pytorch.org/whl/cu121
   ```
2. Set `GLINER2_DEVICE=cuda` in `.env`.
3. Restart: `sudo systemctl restart gliner2-service`.
