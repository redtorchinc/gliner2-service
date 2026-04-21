# GLiNER2 FastAPI Service — Implementation Spec

## Goal

Wrap the `gliner2` Python library as a self-contained HTTP service on Linux. A single `install.sh` script must take a clean clone of the repo all the way to a running, auto-restarting systemd service exposing a FastAPI app on **port 8077** (chosen to avoid common conflicts).

After running `install.sh` the user should be able to:

```bash
curl http://localhost:8077/health
curl -X POST http://localhost:8077/v1/extract/entities \
  -H 'Content-Type: application/json' \
  -d '{"text":"Apple CEO Tim Cook announced iPhone 15 in Cupertino.","entity_types":["company","person","product","location"]}'
```

…and get valid responses. The service must survive reboots.

---

## Deliverables (final repo layout)

Claude Code should produce this structure in the repo root (the existing `gliner2/` package directory is preserved untouched):

```
.
├── gliner2/                      # (existing — do not modify)
├── service/
│   ├── __init__.py
│   ├── main.py                   # FastAPI app factory + routes
│   ├── model_manager.py          # Lazy singleton around GLiNER2
│   ├── schemas.py                # Pydantic request/response models
│   ├── config.py                 # Settings (env-backed)
│   └── logging_conf.py           # uvicorn + app logging config
├── scripts/
│   ├── install.sh                # One-shot installer (this is the finish line)
│   ├── uninstall.sh              # Removes service + venv
│   ├── run_dev.sh                # Foreground dev runner (no systemd)
│   └── gliner2-service.service   # systemd unit template
├── requirements-service.txt      # Service-only deps on top of gliner2
├── SERVICE.md                    # Operator-facing README for the service
└── .env.example                  # Example env file
```

Do **not** edit anything inside `gliner2/`, `tests/`, `tutorial/`, `benchmarks/`, or the existing `pyproject.toml` / `README.md`. Everything new lives under `service/` and `scripts/`.

---

## Runtime target

- **OS:** Linux with systemd (Ubuntu 22.04+ / Debian 12+ / any modern distro with `systemctl`). Do not assume root; use `sudo` only where required and prompt clearly.
- **Python:** 3.10+ (the library claims 3.8+, but FastAPI/Pydantic v2 are nicer on 3.10+). Detect and fail fast if older.
- **Device:** CPU by default. If `torch.cuda.is_available()` at startup, use CUDA. Controlled by env var `GLINER2_DEVICE` (`auto` | `cpu` | `cuda`), default `auto`.
- **Port:** 8077 (configurable via `GLINER2_PORT`).
- **Bind host:** `127.0.0.1` by default (configurable via `GLINER2_HOST`). Do not expose publicly by default.
- **Workers:** 1 uvicorn worker. The model is heavy and not fork-safe across workers; scale with replicas/a reverse proxy if needed. Document this in `SERVICE.md`.

---

## Dependencies

`requirements-service.txt` must contain:

```
fastapi>=0.110
uvicorn[standard]>=0.27
pydantic>=2.5
python-dotenv>=1.0
```

The `gliner2` package itself is installed from the repo root via `pip install .`, which pulls in `gliner` and `pydantic` per `pyproject.toml`. `torch` is required by `gliner` transitively — let pip handle it, but in `install.sh` detect if CUDA is present and print a note suggesting the user may want to manually install a matching `torch` build before running the installer if they want GPU support. Do not try to be clever about auto-picking a CUDA wheel.

---

## `service/config.py`

Use `pydantic-settings` style or a plain dataclass loaded from env. Keep it small:

```python
# Fields (env var → attribute)
GLINER2_MODEL          → model: str             = "fastino/gliner2-base-v1"
GLINER2_DEVICE         → device: str            = "auto"   # auto|cpu|cuda
GLINER2_QUANTIZE       → quantize: bool         = False
GLINER2_COMPILE        → compile: bool          = False
GLINER2_HOST           → host: str              = "127.0.0.1"
GLINER2_PORT           → port: int              = 8077
GLINER2_MAX_TEXT_CHARS → max_text_chars: int    = 200_000
GLINER2_MAX_BATCH_SIZE → max_batch_size: int    = 64
GLINER2_API_KEY        → api_key: str | None    = None     # optional bearer
GLINER2_LOG_LEVEL      → log_level: str         = "INFO"
```

Load `.env` from the repo root if present. Settings instance is a module-level singleton.

---

## `service/model_manager.py`

A thread-safe lazy singleton. The model is loaded **on first request**, not at import time, so the service starts fast and health checks work before the model is hot.

```python
class ModelManager:
    def __init__(self, settings): ...
    def get(self) -> GLiNER2:      # loads if needed, returns instance
    def status(self) -> dict:      # {"loaded": bool, "device": str, "model": str, "loaded_at": iso8601|null}
    def reload(self): ...          # drop and reload (used by /admin/reload)
```

- Use `threading.Lock` around load. Requests that arrive during a load wait on the lock.
- Respect `quantize` and `compile` settings via `GLiNER2.from_pretrained(..., quantize=..., compile=..., map_location=...)`.
- On load failure, keep the manager in a failed state with the exception recorded; `/health` should reflect it but the service should not crash.

---

## `service/schemas.py`

Pydantic v2 models for each endpoint. These mirror the `gliner2` public API 1:1 so callers don't have to learn a new vocabulary.

### Shared options

```python
class CommonOptions(BaseModel):
    threshold: float = 0.5
    format_results: bool = True
    include_confidence: bool = False
    include_spans: bool = False
    max_len: int | None = None
```

### Entities

```python
class EntitiesRequest(CommonOptions):
    text: str
    # list[str]  OR  dict[str, str] (type → description)
    entity_types: list[str] | dict[str, str]

class BatchEntitiesRequest(CommonOptions):
    texts: list[str]
    entity_types: list[str] | dict[str, str]
    batch_size: int = 8
```

### Classification

```python
class ClassifyRequest(CommonOptions):
    text: str
    # task_name → list[str] labels, OR task_name → {"labels": [...], "multi_label": bool}
    tasks: dict[str, list[str] | dict]

class BatchClassifyRequest(CommonOptions):
    texts: list[str]
    tasks: dict[str, list[str] | dict]
    batch_size: int = 8
```

### JSON / Structure

```python
class ExtractJsonRequest(CommonOptions):
    text: str
    # parent → list of field specs, matching gliner2's extract_json format
    structures: dict[str, list[str]]

class BatchExtractJsonRequest(CommonOptions):
    texts: list[str]
    structures: dict[str, list[str]]
    batch_size: int = 8
```

### Relations

```python
class RelationsRequest(CommonOptions):
    text: str
    relation_types: list[str] | dict[str, str]

class BatchRelationsRequest(CommonOptions):
    texts: list[str]
    relation_types: list[str] | dict[str, str]
    batch_size: int = 8
```

### Full combined extract (schema builder equivalent via dict form)

```python
class ExtractRequest(CommonOptions):
    text: str
    # Full schema dict accepted by Schema.from_dict():
    # {
    #   "entities": [...] | {name: desc, ...},
    #   "classifications": [{"task": ..., "labels": [...], "multi_label": bool}, ...],
    #   "structures": {name: {"fields": [{"name":..., "dtype":..., "choices":..., "description":...}, ...]}, ...},
    #   "relations": [...] | {name: desc, ...}
    # }
    schema_def: dict  # field is called schema_def to avoid Pydantic's BaseModel.schema() clash

class BatchExtractRequest(CommonOptions):
    texts: list[str]
    schema_def: dict
    batch_size: int = 8
```

### Validation rules (applied in routes)

- `text` non-empty, length ≤ `settings.max_text_chars`.
- `texts` non-empty, each element ≤ `settings.max_text_chars`, `len(texts)` ≤ `settings.max_batch_size`.
- `threshold` ∈ [0, 1].
- `batch_size` ≥ 1 and ≤ `settings.max_batch_size`.
- Return HTTP 422 on validation failure (Pydantic default) or HTTP 400 for the app-level limits with a clear message.

### Responses

Keep responses thin — pass through whatever `gliner2` returns under a `result` key, plus metadata:

```python
class ExtractionResponse(BaseModel):
    result: dict | list     # dict for single-text endpoints, list[dict] for batch
    elapsed_ms: float
    model: str
```

---

## `service/main.py` — routes

All routes are under `/v1`. All POST bodies are JSON.

| Method | Path                           | Handler                                 |
|--------|--------------------------------|-----------------------------------------|
| GET    | `/health`                      | `{"status": "ok", "model_loaded": bool}`|
| GET    | `/v1/info`                     | model name, device, version, settings summary |
| POST   | `/v1/extract/entities`         | `GLiNER2.extract_entities`              |
| POST   | `/v1/extract/entities/batch`   | `GLiNER2.batch_extract_entities`        |
| POST   | `/v1/classify`                 | `GLiNER2.classify_text`                 |
| POST   | `/v1/classify/batch`           | `GLiNER2.batch_classify_text`           |
| POST   | `/v1/extract/json`             | `GLiNER2.extract_json`                  |
| POST   | `/v1/extract/json/batch`       | `GLiNER2.batch_extract_json`            |
| POST   | `/v1/extract/relations`        | `GLiNER2.extract_relations`             |
| POST   | `/v1/extract/relations/batch`  | `GLiNER2.batch_extract_relations`       |
| POST   | `/v1/extract`                  | `GLiNER2.extract` with `Schema.from_dict(schema_def)` |
| POST   | `/v1/extract/batch`            | `GLiNER2.batch_extract` with `Schema.from_dict(schema_def)` |
| POST   | `/v1/admin/reload`             | reloads the model (requires auth if `api_key` is set) |

### Handler pattern (pseudo-code)

```python
@app.post("/v1/extract/entities", response_model=ExtractionResponse)
async def extract_entities(req: EntitiesRequest, model=Depends(get_model)):
    _validate_text(req.text)
    t0 = time.perf_counter()
    # Run the blocking torch call in a thread so we don't block the event loop
    result = await run_in_threadpool(
        model.extract_entities,
        text=req.text,
        entity_types=req.entity_types,
        threshold=req.threshold,
        format_results=req.format_results,
        include_confidence=req.include_confidence,
        include_spans=req.include_spans,
        max_len=req.max_len,
    )
    return ExtractionResponse(
        result=result,
        elapsed_ms=(time.perf_counter() - t0) * 1000,
        model=settings.model,
    )
```

- **Always** wrap model calls in `fastapi.concurrency.run_in_threadpool`. These are CPU/GPU-bound, not async.
- **Never** hold the model lock across an inference call — the manager's lock is load-only.
- For `/v1/extract` and `/v1/extract/batch`, call `Schema.from_dict(req.schema_def)` inside the try/except; on `ValidationError` from gliner2 return 400 with the message.

### Auth

If `GLINER2_API_KEY` is set, require `Authorization: Bearer <key>` on every route except `/health`. Implement as a FastAPI dependency. Constant-time compare.

### Error handling

- Pydantic validation → 422 (default).
- App limits (text too long, batch too big) → 400 with `{"detail": "..."}`.
- `gliner2` validation errors → 400.
- Anything else → 500 with a short generic message; full traceback goes to the log.
- Add an exception handler that logs `request.method request.url.path` with the exception.

### CORS

Off by default. If we ever need it, add a `GLINER2_CORS_ORIGINS` env var later — do not add CORS in v1.

---

## `service/logging_conf.py`

Set up plain Python logging with the level from settings. Uvicorn should inherit. Do not use JSON logging — `journalctl` will handle structure. Format:

```
%(asctime)s %(levelname)s %(name)s %(message)s
```

---

## `scripts/install.sh`

This is the finish line. It must be idempotent — running it twice should be safe. Use `set -euo pipefail`.

### Steps

1. **Preflight**
   - Require Linux. Abort on macOS/WSL with a clear message (WSL may work but don't promise it).
   - Require `systemctl` on PATH. If missing, explain and exit.
   - Require Python ≥ 3.10. Check `python3 --version`.
   - Warn if not running as the user who will own the service. Recommend running as a non-root user (say, the current user).
   - Print a banner showing: install dir, venv path, port, service name, user.

2. **Resolve paths**
   - `REPO_DIR` = absolute path of the repo root (derive from script location).
   - `VENV_DIR` = `${REPO_DIR}/.venv-service`.
   - `SERVICE_USER` = current user (`$USER`).
   - `SERVICE_NAME` = `gliner2-service`.

3. **Create venv**
   - `python3 -m venv "$VENV_DIR"` (skip if it already exists with a matching Python).
   - Activate and upgrade pip: `pip install --upgrade pip wheel setuptools`.

4. **Install deps**
   - `pip install -r "${REPO_DIR}/requirements-service.txt"`.
   - `pip install "${REPO_DIR}"` (installs `gliner2` itself, which pulls `gliner`, `torch`, etc. — this step will be slow the first time; print a note).

5. **Smoke test (offline)**
   - `"$VENV_DIR/bin/python" -c "from gliner2 import GLiNER2; from service.main import create_app; print('import ok')"`.
   - Do **not** download the model here — that happens on first request. Downloading it now would double the install time.

6. **Generate systemd unit**
   - Render `scripts/gliner2-service.service` (a template with `@REPO_DIR@`, `@VENV_DIR@`, `@USER@`, `@PORT@`, `@HOST@` placeholders) into `/tmp/${SERVICE_NAME}.service`.
   - `sudo mv /tmp/${SERVICE_NAME}.service /etc/systemd/system/${SERVICE_NAME}.service`.
   - `sudo systemctl daemon-reload`.
   - `sudo systemctl enable "${SERVICE_NAME}"`.
   - `sudo systemctl restart "${SERVICE_NAME}"`.

7. **Post-check**
   - Wait up to 15 seconds for the port to start accepting connections (loop with `curl -sf http://127.0.0.1:${PORT}/health`).
   - On success: print the curl example, `journalctl` command, and `systemctl status` command.
   - On failure: print `journalctl -u ${SERVICE_NAME} -n 50 --no-pager` output and exit non-zero.

### systemd unit template (`scripts/gliner2-service.service`)

```ini
[Unit]
Description=GLiNER2 FastAPI Service
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=@USER@
Group=@USER@
WorkingDirectory=@REPO_DIR@
Environment="PATH=@VENV_DIR@/bin"
Environment="GLINER2_HOST=@HOST@"
Environment="GLINER2_PORT=@PORT@"
EnvironmentFile=-@REPO_DIR@/.env
ExecStart=@VENV_DIR@/bin/uvicorn service.main:app \
    --host ${GLINER2_HOST} \
    --port ${GLINER2_PORT} \
    --workers 1 \
    --log-level info \
    --no-access-log
Restart=on-failure
RestartSec=3
# Give the model time to load on first request
TimeoutStartSec=120
# Reasonable hardening — safe for a local service
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=full
ProtectHome=read-only
# Allow writing to the HF cache, the venv, and the repo dir
ReadWritePaths=@REPO_DIR@ %h/.cache/huggingface

[Install]
WantedBy=multi-user.target
```

`service/main.py` must expose a module-level `app = create_app()` so `uvicorn service.main:app` works.

### `scripts/run_dev.sh`

```bash
#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv-service/bin/activate 2>/dev/null || { echo "Run install.sh first"; exit 1; }
exec uvicorn service.main:app --host "${GLINER2_HOST:-127.0.0.1}" --port "${GLINER2_PORT:-8077}" --reload
```

### `scripts/uninstall.sh`

- `sudo systemctl stop gliner2-service || true`
- `sudo systemctl disable gliner2-service || true`
- `sudo rm -f /etc/systemd/system/gliner2-service.service`
- `sudo systemctl daemon-reload`
- Ask before deleting `.venv-service`. Do not delete the repo.

---

## `.env.example`

```
# GLiNER2 service configuration
GLINER2_MODEL=fastino/gliner2-base-v1
GLINER2_DEVICE=auto
GLINER2_QUANTIZE=false
GLINER2_COMPILE=false
GLINER2_HOST=127.0.0.1
GLINER2_PORT=8077
GLINER2_MAX_TEXT_CHARS=200000
GLINER2_MAX_BATCH_SIZE=64
GLINER2_LOG_LEVEL=INFO
# GLINER2_API_KEY=change-me-to-require-bearer-auth
```

---

## `SERVICE.md` — operator README

Must cover:

1. Quickstart: `git clone … && cd … && ./scripts/install.sh`.
2. Environment variables table (copy from `.env.example` with descriptions).
3. All endpoints with one `curl` example each (use the existing tutorial examples from the repo — e.g. the Apple/Tim Cook sentence — for continuity).
4. First-request latency warning: the model downloads (~500 MB) and loads on the first request; subsequent requests are fast.
5. Logs: `journalctl -u gliner2-service -f`.
6. Reloading the model: `curl -X POST http://localhost:8077/v1/admin/reload` (or `systemctl restart gliner2-service`).
7. Uninstall: `./scripts/uninstall.sh`.
8. Scaling note: single worker is required because the model is held in memory per process. For more throughput run multiple instances behind a reverse proxy, don't increase `--workers`.
9. GPU note: if you want CUDA, install a matching `torch` build into `.venv-service` before or after running `install.sh`, then set `GLINER2_DEVICE=cuda` in `.env` and restart.

---

## Acceptance checklist — how Claude Code knows it's done

Run, in order, on a fresh Ubuntu 22.04+ VM:

1. `git clone <repo> && cd <repo>` ✅
2. `./scripts/install.sh` completes with exit code 0 and the post-check succeeds. ✅
3. `curl -sf http://localhost:8077/health` → `{"status":"ok","model_loaded":false}`. ✅
4. First real request (the Apple/Tim Cook entity example above) returns `200` with the expected entities. Response time on the first call may be 30–90s (model download + load); log a clear one-time message. ✅
5. Second request to the same endpoint returns in under a couple of seconds on CPU. ✅
6. `sudo systemctl restart gliner2-service` followed by `/health` returns OK within 5s (model reloads lazily on next inference request). ✅
7. `sudo reboot`; after reboot `curl /health` still works without manual intervention. ✅
8. Each of the six non-batch endpoints (`entities`, `classify`, `json`, `relations`, plus `extract`) returns sane results for a small example copied from the repo's tutorials. ✅
9. `./scripts/uninstall.sh` stops the service, removes the unit, and (with confirmation) removes `.venv-service`. Re-running `install.sh` after that works. ✅

---

## Things Claude Code should NOT do

- Don't modify `pyproject.toml`, `gliner2/`, or anything the test suite touches.
- Don't add a database, Redis, or a task queue.
- Don't add Docker. Systemd is the target.
- Don't try to auto-install CUDA torch wheels.
- Don't expose the service on `0.0.0.0` by default.
- Don't load the model in `create_app()` — strictly lazy.
- Don't use gunicorn. Plain uvicorn, single worker.
- Don't invent new concepts in the request schema; mirror `gliner2`'s existing method signatures exactly so that anyone reading the tutorials already knows how to call the service.

---

## One-line summary for the agent

**Build a `service/` FastAPI package and a `scripts/install.sh` that, starting from a fresh clone on Linux, ends with a systemd-managed uvicorn process on port 8077 exposing every public `GLiNER2` inference method as a JSON endpoint.**
