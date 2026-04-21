#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv-service/bin/activate 2>/dev/null || { echo "Run install.sh first"; exit 1; }
exec uvicorn service.main:app --host "${GLINER2_HOST:-0.0.0.0}" --port "${GLINER2_PORT:-8077}" --reload
