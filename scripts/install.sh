#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# GLiNER2 Service — One-shot installer
# ============================================================================

# ---------------------------------------------------------------------------
# Resolve paths
# ---------------------------------------------------------------------------
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
VENV_DIR="${REPO_DIR}/.venv-service"
SERVICE_NAME="gliner2-service"
SERVICE_USER="${USER}"
HOST="${GLINER2_HOST:-127.0.0.1}"
PORT="${GLINER2_PORT:-8077}"

# ---------------------------------------------------------------------------
# Preflight checks
# ---------------------------------------------------------------------------
echo "=== GLiNER2 Service Installer ==="
echo ""

# Require Linux
if [[ "$(uname -s)" != "Linux" ]]; then
    echo "ERROR: This installer targets Linux with systemd."
    echo "  Detected OS: $(uname -s)"
    echo "  For local development on macOS/Windows, use: scripts/run_dev.sh"
    exit 1
fi

# Require systemctl
if ! command -v systemctl &>/dev/null; then
    echo "ERROR: systemctl not found. This installer requires a systemd-based Linux distribution."
    exit 1
fi

# Require Python >= 3.10
PYTHON="python3"
if ! command -v "$PYTHON" &>/dev/null; then
    echo "ERROR: python3 not found on PATH."
    exit 1
fi

PY_VERSION=$("$PYTHON" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
PY_MAJOR=$("$PYTHON" -c 'import sys; print(sys.version_info.major)')
PY_MINOR=$("$PYTHON" -c 'import sys; print(sys.version_info.minor)')
if (( PY_MAJOR < 3 || (PY_MAJOR == 3 && PY_MINOR < 10) )); then
    echo "ERROR: Python >= 3.10 is required (found ${PY_VERSION})."
    exit 1
fi

# CUDA hint
if command -v nvidia-smi &>/dev/null; then
    echo "NOTE: NVIDIA GPU detected. If you want GPU inference, install a matching"
    echo "  PyTorch build into the venv BEFORE or AFTER this script, then set"
    echo "  GLINER2_DEVICE=cuda in .env and restart the service."
    echo ""
fi

# Banner
echo "  Install dir : ${REPO_DIR}"
echo "  Venv        : ${VENV_DIR}"
echo "  Service     : ${SERVICE_NAME}"
echo "  User        : ${SERVICE_USER}"
echo "  Bind        : ${HOST}:${PORT}"
echo ""

# ---------------------------------------------------------------------------
# Create virtual environment
# ---------------------------------------------------------------------------
if [ -d "${VENV_DIR}" ]; then
    echo "Reusing existing venv at ${VENV_DIR}"
else
    echo "Creating virtual environment ..."
    "$PYTHON" -m venv "${VENV_DIR}"
fi

# Activate
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

echo "Upgrading pip ..."
pip install --upgrade pip -q
pip install --upgrade wheel "setuptools<82" -q

# ---------------------------------------------------------------------------
# Install dependencies
# ---------------------------------------------------------------------------
echo "Installing service dependencies ..."
pip install -r "${REPO_DIR}/requirements-service.txt" -q

echo "Installing gliner2 package (this may take a while on first run) ..."
pip install "${REPO_DIR}" -q

# ---------------------------------------------------------------------------
# Smoke test (offline — does NOT download the model)
# ---------------------------------------------------------------------------
echo "Running import smoke test ..."
"${VENV_DIR}/bin/python" -c "from gliner2 import GLiNER2; from service.main import create_app; print('  Import OK')"

# ---------------------------------------------------------------------------
# systemd unit
# ---------------------------------------------------------------------------
echo "Installing systemd unit ..."

UNIT_SRC="${REPO_DIR}/scripts/gliner2-service.service"
UNIT_TMP="/tmp/${SERVICE_NAME}.service"

sed \
    -e "s|@REPO_DIR@|${REPO_DIR}|g" \
    -e "s|@VENV_DIR@|${VENV_DIR}|g" \
    -e "s|@USER@|${SERVICE_USER}|g" \
    -e "s|@HOST@|${HOST}|g" \
    -e "s|@PORT@|${PORT}|g" \
    "${UNIT_SRC}" > "${UNIT_TMP}"

sudo mv "${UNIT_TMP}" "/etc/systemd/system/${SERVICE_NAME}.service"
sudo systemctl daemon-reload
sudo systemctl enable "${SERVICE_NAME}"
sudo systemctl restart "${SERVICE_NAME}"

# ---------------------------------------------------------------------------
# Post-check — wait for the service to respond
# ---------------------------------------------------------------------------
echo ""
echo "Waiting for service to start ..."

OK=0
for i in $(seq 1 15); do
    if curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
        OK=1
        break
    fi
    sleep 1
done

if [ "${OK}" -eq 1 ]; then
    echo ""
    echo "=== GLiNER2 service is running ==="
    echo ""
    echo "  Health check:"
    echo "    curl http://127.0.0.1:${PORT}/health"
    echo ""
    echo "  Example request (first call downloads the model — may take 30-90s):"
    echo "    curl -X POST http://127.0.0.1:${PORT}/v1/extract/entities \\"
    echo "      -H 'Content-Type: application/json' \\"
    echo "      -d '{\"text\":\"Apple CEO Tim Cook announced iPhone 15 in Cupertino.\",\"entity_types\":[\"company\",\"person\",\"product\",\"location\"]}'"
    echo ""
    echo "  Logs:"
    echo "    journalctl -u ${SERVICE_NAME} -f"
    echo ""
    echo "  Status:"
    echo "    systemctl status ${SERVICE_NAME}"
    echo ""
else
    echo ""
    echo "ERROR: Service did not respond within 15 seconds."
    echo "Recent logs:"
    echo ""
    journalctl -u "${SERVICE_NAME}" -n 50 --no-pager || true
    exit 1
fi
