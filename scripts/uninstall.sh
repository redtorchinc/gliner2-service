#!/usr/bin/env bash
set -euo pipefail

SERVICE_NAME="gliner2-service"
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
VENV_DIR="${REPO_DIR}/.venv-service"

echo "=== GLiNER2 Service Uninstaller ==="
echo ""

# Stop and disable systemd service
echo "Stopping service ..."
sudo systemctl stop "${SERVICE_NAME}" 2>/dev/null || true
sudo systemctl disable "${SERVICE_NAME}" 2>/dev/null || true
sudo rm -f "/etc/systemd/system/${SERVICE_NAME}.service"
sudo systemctl daemon-reload
echo "Systemd unit removed."

# Remove venv
if [ -d "${VENV_DIR}" ]; then
    echo ""
    read -r -p "Delete virtual environment at ${VENV_DIR}? [y/N] " answer
    case "${answer}" in
        [yY]|[yY][eE][sS])
            rm -rf "${VENV_DIR}"
            echo "Virtual environment removed."
            ;;
        *)
            echo "Keeping virtual environment."
            ;;
    esac
fi

echo ""
echo "Uninstall complete. The repository has not been deleted."
