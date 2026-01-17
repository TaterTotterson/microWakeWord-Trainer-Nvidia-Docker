#!/usr/bin/env bash
set -euo pipefail

ROOTDIR="$(dirname "$(realpath "$0")")"

# Training convention
DATA_DIR="${DATA_DIR:-/data}"
HOST="${REC_HOST:-0.0.0.0}"
PORT="${REC_PORT:-8888}"

# Keep recorder deps separate from training venv
VENV_DIR="${DATA_DIR}/.recorder-venv"
PY="${VENV_DIR}/bin/python"
PIP="${PY} -m pip"
PIN_FILE="${VENV_DIR}/.pinned_installed"

FASTAPI_VERSION="${REC_FASTAPI_VERSION:-0.115.6}"
UVICORN_VERSION="${REC_UVICORN_VERSION:-0.30.6}"
PY_MULTIPART_VERSION="${REC_PY_MULTIPART_VERSION:-0.0.9}"

echo "microWakeWord Recorder (Docker)"
echo "-> ROOTDIR:  ${ROOTDIR}"
echo "-> DATA_DIR: ${DATA_DIR}"
echo "-> URL:      http://localhost:${PORT}/"

mkdir -p "${DATA_DIR}"

# -----------------------------
# Recorder venv (separate)
# -----------------------------
if [[ ! -x "${PY}" ]]; then
  echo "Creating recorder venv: ${VENV_DIR}"
  python3 -m venv "${VENV_DIR}"
fi

# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

if [[ ! -f "${PIN_FILE}" ]]; then
  echo "Installing pinned recorder deps"
  ${PIP} install -U pip setuptools wheel
  ${PIP} install \
    "fastapi==${FASTAPI_VERSION}" \
    "uvicorn[standard]==${UVICORN_VERSION}" \
    "python-multipart==${PY_MULTIPART_VERSION}"
  touch "${PIN_FILE}"
else
  echo "Reusing existing recorder venv (no upgrades)"
fi

# -----------------------------
# Recorder server env
# -----------------------------
export DATA_DIR="${DATA_DIR}"
export STATIC_DIR="${ROOTDIR}/static"
export PERSONAL_DIR="${DATA_DIR}/personal_samples"

# IMPORTANT: leave training venv creation to /api/train inside recorder_server.py
# but still set TRAIN_CMD so the server knows how to invoke training once ready
export TRAIN_CMD="source '${DATA_DIR}/.venv/bin/activate' && train_wake_word --data-dir='${DATA_DIR}'"

echo "Launching uvicorn on ${HOST}:${PORT}"
cd "${ROOTDIR}"
exec "${VENV_DIR}/bin/uvicorn" recorder_server:app --host "${HOST}" --port "${PORT}"