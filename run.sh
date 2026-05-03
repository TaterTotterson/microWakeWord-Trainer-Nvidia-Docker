#!/usr/bin/env bash
set -euo pipefail

ROOTDIR="$(dirname "$(realpath "$0")")"

# Training convention
DATA_DIR="${DATA_DIR:-/data}"
HOST="${REC_HOST:-0.0.0.0}"
PORT="${REC_PORT:-8789}"

# Keep trainer UI deps separate from the training venv
VENV_DIR="${DATA_DIR}/.recorder-venv"
PY="${VENV_DIR}/bin/python"
PIP="${PY} -m pip"
PIN_FILE="${VENV_DIR}/.pinned_installed"

FASTAPI_VERSION="${REC_FASTAPI_VERSION:-0.115.6}"
UVICORN_VERSION="${REC_UVICORN_VERSION:-0.30.6}"
PY_MULTIPART_VERSION="${REC_PY_MULTIPART_VERSION:-0.0.9}"
ESPHOME_VERSION="${REC_ESPHOME_VERSION:-2026.4.3}"

echo "microWakeWord Trainer UI (Docker)"
echo "-> ROOTDIR:  ${ROOTDIR}"
echo "-> DATA_DIR: ${DATA_DIR}"
echo "-> URL:      http://localhost:${PORT}/"

mkdir -p "${DATA_DIR}"

# -----------------------------
# Trainer UI venv (separate)
# -----------------------------
if [[ ! -x "${PY}" ]]; then
  echo "Creating trainer UI venv: ${VENV_DIR}"
  python3 -m venv "${VENV_DIR}"
fi

# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

if [[ ! -f "${PIN_FILE}" ]]; then
  echo "Installing pinned trainer UI deps"
  ${PIP} install -U pip setuptools wheel
  ${PIP} install \
    "fastapi==${FASTAPI_VERSION}" \
    "uvicorn[standard]==${UVICORN_VERSION}" \
    "python-multipart==${PY_MULTIPART_VERSION}" \
    "esphome==${ESPHOME_VERSION}"
  touch "${PIN_FILE}"
else
  echo "Reusing existing trainer UI venv (no upgrades)"
  if ! "${PY}" - "${ESPHOME_VERSION}" <<'PY' >/dev/null 2>&1
import importlib.metadata
import sys

expected = sys.argv[1]
installed = importlib.metadata.version("esphome")
raise SystemExit(0 if installed == expected else 1)
PY
  then
    echo "Firmware tab dependencies missing or stale; installing ESPHome firmware dependencies"
    ${PIP} install \
      "fastapi==${FASTAPI_VERSION}" \
      "uvicorn[standard]==${UVICORN_VERSION}" \
      "python-multipart==${PY_MULTIPART_VERSION}" \
      "esphome==${ESPHOME_VERSION}"
  fi
fi

# -----------------------------
# Trainer server env
# -----------------------------
export DATA_DIR="${DATA_DIR}"
export STATIC_DIR="${ROOTDIR}/static"
export PERSONAL_DIR="${DATA_DIR}/personal_samples"
export CAPTURED_DIR="${DATA_DIR}/captured_audio"
export NEGATIVE_DIR="${DATA_DIR}/negative_samples"
export TRAINED_WAKE_WORDS_DIR="${DATA_DIR}/trained_wake_words"

# IMPORTANT: leave training venv creation to /api/train inside trainer_server.py
# but still set TRAIN_CMD so the server knows how to invoke training once ready
export TRAIN_CMD="source '${DATA_DIR}/.venv/bin/activate' && train_wake_word --data-dir='${DATA_DIR}'"

echo "Launching uvicorn on ${HOST}:${PORT}"
cd "${ROOTDIR}"
exec "${VENV_DIR}/bin/uvicorn" trainer_server:app --host "${HOST}" --port "${PORT}"
