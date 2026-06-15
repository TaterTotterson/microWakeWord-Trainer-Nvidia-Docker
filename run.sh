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

echo "microWakeWord Trainer UI (Docker)"
echo "-> ROOTDIR:  ${ROOTDIR}"
echo "-> DATA_DIR: ${DATA_DIR}"
echo "-> URL:      http://localhost:${PORT}/"

mkdir -p "${DATA_DIR}"

install_ui_deps() {
  ${PIP} install \
    "fastapi==${FASTAPI_VERSION}" \
    "uvicorn[standard]==${UVICORN_VERSION}" \
    "python-multipart==${PY_MULTIPART_VERSION}" \
    "zeroconf>=0.132.2" \
    "silero-vad>=5.0.0" \
    "numpy>=1.24.0"
}

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
  install_ui_deps
  touch "${PIN_FILE}"
else
  echo "Reusing existing trainer UI venv (no upgrades)"
  if ! "${PY}" - "${FASTAPI_VERSION}" "${UVICORN_VERSION}" "${PY_MULTIPART_VERSION}" <<'PY' >/dev/null 2>&1
import importlib.metadata as md
import sys

fastapi_version, uvicorn_version, multipart_version = sys.argv[1:4]

def version_tuple(value):
    parts = []
    for token in str(value).replace("-", ".").split("."):
        if token.isdigit():
            parts.append(int(token))
        else:
            digits = "".join(ch for ch in token if ch.isdigit())
            if digits:
                parts.append(int(digits))
            break
    return tuple(parts)

exact = {
    "fastapi": fastapi_version,
    "uvicorn": uvicorn_version,
    "python-multipart": multipart_version,
}
minimum = {
    "silero-vad": "5.0.0",
    "numpy": "1.24.0",
    "zeroconf": "0.132.2",
}
present = ("torch",)

for package, expected in exact.items():
    if md.version(package) != expected:
        raise SystemExit(1)
for package, minimum_version in minimum.items():
    if version_tuple(md.version(package)) < version_tuple(minimum_version):
        raise SystemExit(1)
for package in present:
    md.version(package)
PY
  then
    echo "UI dependencies missing or stale; installing recorder dependencies"
    install_ui_deps
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
