#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(dirname "$(realpath "$0")")"
CLI_DIR="${SCRIPT_DIR}/cli"
DATA_DIR="${DATA_DIR:-/data}"

export DATA_DIR
export PATH="${CLI_DIR}:${SCRIPT_DIR}:${DATA_DIR}/.venv/bin:${PATH}"

# ---- No arguments: interactive shell ----
if [ $# -eq 0 ]; then
    exec /bin/bash -l
fi

# ---- If first arg is "train" or "train_wake_word", run the full pipeline ----
if [ "$1" = "train" ] || [ "$1" = "train_wake_word" ]; then
    shift

    if [ $# -eq 0 ]; then
        echo "Error: wake word argument required." >&2
        echo "Usage: train [options] <wake_word> [<wake_word_title>]" >&2
        exit 1
    fi

    # Phase 1: Ensure Python venv exists
    echo "===== Phase 1: Ensuring Python venv ====="
    if [ ! -f "${DATA_DIR}/.venv/bin/activate" ]; then
        "${CLI_DIR}/setup_python_venv" --data-dir="${DATA_DIR}"
    else
        echo "Python venv already exists at ${DATA_DIR}/.venv, skipping setup."
    fi
    # shellcheck source=/dev/null
    source "${DATA_DIR}/.venv/bin/activate"

    # Phase 2: Ensure training datasets are downloaded
    echo "===== Phase 2: Ensuring training datasets ====="
    "${CLI_DIR}/setup_training_datasets" --data-dir="${DATA_DIR}"

    # Phase 3: Run training
    echo "===== Phase 3: Training ====="
    "${SCRIPT_DIR}/train_wake_word" --data-dir="${DATA_DIR}" "$@"

    # Phase 4: Push to GitHub (if configured)
    echo "===== Phase 4: GitHub push (if configured) ====="
    "${SCRIPT_DIR}/github_push.sh" || true

    exit 0
fi

# ---- Anything else: execute directly ----
exec "$@"
