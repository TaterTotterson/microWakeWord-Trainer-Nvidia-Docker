#!/usr/bin/env bash
set -euo pipefail

: "${NB_UID:=0}"
: "${NB_GID:=0}"
umask 002

NOTEBOOK_SRC="/root/microWakeWord_training_notebook.ipynb"
NOTEBOOK_DST="/data/microWakeWord_training_notebook.ipynb"

mkdir -p /data /data/generated_samples

if [[ ! -f "$NOTEBOOK_DST" ]]; then
  echo "No training notebook found in /data; copying default…"
  cp -n "$NOTEBOOK_SRC" "$NOTEBOOK_DST"
fi

# Try to align ownership for convenience (ignore errors if not permitted)
if [[ "$NB_UID" != "0" || "$NB_GID" != "0" ]]; then
  chown -R "$NB_UID:$NB_GID" /data || true
fi

exec "$@"