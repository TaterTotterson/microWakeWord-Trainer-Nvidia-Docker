#!/bin/bash
set -euo pipefail

DATA_DIR="${DATA_DIR:-/data}"
OUTPUT_DIR="${DATA_DIR}/output"

# Skip if not configured
if [ -z "${GITHUB_TOKEN:-}" ] || [ -z "${GITHUB_REPO:-}" ]; then
    echo "GitHub push not configured (set GITHUB_TOKEN and GITHUB_REPO to enable)."
    exit 0
fi

GITHUB_BRANCH="${GITHUB_BRANCH:-main}"
GITHUB_PATH="${GITHUB_PATH:-.}"

# Find latest output directory
LATEST_OUTPUT="$(find "${OUTPUT_DIR}" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | sort | tail -1)"
if [ -z "${LATEST_OUTPUT}" ]; then
    echo "No output directory found in ${OUTPUT_DIR}, skipping GitHub push."
    exit 0
fi

# Check that there are actually files to push
TFLITE_COUNT="$(find "${LATEST_OUTPUT}" -maxdepth 1 -name '*.tflite' 2>/dev/null | wc -l)"
JSON_COUNT="$(find "${LATEST_OUTPUT}" -maxdepth 1 -name '*.json' 2>/dev/null | wc -l)"
if [ "${TFLITE_COUNT}" -eq 0 ] && [ "${JSON_COUNT}" -eq 0 ]; then
    echo "No .tflite or .json files found in ${LATEST_OUTPUT}, skipping GitHub push."
    exit 0
fi

echo "Pushing artifacts from ${LATEST_OUTPUT} to ${GITHUB_REPO}@${GITHUB_BRANCH}:${GITHUB_PATH}"

TMPDIR="$(mktemp -d)"
trap 'rm -rf "${TMPDIR}"' EXIT

REPO_URL="https://x-access-token:${GITHUB_TOKEN}@github.com/${GITHUB_REPO}.git"

echo "Cloning ${GITHUB_REPO} (branch: ${GITHUB_BRANCH})..."
if ! git clone --depth 1 --branch "${GITHUB_BRANCH}" "${REPO_URL}" "${TMPDIR}/repo" 2>&1; then
    echo "Error: Failed to clone repository. Check GITHUB_TOKEN and GITHUB_REPO." >&2
    exit 1
fi

TARGET="${TMPDIR}/repo/${GITHUB_PATH}"
mkdir -p "${TARGET}"

cp "${LATEST_OUTPUT}"/*.tflite "${TARGET}/" 2>/dev/null || true
cp "${LATEST_OUTPUT}"/*.json "${TARGET}/" 2>/dev/null || true

cd "${TMPDIR}/repo"
git add -A

# Derive wake word name from output directory for commit message
DIRNAME="$(basename "${LATEST_OUTPUT}")"
WAKE_WORD="$(echo "${DIRNAME}" | sed 's/^[0-9_T-]*-//' | sed 's/-[0-9]*-[0-9]*$//')"
COMMIT_MSG="${GITHUB_COMMIT_MSG:-"Add trained model: ${WAKE_WORD}"}"

git -c user.name="mww-trainer" -c user.email="mww-trainer@noreply" commit -m "${COMMIT_MSG}" || {
    echo "Nothing to commit (model files unchanged)."
    exit 0
}

echo "Pushing to ${GITHUB_REPO}@${GITHUB_BRANCH}..."
git push origin "${GITHUB_BRANCH}"
echo "Successfully pushed model to ${GITHUB_REPO}."
