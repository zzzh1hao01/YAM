#!/usr/bin/env bash
# =============================================================================
# One-time environment bootstrap for SO100 -> YAM data collection.
#
# Creates the ai2_yam conda env (python 3.11) if missing and installs the
# three subprojects in the order upstream recommends (i2rt -> gello_software
# -> lerobot) plus the SO100 leader-bus SDK. Idempotent: safe to re-run.
#
# Usage:  ./setup_so100.sh
# After:  conda activate ai2_yam   (then see COLLECTION_QUICKSTART.md at the molmoact2 repo root)
# =============================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

command -v conda >/dev/null 2>&1 || {
  echo "FAIL: conda not found. Install miniconda first: https://docs.conda.io/en/latest/miniconda.html"
  exit 1
}

if ! conda env list | grep -qE '^\s*ai2_yam\s'; then
  echo "==> Creating conda env ai2_yam (python 3.11)"
  conda create -n ai2_yam python=3.11 -y
else
  echo "==> conda env ai2_yam already exists"
fi

run() { conda run -n ai2_yam --no-capture-output "$@"; }

echo "==> Installing i2rt (CAN motor drivers)"
run pip install -e "$REPO_ROOT/i2rt"

echo "==> Installing gello_software (teleop/collection runtime)"
run pip install -r "$REPO_ROOT/gello_software/requirements.txt"
run pip install -e "$REPO_ROOT/gello_software"

echo "==> Installing lerobot (dataset conversion)"
run pip install -e "$REPO_ROOT/lerobot"

echo "==> Installing SO100 leader-bus SDK + upload deps"
run pip install feetech-servo-sdk huggingface_hub

echo
echo "Done. Next steps:"
echo "  conda activate ai2_yam"
echo "  # then follow COLLECTION_QUICKSTART.md at the molmoact2 repo root (config placeholders, startup, collection)"
