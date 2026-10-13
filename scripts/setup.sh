#!/usr/bin/env bash
set -euo pipefail
RETAIL_REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RETAIL_VENV_DIR="${RETAIL_VENV_DIR:-/workspace/.venvs/retail-forecasting}"
RETAIL_DELTA_DIR="${RETAIL_DELTA_DIR:-/workspace/.tools/delta}"
python3 -m venv "$RETAIL_VENV_DIR"
"$RETAIL_VENV_DIR/bin/python" -m pip install --no-input -r "$RETAIL_REPO_DIR/requirements.lock"
"$RETAIL_VENV_DIR/bin/python" -m pip install --no-input --no-deps --no-build-isolation -e "$RETAIL_REPO_DIR"
mkdir -p "$RETAIL_DELTA_DIR"
"$RETAIL_VENV_DIR/bin/python" "$RETAIL_REPO_DIR/scripts/prepare_delta_jars.py" \
  --output-dir "$RETAIL_DELTA_DIR"
"$RETAIL_VENV_DIR/bin/python" -m pip check
