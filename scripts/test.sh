#!/usr/bin/env bash
set -euo pipefail
RETAIL_REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RETAIL_VENV_DIR="${RETAIL_VENV_DIR:-/workspace/.venvs/retail-forecasting}"
RETAIL_DELTA_DIR="${RETAIL_DELTA_DIR:-/workspace/.tools/delta}"
cd "$RETAIL_REPO_DIR"
export SPARK_LOCAL_IP=127.0.0.1
export RETAIL_DELTA_JARS
RETAIL_DELTA_JARS="$("$RETAIL_VENV_DIR/bin/python" scripts/prepare_delta_jars.py \
  --output-dir "$RETAIL_DELTA_DIR")"
"$RETAIL_VENV_DIR/bin/ruff" check .
"$RETAIL_VENV_DIR/bin/ruff" format --check .
"$RETAIL_VENV_DIR/bin/python" -m pytest -q
"$RETAIL_VENV_DIR/bin/python" -m build --wheel --no-isolation
