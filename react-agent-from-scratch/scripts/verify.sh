#!/usr/bin/env bash
# Verify the project end-to-end without contacting OpenAI.
# Steps: install deps via uv (creates .venv) -> run pytest (uses fake chat model).
set -euo pipefail

cd "$(dirname "$0")/.."

echo ">> uv sync"
uv sync --quiet

echo ">> pytest"
uv run pytest -q
