"""Pytest setup: make `react_agent` importable and isolate env from real keys.

These tests must NEVER call out to OpenAI. To stay safe:
- We strip all provider api keys (OPENAI/MINIMAX/ZHIPUAI) and the
  LLM_PROVIDER selector for the duration of the test session so
  accidentally hitting the network is hard.
- We add the package source dir to sys.path so `import react_agent`
  works even when the editable install has not been run.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# src/ on path so `from react_agent import ...` works without an install step.
SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def pytest_configure(config):  # noqa: D401  (pytest hook signature)
    """Unset every provider api key at test time so `require_env` /
    `make_default_model` would fail loudly if a test accidentally
    invoked them. Most tests use FakeChatModel instead."""
    for key in ("OPENAI_API_KEY", "MINIMAX_API_KEY", "ZHIPUAI_API_KEY"):
        os.environ.pop(key, None)
    # Clear the provider selector too, so routing stays deterministic.
    os.environ.pop("LLM_PROVIDER", None)
    # Also clear anything that pytest-dotenv might inject later.
    os.environ["PYTEST_DOTENV_DISABLED"] = "1"
