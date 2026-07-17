"""Provider-routing unit tests for `make_default_model()`.

We never touch the network: constructing a `ChatOpenAI` only stores the
config (model / base_url / api_key); the real HTTP call happens on
`invoke`, which we never do here. So these tests are fully offline and
assert only the routing / override logic in `config.py`.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Path bootstrap so this file runs under `python -m pytest tests/`.
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from react_agent.config import DEFAULT_PROVIDER, PROVIDERS, make_default_model


def _select(monkeypatch, provider: str, *, with_key: bool = True) -> None:
    """Point `make_default_model` at `provider` and optionally seed its key."""
    cfg = PROVIDERS[provider]
    monkeypatch.setenv("LLM_PROVIDER", provider)
    if with_key:
        monkeypatch.setenv(cfg.key_env, f"{provider}-fake-key")
    else:
        monkeypatch.delenv(cfg.key_env, raising=False)


@pytest.mark.parametrize("provider", ["zhipu", "minimax", "openai"])
def test_each_provider_routes_to_its_defaults(monkeypatch, provider):
    _select(monkeypatch, provider)
    llm = make_default_model()
    cfg = PROVIDERS[provider]
    assert llm.model_name == cfg.model_default
    # zhipu / minimax pin a base url; openai leaves it to ChatOpenAI's default.
    if cfg.base_url_default:
        assert str(llm.openai_api_base) == cfg.base_url_default


def test_model_override_env_is_respected(monkeypatch):
    _select(monkeypatch, "zhipu")
    monkeypatch.setenv("ZHIPU_MODEL", "GLM-custom")
    assert make_default_model().model_name == "GLM-custom"


def test_base_url_override_env_is_respected(monkeypatch):
    _select(monkeypatch, "minimax")
    monkeypatch.setenv("MINIMAX_BASE_URL", "https://custom.example.com/v1")
    assert str(make_default_model().openai_api_base) == "https://custom.example.com/v1"


def test_default_provider_used_when_llm_provider_unset(monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    cfg = PROVIDERS[DEFAULT_PROVIDER]
    monkeypatch.setenv(cfg.key_env, "fake")
    assert make_default_model().model_name == cfg.model_default


def test_unknown_provider_exits_with_code_2(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "nope")
    with pytest.raises(SystemExit) as exc:
        make_default_model()
    assert exc.value.code == 2


@pytest.mark.parametrize("provider", ["zhipu", "minimax", "openai"])
def test_missing_key_for_selected_provider_exits(monkeypatch, provider):
    _select(monkeypatch, provider, with_key=False)
    with pytest.raises(SystemExit) as exc:
        make_default_model()
    assert exc.value.code == 2
