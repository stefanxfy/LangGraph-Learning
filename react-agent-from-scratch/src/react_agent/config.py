"""Configuration: env loading + multi-provider model factory.

Generalises notebook cell-2 (`_set_env("OPENAI_API_KEY")`) and the model
instantiation at the top of cell-7 (`ChatOpenAI(model="gpt-4o-mini")`)
into a pluggable provider registry, so the REPL can run against any
OpenAI-compatible endpoint — Zhipu GLM, MiniMax, or OpenAI itself.

All three providers expose OpenAI-compatible `/chat/completions`
endpoints and support OpenAI-style `tools` (function calling), so the
graph's `.bind_tools(...)` call works unchanged; only this factory needs
to know which `base_url` + api key to use.

`LLM_PROVIDER` selects the provider (default: `zhipu`); each provider's
api key, model name, and base url can be set or overridden independently
via the `<PROVIDER>_MODEL` / `<PROVIDER>_BASE_URL` env vars. Tests
bypass this entirely by constructing their own chat model and passing
it into `build_app(...)`.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI


@dataclass(frozen=True)
class Provider:
    """Static config for one OpenAI-compatible provider.

    `base_url_default=None` means "use ChatOpenAI's official endpoint"
    (only OpenAI); the other providers always pin a base url.
    """

    key_env: str  # env var holding the api key (required)
    model_env: str  # env var overriding the model name (optional)
    model_default: str
    base_url_env: str  # env var overriding the base url (optional)
    base_url_default: str | None


# Add a provider here = add one entry; the routing logic never changes.
PROVIDERS: dict[str, Provider] = {
    "zhipu": Provider(
        key_env="ZHIPUAI_API_KEY",
        model_env="ZHIPU_MODEL",
        model_default="GLM-5.1",
        base_url_env="ZHIPU_BASE_URL",
        base_url_default="https://open.bigmodel.cn/api/coding/paas/v4",
    ),
    "minimax": Provider(
        key_env="MINIMAX_API_KEY",
        model_env="MINIMAX_MODEL",
        model_default="MinMax-M3",
        base_url_env="MINIMAX_BASE_URL",
        base_url_default="https://api.minimaxi.com/v1",
    ),
    "openai": Provider(
        key_env="OPENAI_API_KEY",
        model_env="OPENAI_MODEL",
        model_default="gpt-4o-mini",
        base_url_env="OPENAI_BASE_URL",
        base_url_default=None,  # fall through to ChatOpenAI's official endpoint
    ),
}

DEFAULT_PROVIDER = "zhipu"


def load_env() -> None:
    """Load `.env` from the current working directory if present.

    Does NOT raise if the file is missing — callers are expected to
    validate required variables explicitly with `require_env`.
    """
    load_dotenv()


def require_env(name: str) -> str:
    """Read a required env var or print a clear error and exit non-zero.

    Returning silently would mask misconfiguration, so we fail fast.
    Exit code 2 is the conventional code for "misuse / setup error".
    """
    val = os.environ.get(name)
    if not val:
        sys.stderr.write(
            f"error: required env var '{name}' is not set. "
            f"Copy .env.example to .env and fill it in, or export it "
            f"in your shell.\n"
        )
        sys.exit(2)
    return val


def make_default_model() -> ChatOpenAI:
    """Construct the chat model for the provider selected by `LLM_PROVIDER`.

    Defaults to the Zhipu GLM endpoint. Exits with code 2 (setup error)
    if `LLM_PROVIDER` is unknown or the selected provider's api key is
    missing — never silently falls back or connects with empty creds.
    """
    provider = os.getenv("LLM_PROVIDER", DEFAULT_PROVIDER).lower()
    if provider not in PROVIDERS:
        sys.stderr.write(
            f"error: unknown LLM_PROVIDER='{provider}'. "
            f"Choose one of: {', '.join(sorted(PROVIDERS))}.\n"
        )
        sys.exit(2)

    cfg = PROVIDERS[provider]
    api_key = require_env(cfg.key_env)
    model = os.getenv(cfg.model_env, cfg.model_default)
    base_url = os.getenv(cfg.base_url_env, cfg.base_url_default)

    kwargs: dict[str, str] = {"model": model, "api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url
    return ChatOpenAI(**kwargs)
