"""Interactive REPL entry: `python -m react_agent`.

Behaviour:
- Reads the selected provider's api key from `.env` (or the environment)
  via `LLM_PROVIDER` (default `zhipu`). Exits with code 2 and a clear
  message if the provider is unknown or its key is missing — never
  connects anywhere.
- Builds the graph once with an in-memory checkpointer.
- Each user message is sent under a fresh config thread; typing
  `:clear` resets the thread id and discards the conversation history.
- Empty lines are ignored. `:quit`, Ctrl-C, or EOF exits cleanly.

This is a multi-turn extension of notebook cell-13's single-shot
`print_stream(graph.stream(...))` call.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterable

from langchain_core.messages import BaseMessage

from .config import load_env
from .graph import build_app

BANNER = (
    "ReAct REPL — type a message and press Enter.\n"
    "  :clear  reset the conversation history (new thread)\n"
    "  :quit   exit the REPL\n"
    "  Ctrl-D  exit the REPL"
)


def _format_message(msg: BaseMessage | tuple) -> str:
    """Render a single streamed message.

    LangGraph's `stream_mode="values"` can emit either a BaseMessage or
    a legacy tuple; we mirror what notebook cell-13's `print_stream`
    did for both cases.
    """
    if isinstance(msg, tuple):
        return str(msg)
    return msg.pretty_print()


def print_stream(stream: Iterable[dict]) -> None:
    """Print each state snapshot from `app.stream(..., stream_mode="values")`.

    Only the most recently appended message is pretty-printed to avoid
    spamming the terminal on multi-turn conversations.
    """
    for s in stream:
        msg = s["messages"][-1]
        print(_format_message(msg))


def repl() -> None:
    load_env()
    # `build_app()` with no model uses the default provider model and binds
    # ALL_TOOLS to it; it fails fast (exit 2) on a bad/missing key before we
    # print the banner. (Passing our own un-bound model here would skip
    # `.bind_tools(...)` and the agent would never see its tools.)
    app = build_app()

    print(BANNER)

    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    while True:
        try:
            user_input = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nbye")
            return

        if not user_input:
            continue
        if user_input == ":quit":
            print("bye")
            return
        if user_input == ":clear":
            thread_id = str(uuid.uuid4())
            config = {"configurable": {"thread_id": thread_id}}
            print("(history cleared)")
            continue

        inputs = {"messages": [("user", user_input)]}
        print_stream(app.stream(inputs, config, stream_mode="values"))


if __name__ == "__main__":
    repl()
