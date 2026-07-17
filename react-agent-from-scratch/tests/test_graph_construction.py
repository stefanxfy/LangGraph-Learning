"""Structural tests on the compiled graph.

We don't call out to OpenAI: a fake chat model supplies canned
responses. That keeps the test cheap and offline.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make tests runnable without `pip install -e` by adding src/ explicitly.
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from langgraph.checkpoint.memory import InMemorySaver

from react_agent.graph import build_app


class _StaticFakeChatModel:
    """Minimal LangChain-compatible chat model returning a single message.

    Returns the provided BaseMessage on every `invoke` call. We use it
    only for *structural* assertions here; the full ReAct loop is
    covered in test_react_loop.py.
    """

    def __init__(self, response):
        self._response = response

    def invoke(self, messages, config=None):  # noqa: D401
        return self._response

    # `bind_tools` is called inside `build_app` when no model is passed;
    # we always pass a pre-bound-compatible fake instead, so no need here.
    def bind_tools(self, tools):  # pragma: no cover
        return self


def test_build_app_returns_compiled_graph():
    from langchain_core.messages import AIMessage

    fake = _StaticFakeChatModel(AIMessage(content="hello"))
    app = build_app(model=fake, checkpointer=InMemorySaver())
    # CompiledStateGraph exposes `.invoke` and `.stream`.
    assert hasattr(app, "invoke")
    assert hasattr(app, "stream")


def test_graph_has_agent_and_tools_nodes():
    from langchain_core.messages import AIMessage

    fake = _StaticFakeChatModel(AIMessage(content="hi"))
    app = build_app(model=fake, checkpointer=InMemorySaver())
    node_names = {n.name for n in app.get_graph().nodes.values()}
    assert {"agent", "tools"}.issubset(node_names)


def test_conditional_edge_keys():
    """The conditional edge mapping should drive tools vs END."""
    from langchain_core.messages import AIMessage

    fake = _StaticFakeChatModel(AIMessage(content="hi"))
    app = build_app(model=fake, checkpointer=InMemorySaver())
    # Edges are named Edge objects in langgraph 1.x; attr-access their fields.
    edges = app.get_graph().edges
    edge_pairs = {(e.source, e.target) for e in edges}
    # Required structural edges
    assert ("__start__", "agent") in edge_pairs
    assert ("tools", "agent") in edge_pairs
    # Conditional edges carry the mapping-string in `.data`
    cond = {(e.source, e.target, e.data) for e in edges if e.conditional}
    assert ("agent", "tools", "continue") in cond
    assert ("agent", "__end__", "end") in cond


def test_invoke_runs_a_single_turn():
    from langchain_core.messages import AIMessage

    fake = _StaticFakeChatModel(AIMessage(content="hi back"))
    app = build_app(model=fake, checkpointer=InMemorySaver())
    out = app.invoke(
        {"messages": [("user", "hi")]},
        config={"configurable": {"thread_id": "t1"}},
    )
    assert out["messages"][-1].content == "hi back"
