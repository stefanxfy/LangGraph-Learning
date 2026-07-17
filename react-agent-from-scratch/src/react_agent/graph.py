"""Graph assembly.

Verbatim translation of notebook cell-11, behind a factory so the
model and checkpointer can be swapped out in tests.

Graph topology:

    START -> agent
    agent -> tools   (when should_continue returns "continue")
    agent -> END     (when should_continue returns "end")
    tools -> agent
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph

from .config import make_default_model
from .nodes import make_call_model, make_tool_node, should_continue
from .state import AgentState
from .tools import ALL_TOOLS


def build_app(
    model: Any = None,
    tools: Sequence | None = None,
    checkpointer: Any | None = None,
) -> CompiledStateGraph:
    """Build and compile the ReAct graph.

    Parameters
    ----------
    model
        A LangChain chat model already bound with the desired tools.
        If `None`, the default provider model (see
        `config.make_default_model`, selected by `LLM_PROVIDER`) is used
        and `tools` are bound automatically. **Requires the selected
        provider's api key.**
    tools
        The list of tools the agent may call. Defaults to `ALL_TOOLS`.
    checkpointer
        A LangGraph checkpointer. Defaults to `InMemorySaver()` so the
        REPL can keep multiple `thread_id` conversations in one process.
    """
    if tools is None:
        tools = ALL_TOOLS

    if model is None:
        model = make_default_model().bind_tools(tools)

    checkpointer = checkpointer or InMemorySaver()

    # Define a new graph
    workflow = StateGraph(AgentState)

    # Define the two nodes we will cycle between
    workflow.add_node("agent", make_call_model(model))
    workflow.add_node("tools", make_tool_node(tools))

    # Set the entrypoint as `agent`
    workflow.set_entry_point("agent")

    # Conditional edge with two outcomes: continue (to tools) or end.
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "continue": "tools",
            "end": END,
        },
    )

    # Normal edge: after tools always return to the agent.
    workflow.add_edge("tools", "agent")

    return workflow.compile(checkpointer=checkpointer)
