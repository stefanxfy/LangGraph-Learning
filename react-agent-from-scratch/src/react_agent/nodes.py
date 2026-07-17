"""Node functions and the conditional-edge router.

Verbatim translation of notebook cell-9, but split into factory
functions so tests can inject alternative tools / models:

- `make_tool_node(tools)`     -> `tool_node(state)`
- `make_call_model(model)`    -> `call_model(state, config)`
- `should_continue(state)`    -> "continue" | "end"
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence

from langchain_core.messages import SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig

from .state import AgentState

# The exact system prompt string from notebook cell-9.
SYSTEM_PROMPT = SystemMessage(
    "You are a helpful AI assistant, please respond to the users query to the best of your ability!"  # noqa: E501
)


def make_tool_node(tools: Sequence) -> Callable[[AgentState], dict]:
    """Build a LangGraph node that invokes the requested tool calls.

    Tool results are wrapped in `ToolMessage` so the next model turn sees
    the actual tool output (and the provider-side tool-call/tool-message
    pairing requirement is satisfied).
    """
    tools_by_name = {t.name: t for t in tools}

    def tool_node(state: AgentState) -> dict:
        outputs = []
        for tool_call in state["messages"][-1].tool_calls:
            tool_result = tools_by_name[tool_call["name"]].invoke(tool_call["args"])
            outputs.append(
                ToolMessage(
                    content=json.dumps(tool_result),
                    name=tool_call["name"],
                    tool_call_id=tool_call["id"],
                )
            )
        return {"messages": outputs}

    return tool_node


def make_call_model(model) -> Callable[..., dict]:
    """Build a LangGraph node that runs the chat model against the state."""

    def call_model(state: AgentState, config: RunnableConfig = None) -> dict:
        response = model.invoke([SYSTEM_PROMPT] + list(state["messages"]), config)
        # We return a list, because this will get added to the existing list.
        return {"messages": [response]}

    return call_model


def should_continue(state: AgentState) -> str:
    """Route the graph: if the last AI message contains tool_calls, continue;
    otherwise end.
    """
    messages = state["messages"]
    last_message = messages[-1]
    if not last_message.tool_calls:
        return "end"
    return "continue"
