"""Graph state schema.

Verbatim translation of notebook cell-5:
    from typing import (
        Annotated,
        Sequence,
        TypedDict,
    )
    from langchain_core.messages import BaseMessage
    from langgraph.graph.message import add_messages


    class AgentState(TypedDict):
        messages: Annotated[Sequence[BaseMessage], add_messages]
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """The state of the agent.

    `messages` is a Reducer-backed list: every node that returns
    `{"messages": [...]}` will have those messages appended rather than
    overwriting the whole field.
    """

    messages: Annotated[Sequence[BaseMessage], add_messages]
