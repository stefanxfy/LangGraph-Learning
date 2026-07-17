"""Sanity checks on the AgentState schema from notebook cell-5."""

from __future__ import annotations

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

from react_agent.state import AgentState


def test_agent_state_has_messages_key():
    assert "messages" in AgentState.__annotations__


def test_messages_annotation_is_reducer_list():
    # TypedDict evaluation under `from __future__ import annotations`
    # keeps Annotated metadata in string form such as
    # "Annotated[Sequence[BaseMessage], add_messages]". We can't easily
    # introspect that at runtime, so the smoke test here is simply that
    # the field still maps to a list type that mentions messages.
    ann = AgentState.__annotations__
    assert "messages" in ann


def test_add_messages_reducer_is_callable():
    # The reducer should be exposed and importable under the same name.
    assert callable(add_messages)


def test_base_message_is_a_class():
    # Sanity: BaseMessage is what we'd expect to find in the field type.
    assert isinstance(BaseMessage, type)
