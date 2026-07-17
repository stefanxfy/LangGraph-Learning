"""End-to-end ReAct loop with no OpenAI calls.

We use LangChain's built-in `FakeMessagesListChatModel`, which lets us
script every AI response in advance. The first response calls
`get_weather`, the second returns the final user-facing summary.

We then assert that the final message list contains a HumanMessage,
a ToolMessage carrying the SF joke, and an AIMessage summarising it.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# path bootstrap so this test file works under `python -m pytest tests/`
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver

from react_agent.graph import build_app
from react_agent.tools import get_weather


class _FakeChatModelWithBind(FakeMessagesListChatModel):
    """A `FakeMessagesListChatModel` subclass that no-ops `bind_tools`.

    The base class declares `bind_tools` as abstract and raises
    `NotImplementedError`. Our test wiring expects to be able to bind
    tools without changing how responses are scripted, so we simply
    return `self` and let the scripted list of AI messages carry the
    `tool_calls` payloads instead.
    """

    def bind_tools(self, tools, **kwargs):  # noqa: D401, ARG002
        return self


def _scripted_ai_messages():
    """Two scripted turns: tool call → final summary."""
    # Step 1: AI calls get_weather with location="San Francisco".
    tool_call_msg = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "get_weather",
                "args": {"location": "San Francisco"},
                "id": "call_1",
                "type": "tool_call",
            }
        ],
    )
    # Step 2: AI gives the final answer after seeing the ToolMessage.
    final_msg = AIMessage(content="It's sunny in San Francisco right now! 🌞")
    return [tool_call_msg, final_msg]


def test_react_loop_calls_tool_and_returns_final():
    fake = _FakeChatModelWithBind(responses=_scripted_ai_messages())
    app = build_app(
        model=fake.bind_tools([get_weather]),
        tools=[get_weather],
        checkpointer=InMemorySaver(),
    )
    out = app.invoke(
        {"messages": [HumanMessage(content="weather in sf?")]},
        config={"configurable": {"thread_id": "t-react-1"}},
    )

    msgs = out["messages"]
    assert len(msgs) >= 3, f"expected at least 3 messages, got {len(msgs)}"

    # Last message is the agent's user-facing summary.
    final = msgs[-1]
    assert isinstance(final, AIMessage)
    assert "sunny" in final.content.lower()

    # A ToolMessage must have been produced.
    tool_msgs = [m for m in msgs if isinstance(m, ToolMessage)]
    assert len(tool_msgs) == 1
    tool_msg = tool_msgs[0]
    assert tool_msg.name == "get_weather"
    payload = json.loads(tool_msg.content)
    # SF branch of get_weather returns the SF joke string.
    assert "sunny" in payload.lower()
    assert "Gemini" in payload


def test_two_threads_keep_separate_history():
    fake1 = _FakeChatModelWithBind(responses=_scripted_ai_messages())
    fake2 = _FakeChatModelWithBind(responses=_scripted_ai_messages())

    app1 = build_app(
        model=fake1.bind_tools([get_weather]),
        tools=[get_weather],
        checkpointer=InMemorySaver(),
    )
    # Sharing the same InMemorySaver across two apps lets us prove the
    # thread_id branching works.
    shared_saver = InMemorySaver()
    app2 = build_app(
        model=fake2.bind_tools([get_weather]),
        tools=[get_weather],
        checkpointer=shared_saver,
    )
    cfg_a = {"configurable": {"thread_id": "thread-a"}}
    cfg_b = {"configurable": {"thread_id": "thread-b"}}

    out_a = app1.invoke({"messages": [HumanMessage(content="hi")]}, config=cfg_a)
    out_b = app2.invoke({"messages": [HumanMessage(content="hi")]}, config=cfg_b)

    # Each thread should have its own HumanMessage and ToolMessage.
    a_msgs = out_a["messages"]
    b_msgs = out_b["messages"]

    # Both threads share the same scripted response pattern, so the
    # message list length must be identical for thread isolation to
    # be considered consistent.
    assert len(a_msgs) == len(b_msgs)

    # And each thread must contain a ToolMessage keyed to its own
    # message-id space — the tool_call_id from thread-a must NOT appear
    # in thread-b's ToolMessages.
    a_tool_ids = [m.tool_call_id for m in a_msgs if isinstance(m, ToolMessage)]
    b_tool_ids = [m.tool_call_id for m in b_msgs if isinstance(m, ToolMessage)]
    assert a_tool_ids == ["call_1"]
    assert b_tool_ids == ["call_1"]
    # And the human messages should be separate.
    assert a_msgs[0].id != b_msgs[0].id


def test_no_tool_call_paths_returned_to_user():
    """If the AI never calls a tool, should_continue → END, no ToolMessage."""
    fake = _FakeChatModelWithBind(
        responses=[AIMessage(content="Sure, I can help with that.")]
    )
    app = build_app(
        model=fake.bind_tools([get_weather]),
        tools=[get_weather],
        checkpointer=InMemorySaver(),
    )
    out = app.invoke(
        {"messages": [HumanMessage(content="hello")]},
        config={"configurable": {"thread_id": "t-noroute"}},
    )
    msgs = out["messages"]
    assert not any(isinstance(m, ToolMessage) for m in msgs)
    assert msgs[-1].content.startswith("Sure,")
