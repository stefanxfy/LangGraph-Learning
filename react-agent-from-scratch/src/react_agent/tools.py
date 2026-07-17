"""Tools available to the agent.

Verbatim translation of notebook cell-7's `get_weather` definition.
The implementation is a placeholder on purpose — that is exactly how
the notebook wrote it ("Don't let the LLM know this though 😊").
"""

from __future__ import annotations

from langchain_core.tools import tool


@tool
def get_weather(location: str) -> str:
    """Call to get the weather from a specific location."""
    # This is a placeholder for the actual implementation
    # Don't let the LLM know this though 😊
    if any(city in location.lower() for city in ["sf", "san francisco"]):
        return "It's sunny in San Francisco, but you better look out if you're a Gemini 😈."  # noqa: E501
    return f"I am not sure what the weather is in {location}"


ALL_TOOLS = [get_weather]
