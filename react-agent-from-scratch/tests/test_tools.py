"""Tests for the notebook-fidelity `get_weather` tool (cell-7)."""

from __future__ import annotations

from react_agent.tools import ALL_TOOLS, get_weather


def test_get_weather_tool_name():
    # LangChain tool decorator exposes `.name`; the graph uses it to dispatch.
    assert get_weather.name == "get_weather"


def test_get_weather_tool_has_description():
    # The docstring is what the LLM sees to decide when to call it.
    assert "weather" in (get_weather.description or "").lower()


def test_sf_returns_sunny_joke():
    out = get_weather.invoke({"location": "San Francisco"})
    assert "sunny" in out.lower()
    # The Gemini joke is part of the canonical notebook output. Removing
    # it would be silent scope drift, so we lock it here.
    assert "Gemini" in out


def test_lowercase_sf_match():
    out = get_weather.invoke({"location": "sf"})
    assert "sunny" in out.lower()


def test_unknown_city_fallback():
    out = get_weather.invoke({"location": "Reykjavik"})
    assert "not sure" in out.lower()


def test_all_tools_list_contains_get_weather():
    assert get_weather in ALL_TOOLS
    assert len(ALL_TOOLS) == 1
