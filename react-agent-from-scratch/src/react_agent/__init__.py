"""ReAct agent implemented from scratch on top of LangGraph.

This package mirrors `examples/react-agent-from-scratch.ipynb`:
- `AgentState`, `get_weather` (tool), and node functions are translated
  directly from the notebook cells.
- `build_app(...)` returns the compiled runnable.
- `python -m react_agent` starts a REPL backed by an in-memory checkpointer.
"""

from .graph import build_app

__all__ = ["build_app"]
__version__ = "0.1.0"
