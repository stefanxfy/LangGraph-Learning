# ReAct agent from scratch (real project)

This is a **runnable Python project** translation of
[`examples/react-agent-from-scratch.ipynb`](../../react-agent-from-scratch.ipynb)
in the parent directory.

The notebook is the canonical "implement ReAct from scratch"
walk-through for LangGraph. We took every code cell and lifted it
into a real `src/` package with:

- proper state / tool / node / graph modules
- a `python -m react_agent` REPL with `:clear` / `:quit` and
  per-thread conversation memory (in-memory checkpointer)
- a pytest suite that runs the **full ReAct loop without ever
  calling OpenAI**, by substituting `langchain_core`'s built-in
  `FakeMessagesListChatModel`
- a `bash scripts/verify.sh` one-shot that runs `uv sync` + `pytest`

## Layout

```
examples/codes/react-agent-from-scratch/
├── pyproject.toml          # uv + hatchling; package under src/
├── README.md
├── .env.example            # template: LLM_PROVIDER + provider keys
├── .gitignore
├── Makefile                # make test / lint / format / run / verify
├── scripts/
│   └── verify.sh           # uv sync + pytest
├── src/react_agent/
│   ├── __init__.py
│   ├── __main__.py         # REPL entry
│   ├── config.py           # .env loading + env-var validation + model factory
│   ├── state.py            # AgentState (notebook cell-5)
│   ├── tools.py            # get_weather (notebook cell-7)
│   ├── nodes.py            # tool_node, call_model, should_continue (cell-9)
│   └── graph.py            # StateGraph assembly (cell-11)
└── tests/
    ├── conftest.py
    ├── test_state.py
    ├── test_tools.py
    ├── test_graph_construction.py
    └── test_react_loop.py  # FakeChatModel-based end-to-end ReAct loop
```

## Cell-by-cell map

| Notebook cell | Project file | Notes |
|---|---|---|
| cell-5 (`AgentState`)           | `src/react_agent/state.py`   | reducer `add_messages` preserved |
| cell-7 (`get_weather`, `model = ChatOpenAI(...)`, `model.bind_tools(tools)`) | `src/react_agent/tools.py`, `src/react_agent/config.py`, `src/react_agent/graph.py` | tool body (incl. Gemini joke) **verbatim**; model factory uses `gpt-4o-mini` |
| cell-9 (`tool_node`, `call_model`, `should_continue`) | `src/react_agent/nodes.py`  | factory split so tests can inject tools / models |
| cell-11 (graph assembly)        | `src/react_agent/graph.py`  | adds default `InMemorySaver` |
| cell-13 (`print_stream(graph.stream(...))`) | `src/react_agent/__main__.py` | upgraded to multi-turn REPL with thread_id |

## Installation

```bash
cd examples/codes/react-agent-from-scratch
uv sync
```

`uv` resolves `langgraph`, `langchain-core`, `langchain-openai`,
`python-dotenv`, `pytest`, and `ruff` into a project-local `.venv`.

## Running the REPL

```bash
cp .env.example .env       # then edit .env: set LLM_PROVIDER + its key
make run                   # equivalent to: uv run python -m react_agent
```

`LLM_PROVIDER` picks the model backend — `zhipu` (default), `minimax`, or
`openai`. All three are OpenAI-compatible, so only the selected provider's
api key is required; model and endpoint can be overridden per provider:

| `LLM_PROVIDER` | Required env var  | Default model | Default base url                            |
|----------------|-------------------|---------------|---------------------------------------------|
| `zhipu`        | `ZHIPUAI_API_KEY` | `GLM-5.1`     | `https://open.bigmodel.cn/api/coding/paas/v4` |
| `minimax`      | `MINIMAX_API_KEY` | `MinMax-M3`   | `https://api.minimaxi.com/v1`               |
| `openai`       | `OPENAI_API_KEY`  | `gpt-4o-mini` | *(official OpenAI endpoint)*                |

Override the model or endpoint any time with `<PROVIDER>_MODEL` /
`<PROVIDER>_BASE_URL` (e.g. `ZHIPU_MODEL=glm-4.6`). Model names are
case-sensitive at the API — if a provider rejects the default, set the
exact name via the matching `_MODEL` var, no code change needed.

Inside the REPL:

```
> what's the weather in sf?
>  # you'll see streamed HumanMessage → AIMessage(tool_call) → ToolMessage → AIMessage(sunny)
> :clear
> :quit
```

An unknown `LLM_PROVIDER`, or a missing api key for the selected provider,
makes the program exit with code **2** and a clear stderr message — no
silent fallbacks, no surprise bill.

## Tests

```bash
make test                  # uv run pytest -q
bash scripts/verify.sh     # uv sync + pytest in one shot
```

Tests assert:

- `get_weather` returns the SF joke string verbatim (locks notebook semantics).
- The compiled graph contains `agent` and `tools` nodes and a
  conditional edge mapping `continue → tools`, `end → END`.
- A scripted `(tool_call → final)` chat run produces a
  HumanMessage + ToolMessage + AIMessage sequence via the full ReAct loop
  — **without any OpenAI calls**.

## Linting / formatting

```bash
make lint     # ruff check
make format   # ruff format
```

## What this project deliberately does NOT do

- No real weather API — `get_weather` is the placeholder from the
  notebook, on purpose. Swapping in `wttr.in` or similar is a
  one-line change in `tools.py`.
- No `create_react_agent` prebuilt — the notebook's whole point is
  "from scratch", and we honour that.
- No LangSmith tracing — the agent does not depend on it.
- No web / FastAPI layer — the user requested a REPL.
