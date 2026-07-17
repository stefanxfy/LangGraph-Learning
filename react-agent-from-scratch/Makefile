.PHONY: help install sync test lint format verify run clean

PY    := uv run python
PIP   := uv pip
RUFF  := uv run ruff
PYT   := uv run pytest

help:
	@echo "Targets:"
	@echo "  install  -> uv sync (install all deps)"
	@echo "  test     -> uv run pytest -q"
	@echo "  lint     -> uv run ruff check"
	@echo "  format   -> uv run ruff format"
	@echo "  verify   -> bash scripts/verify.sh (sync + tests in one shot)"
	@echo "  run      -> uv run python -m react_agent"
	@echo "  clean    -> remove caches and venv"

install sync:
	uv sync

test:
	$(PYT) -q

lint:
	$(RUFF) check

format:
	$(RUFF) format

verify:
	bash scripts/verify.sh

run:
	$(PY) -m react_agent

clean:
	rm -rf .pytest_cache .ruff_cache .venv __pycache__ */__pycache__ */*/__pycache__
	find . -name "*.pyc" -delete
