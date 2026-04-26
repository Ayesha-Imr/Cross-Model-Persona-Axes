.PHONY: install smoke full figures clean test

install:
	uv sync

smoke:
	uv run python -m mpa.run --config configs/smoke.yaml

full:
	uv run python -m mpa.run --config configs/full.yaml

dryrun:
	uv run python -m mpa.run --config configs/full.yaml --dry-run

figures:
	uv run python -m mpa.stages.visualize --config configs/full.yaml

test:
	uv run pytest -q

clean:
	rm -rf runs/ __pycache__ .pytest_cache .ruff_cache
