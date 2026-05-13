default:
    @just --list

check:
    uv run ruff check .
    uv run ruff format --check .
    uv run mypy .
    uv run pytest

fix:
    uv run ruff check --fix .
    uv run ruff format .

test:
    uv run pytest

lint:
    uv run ruff check .

types:
    uv run mypy .
