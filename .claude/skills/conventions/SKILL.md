---
name: python-conventions
description: "Apply Python project conventions — UV package manager, ruff style (per-project pyproject.toml), type hints on signatures, Google-style docstrings. Use when writing or editing Python code in a project with a pyproject.toml."
allowed-tools: Read, Edit, Grep, Glob, Bash
provenance:
  source: skill-library/python-core/conventions
  catalog-version: "1.0.0"
  synced-at: "2026-05-13T05:43:30Z"
  content-hash: "sha256:c5edd15d6146ca5846d9c5810e0db642a4fb046ae42d486bbded32a4c0c34747"
  customized: false
---

## When to apply

Apply these conventions whenever writing new Python code or editing existing code in a project that has a `pyproject.toml`. They are the baseline expectations for code that will pass the project's quality gates. For automated verification, see the `python-quality-gates` skill.

## Package Management

- **Always use `uv`** as the Python package manager. Never invoke `pip`, `pipenv`, or `poetry` directly.
- **Run project tools via `uv run`** (e.g. `uv run pytest`, `uv run ruff check .`). This ensures the project virtualenv is active.
- **Add dependencies via `uv add <pkg>`** — or `uv add --dev <pkg>` for dev-only deps. Never edit `pyproject.toml` dependency lists by hand.
- **Lock file**: `uv.lock` is committed and authoritative. Never delete it; never edit it manually.

## Code Style

### Strings

Quote style is driven by `[tool.ruff.format]` in the project's `pyproject.toml`. Read it before editing; do not impose a quote style the project did not choose.

- If `quote-style = "single"` (or unset — library default): single quotes for all string literals (`'foo'`, not `"foo"`).
- If `quote-style = "double"`: double quotes throughout.
- Triple double-quotes for docstrings regardless of quote-style (`"""..."""`).
- f-strings are fine and preferred over `.format()` or `%`-formatting.
- Enforced by `ruff format` — do not fight the formatter.

### Line Length

Line length is driven by `[tool.ruff] line-length` in the project's `pyproject.toml`.

- If unset: the library default is **408 characters** (AI-generation friendly — rarely wraps).
- Project may override to any value. Common portfolio values: 88 (Black default), 100, 120, 408.
- Wrap long signatures, long string concatenation, and long dict literals across multiple lines when readability benefits — but do not artificially break short lines, and do not exceed the project's configured limit.
- When editing an existing project for the first time in a session, read the project's `line-length` from `pyproject.toml` before writing code.

### Type Hints
- **All function signatures** must have type hints on parameters and return types
- All class attributes must be annotated (`name: str = 'default'`)
- Use `from __future__ import annotations` at the top of new modules for forward-compatible syntax
- Prefer built-in generics (`list[str]`, `dict[str, int]`) over `typing.List` / `typing.Dict`
- Use `X | None` over `Optional[X]`

### Docstrings
- **Google-style format** for public functions, classes, and modules
- Required for: public functions, public classes, modules with non-trivial purpose
- Not required for: private helpers (prefixed `_`), simple property accessors, obvious one-liners
- Format:
  ```python
  def fetch_user(user_id: int) -> User:
      """Retrieve a user by ID.

      Args:
          user_id: The numeric user identifier.

      Returns:
          The matching User object.

      Raises:
          UserNotFoundError: If no user exists with that ID.
      """
  ```

## Imports

- Sort imports in three groups separated by blank lines: stdlib, third-party, local
- ruff handles this automatically (isort-compatible) — do not sort by hand
- Prefer absolute imports over relative within the project
- Avoid `from x import *`

## Verification checklist

Before considering Python code complete:

- [ ] Quote style matches `[tool.ruff.format] quote-style` in `pyproject.toml` (or single quotes if unset)
- [ ] All function signatures have type hints
- [ ] Public functions have Google-style docstrings
- [ ] No lines exceed `[tool.ruff] line-length` in `pyproject.toml` (or 408 if unset)
- [ ] Imports sorted into stdlib / third-party / local groups
- [ ] No `pip` / `pipenv` / `poetry` invocations introduced
- [ ] No hand-edits to `pyproject.toml` dependency lists or `uv.lock`

For automated verification of these rules, invoke `python-quality-gates`.

## Boundaries

- This skill describes conventions; it does not enforce them. Enforcement is via ruff / mypy / pytest in the `python-quality-gates` skill.
- Do NOT change conventions on a per-project basis without updating this skill via `/skills-propose`. Project-specific deltas belong in the project's local `CLAUDE.md`, not in this skill.
- Do NOT add new conventions here that aren't already standard across the portfolio. This skill is a baseline, not a kitchen sink.
