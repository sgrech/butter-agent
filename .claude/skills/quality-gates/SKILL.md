---
name: python-quality-gates
description: "Run Python quality gates — ruff check + format, mypy, pytest. Prefers just check / just fix when a justfile is present, falls back to direct uv run commands otherwise."
argument-hint: [--fix] [--no-tests]
allowed-tools: Bash, Read
provenance:
  source: skill-library/python-core/quality-gates
  catalog-version: "1.0.0"
  synced-at: "2026-05-13T05:43:30Z"
  content-hash: "sha256:f0a36f477326b859d3e368485804d078f994b0f9e4e87dfa8b15c00d12a4d4be"
  customized: false
---

## Parameters

- `--fix` (optional): Auto-fix lint and format issues before reporting (`just fix` or `uv run ruff check --fix . && uv run ruff format .`)
- `--no-tests` (optional): Skip the pytest suite — useful for fast lint/type checks during iteration

## When to invoke

- After writing or editing Python code
- Before committing, pushing, or shipping
- When the user asks "check it", "run quality gates", "is this clean", "lint this", etc.

## Workflow

### Phase 1: Detect available tooling

1. Check for `justfile` at the project root. If present, prefer `just` recipes (`just check`, `just fix`).
2. Check for `pyproject.toml` to confirm this is a Python project. If absent, report and stop — this is not a Python project.
3. Verify `uv` is installed via `uv --version`. If absent, report and stop.

### Phase 2: Run the gates

**With justfile (preferred):**
```bash
just check                    # Run all configured gates
# or, when --fix is passed:
just fix && just check
```

**Without justfile (fallback):**
```bash
uv run ruff check .           # Lint
uv run ruff format --check .  # Format check
uv run mypy .                 # Type check
uv run pytest                 # Tests (skip if --no-tests was passed)
```

When `--fix` is passed without a justfile:
```bash
uv run ruff check --fix .     # Auto-fix lint
uv run ruff format .          # Auto-format
# then run the check sequence above
```

### Phase 3: Report

Report each gate's status with a tight summary:

```
Quality gates — <project-name>
- ruff check     ✓ pass (0 issues)
- ruff format    ✓ pass
- mypy           ✗ fail (3 errors)
- pytest         ✓ pass (47 passed, 0 failed)
```

For failing gates:
- Surface the first 5 errors verbatim (file:line + message)
- Ask whether to attempt fixes
- Do NOT bury failures in summary prose

## Boundaries

- Do NOT modify project source code unless `--fix` is passed.
- Do NOT skip a failing gate to make the run look clean. Report every failure.
- Do NOT add or upgrade tooling versions — gates run with whatever is in `pyproject.toml`.
- If a gate is misconfigured (e.g. mypy strict mode breaking on legitimate code), report it transparently; do not silence it.
- Do NOT chain into `git-commit` or other shipping workflows automatically — that is the caller's decision.
