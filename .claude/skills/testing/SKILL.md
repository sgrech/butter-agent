---
name: testing
description: "Write pytest tests following project conventions — async mocking, factory fixtures, SimpleNamespace stubs"
argument-hint: <module-path> [--scope=unit|e2e]
allowed-tools: Grep, Glob, Read, Write, Edit, Bash
provenance:
  source: skill-library/python-core/testing
  catalog-version: "1.0.0"
  synced-at: "2026-05-13T05:43:30Z"
  content-hash: "sha256:ba63b2dc90a09bd56636c13f3e91a634b4e15d88cddbfd6dd873a2379e7fd57c"
  customized: false
---

## Parameters

- `module-path` (required): Path to the module to test (e.g., `services/knowledge_storage.py`)
- `scope` (optional): Test scope — `unit` (default, mocked) or `e2e` (real dependencies)

## Instructions

Write pytest tests for a module following project conventions.

### Phase 1: Understand the Module

1. Read the target module to understand its public API, dependencies, and return types
2. Read existing tests in the project to match naming, fixture, and assertion patterns
3. Check `conftest.py` files for available shared fixtures

### Phase 2: Write Tests

Follow these rules exactly:

#### Test Structure
- Test naming: `test_<action>_<scenario>` (e.g., `test_store_adds_and_flushes`, `test_get_returns_none_when_not_found`)
- Each test creates its own mocks, stubs, and service instances — no shared mutable state
- Group tests by the method they cover using comment headers: `# --------------- service tests ---------------`

#### Async Tests
- Decorator requirement depends on pytest-asyncio mode (see `[tool.pytest.ini_options] asyncio_mode` in `pyproject.toml`):
  - **`asyncio_mode = 'auto'`** — omit the decorator. All `async def test_*` functions are auto-collected as asyncio tests; adding `@pytest.mark.asyncio()` is pure ceremony with no functional difference.
  - **`asyncio_mode = 'strict'`** (the default) or **unset** — `@pytest.mark.asyncio()` is required on every async test. Note the trailing `()`, it's required. Missing this decorator under strict mode causes silent skips or confusing errors.
- Before writing the first test in a project, read `pyproject.toml` to confirm which mode is in use.

#### Database Mocking
- Unit tests use `AsyncMock()` for database sessions, never a real session
- Set up individual methods: `mock_db.add = MagicMock()`, `mock_db.flush = AsyncMock()`, `mock_db.refresh = AsyncMock()`
- For query results: `mock_result = MagicMock()` with `.scalar_one_or_none.return_value` or `.scalars.return_value.all.return_value`

#### Stubs
- Use `SimpleNamespace` for lightweight DB model stubs without SQLAlchemy overhead
- Include all fields the service reads in the stub

#### Patching
- Patch at the consuming module path, not the source: `'mypackage.services.storage._vector_repo'`, not `'mypackage.repositories.vector.VectorRepository'`
- Use `patch.object(service, 'method', new_callable=AsyncMock, return_value=...)` for method-level patching
- Use parenthesized `with (patch_a, patch_b):` for multiple patches

#### Assertions
- `pytest.raises(ExceptionType, match=r'...')` — always include `match` regex for error cases
- Verify mock call sequences: `mock_db.add.assert_called_once()`, `mock_db.flush.assert_awaited_once()`
- Use `.assert_called_once_with(expected)` or `mock.call_args.kwargs['key']` — never index into `call_args[0][1]`

#### Factory Fixtures
- Use `@pytest.fixture` functions that return factory callables
- Provide sensible defaults for all required fields, accept `**overrides`

### Canonical Example

```python
"""Tests for knowledge storage service."""

from __future__ import annotations

import uuid
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# --------------- helpers ---------------

def _make_chunk_db(content: str = 'Test content', domain: str = 'test-domain', **overrides: Any) -> Any:
    """Build a minimal ChunkDB stub for testing without a real database."""
    defaults = {
        'id': uuid.uuid4(), 'content': content, 'domain': domain,
        'source_id': uuid.uuid4(), 'tags': [], 'is_active': True,
        'vector_id': None, 'created_at': None, 'updated_at': None,
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


# --------------- service tests ---------------

# Decorator required under strict mode; omit when asyncio_mode = 'auto'.
@pytest.mark.asyncio()
async def test_store_adds_and_flushes() -> None:
    """store() should add to session, flush, and refresh."""
    service = KnowledgeStorageService()
    mock_db = AsyncMock()
    mock_db.add = MagicMock()
    mock_db.flush = AsyncMock()
    mock_db.refresh = AsyncMock()

    chunk = await service.store(mock_db, content='Test', domain='test', source='test', tags=[])

    mock_db.add.assert_called_once()
    mock_db.flush.assert_awaited_once()
    mock_db.refresh.assert_awaited_once()


@pytest.mark.asyncio()
async def test_get_returns_none_when_not_found() -> None:
    """get() should return None when no chunk matches the UUID."""
    service = KnowledgeStorageService()
    mock_db = AsyncMock()
    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = None
    mock_db.execute = AsyncMock(return_value=mock_result)

    result = await service.get(mock_db, uuid.uuid4())
    assert result is None


# --------------- error case tests ---------------

def test_enum_rejects_invalid_value() -> None:
    """Enum should reject values not in the enum."""
    with pytest.raises(ValueError, match='invalid'):
        SomeEnum('invalid')
```

### Factory Fixture Pattern (conftest.py)

```python
@pytest.fixture
def make_chunk() -> Callable[..., Any]:
    """Factory fixture that builds a ChunkDB stub with sensible defaults."""
    def _make(**overrides: object) -> Any:
        defaults: dict[str, object] = {
            'id': uuid.uuid4(), 'content': 'Default test content',
            'domain': 'test-domain', 'source_id': uuid.uuid4(),
            'tags': [], 'is_active': True,
        }
        defaults.update(overrides)
        return SimpleNamespace(**defaults)
    return _make
```

### Phase 3: Verify

- [ ] Test naming follows `test_<action>_<scenario>` convention
- [ ] Async tests match configured `pytest-asyncio` mode: omit `@pytest.mark.asyncio()` under `asyncio_mode = 'auto'`; include it under strict/unset mode
- [ ] Database mocked with `AsyncMock()` (not real session)
- [ ] Patches target the consuming module path
- [ ] `pytest.raises` includes `match=` regex
- [ ] No shared mutable state between tests
- [ ] Factory fixtures use `**overrides` for flexibility
- [ ] Mock assertions verify call sequence
- [ ] Each test creates its own service instance and mocks
- [ ] `SimpleNamespace` used for lightweight DB stubs

### Boundaries
- Do NOT modify the module under test
- Do NOT create integration/e2e tests when `--scope=unit` (default)
- Do NOT add test dependencies to pyproject.toml
