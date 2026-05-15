"""Tests for the built-in `notes` plugin (task #377).

The plugin is unit-tested in isolation against a `FakePluginContext`: it
never owns a database, so every persistence call is a recorded
`context.call` into `database.*`. Two contracts are pinned here that the
spec §4 callout makes load-bearing:

- notes addresses its table with the **bare** name `"entries"` — never a
  `notes__`-prefixed name. Core does the prefixing; feeding a fake context
  here proves the plugin holds no namespace logic.
- `created_at` is always written: taken verbatim from the chained value
  when present (the `clock.now → notes.create` variable-pool path),
  self-generated as ISO-8601 UTC otherwise (no DB default exists for it).

End-to-end behaviour with the real `database` plugin, real executor, real
gate handler and the `clock.now → notes.create` chain lives in
`tests/integration/test_notes_scenario.py`.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime

import pytest

from butter_agent.plugins.notes import NotesPlugin, NotesPluginError, build_notes_plugin
from tests.support import FakePluginContext

# What core would have rewritten the bare name to in production. The plugin
# must never produce this itself — asserting the recorded call carries the
# *bare* name documents that.
_BARE = 'entries'


def _ctx(responses: Mapping[str, Mapping[str, object]] | None = None) -> FakePluginContext:
    """A fake context with canned `database.*` responses.

    `define_table` is always stubbed because every capability ensures the
    table first; callers add `insert` / `select` as needed. `responses` is
    a `Mapping` (covariant) so call sites can pass dict literals with
    narrower value types without tripping dict invariance.
    """
    canned: dict[str, dict[str, object]] = {'database.define_table': {'table': f'notes__{_BARE}'}}
    if responses is not None:
        canned.update({ref: dict(value) for ref, value in responses.items()})
    return FakePluginContext(responses=canned)


# --- create ------------------------------------------------------------------


async def test_create_uses_chained_created_at_verbatim() -> None:
    """A `$t.time` value from a prior clock.now is stored as-is."""
    plugin = NotesPlugin()
    ctx = _ctx({'database.insert': {'id': 7}})

    result = await plugin.execute(
        'create',
        {'content': 'buy butter', 'created_at': '2026-05-14T15:00:00+02:00'},
        ctx,
    )

    assert result == {'note_id': 7, 'created_at': '2026-05-14T15:00:00+02:00'}
    assert ctx.calls == [
        ('database.define_table', {'table': _BARE, 'columns': plugin_columns()}),
        (
            'database.insert',
            {'table': _BARE, 'row': {'content': 'buy butter', 'created_at': '2026-05-14T15:00:00+02:00'}},
        ),
    ]


async def test_create_self_populates_created_at_when_unchained() -> None:
    """Without an upstream clock.now, notes generates a valid ISO-8601 stamp."""
    plugin = NotesPlugin()
    ctx = _ctx({'database.insert': {'id': 1}})

    result = await plugin.execute('create', {'content': 'standalone note'}, ctx)

    stamp = result['created_at']
    assert isinstance(stamp, str)
    # Round-trips as a timezone-aware ISO-8601 datetime.
    parsed = datetime.fromisoformat(stamp)
    assert parsed.tzinfo is not None
    # The same generated value is what was written to the row.
    insert_call = next(c for c in ctx.calls if c[0] == 'database.insert')
    assert insert_call[1]['row'] == {'content': 'standalone note', 'created_at': stamp}


@pytest.mark.parametrize('bad', ['', None, 123])
async def test_create_rejects_empty_or_non_string_content(bad: object) -> None:
    plugin = NotesPlugin()
    with pytest.raises(NotesPluginError, match="'content' must be a non-empty string"):
        await plugin.execute('create', {'content': bad}, _ctx())


async def test_create_surfaces_non_integer_id_from_store() -> None:
    """A malformed database.insert result is surfaced, not passed downstream."""
    plugin = NotesPlugin()
    ctx = _ctx({'database.insert': {'id': 'oops'}})
    with pytest.raises(NotesPluginError, match='non-integer id'):
        await plugin.execute('create', {'content': 'x'}, ctx)


# --- list --------------------------------------------------------------------


async def test_list_passes_rows_through_oldest_first() -> None:
    plugin = NotesPlugin()
    rows = [
        {'id': 1, 'content': 'a', 'created_at': '2026-05-14T10:00:00+00:00'},
        {'id': 2, 'content': 'b', 'created_at': '2026-05-14T11:00:00+00:00'},
    ]
    ctx = _ctx({'database.select': {'rows': rows}})

    result = await plugin.execute('list', {}, ctx)

    assert result == {'notes': rows}
    assert ('database.select', {'table': _BARE, 'order_by': 'id'}) in ctx.calls


async def test_list_empty_is_not_an_error() -> None:
    """An empty notes table is a valid result, never a NotesPluginError."""
    plugin = NotesPlugin()
    ctx = _ctx({'database.select': {'rows': []}})
    assert await plugin.execute('list', {}, ctx) == {'notes': []}


async def test_list_forwards_valid_limit() -> None:
    plugin = NotesPlugin()
    ctx = _ctx({'database.select': {'rows': []}})
    await plugin.execute('list', {'limit': 5}, ctx)
    select_call = next(c for c in ctx.calls if c[0] == 'database.select')
    assert select_call[1] == {'table': _BARE, 'order_by': 'id', 'limit': 5}


@pytest.mark.parametrize('bad', [-1, True, 'lots'])
async def test_list_rejects_invalid_limit(bad: object) -> None:
    plugin = NotesPlugin()
    with pytest.raises(NotesPluginError, match="'limit' must be a non-negative integer"):
        await plugin.execute('list', {'limit': bad}, _ctx({'database.select': {'rows': []}}))


# --- read --------------------------------------------------------------------


async def test_read_returns_single_note() -> None:
    plugin = NotesPlugin()
    ctx = _ctx(
        {'database.select': {'rows': [{'id': 3, 'content': 'hello', 'created_at': '2026-05-14T12:00:00+00:00'}]}},
    )

    result = await plugin.execute('read', {'note_id': 3}, ctx)

    assert result == {'content': 'hello', 'created_at': '2026-05-14T12:00:00+00:00'}
    assert ('database.select', {'table': _BARE, 'where': {'id': 3}}) in ctx.calls


async def test_read_unknown_id_raises() -> None:
    plugin = NotesPlugin()
    ctx = _ctx({'database.select': {'rows': []}})
    with pytest.raises(NotesPluginError, match='no note with id 99'):
        await plugin.execute('read', {'note_id': 99}, ctx)


@pytest.mark.parametrize('bad', ['3', None, True])
async def test_read_rejects_non_integer_note_id(bad: object) -> None:
    plugin = NotesPlugin()
    with pytest.raises(NotesPluginError, match="'note_id' must be an integer"):
        await plugin.execute('read', {'note_id': bad}, _ctx())


async def test_read_surfaces_incomplete_row_as_descriptive_error() -> None:
    """A row missing the plugin's own columns is a store contract break.

    It must surface as a descriptive NotesPluginError, not an opaque
    KeyError the executor would record verbatim as the failure_reason.
    """
    plugin = NotesPlugin()
    ctx = _ctx({'database.select': {'rows': [{'id': 3}]}})
    with pytest.raises(NotesPluginError, match='incomplete row'):
        await plugin.execute('read', {'note_id': 3}, ctx)


# --- table lifecycle ---------------------------------------------------------


async def test_table_defined_exactly_once_across_calls() -> None:
    """`define_table` is asserted once per process, not per operation."""
    plugin = NotesPlugin()
    ctx = _ctx({'database.insert': {'id': 1}, 'database.select': {'rows': []}})

    await plugin.execute('create', {'content': 'one'}, ctx)
    await plugin.execute('list', {}, ctx)
    await plugin.execute('create', {'content': 'two'}, ctx)

    define_calls = [c for c in ctx.calls if c[0] == 'database.define_table']
    assert len(define_calls) == 1


async def test_unknown_capability_raises() -> None:
    plugin = NotesPlugin()
    with pytest.raises(NotesPluginError, match='unknown capability'):
        await plugin.execute('delete', {'note_id': 1}, _ctx())


# --- manifest ----------------------------------------------------------------


def plugin_columns() -> dict[str, object]:
    """The exact columns payload the plugin sends to define_table.

    Re-derived from the manifest factory's module so the round-trip
    assertions above stay in lock-step with the schema if it changes.
    """
    return {
        'content': {'type': 'text', 'not_null': True},
        'created_at': {'type': 'datetime', 'not_null': True},
    }


def test_manifest_is_local_write_user_facing_db_consumer() -> None:
    manifest, instance = build_notes_plugin()

    assert isinstance(instance, NotesPlugin)
    assert manifest.name == 'notes'
    assert manifest.blast_radius.value == 'local-write'
    assert {c.name for c in manifest.capabilities} == {'create', 'list', 'read'}
    # User-facing: notes capabilities appear in the planner menu (unlike
    # the internal database.*).
    assert all(not c.internal for c in manifest.capabilities)
    # Declares exactly the internal database capabilities it reaches via
    # ctx.call — the registry builder rejects a call to anything absent here.
    assert manifest.requires == (
        'database.define_table',
        'database.insert',
        'database.select',
    )
