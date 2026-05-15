"""Tests for the built-in `database` plugin (task #381 slice 3).

The plugin is unit-tested in isolation: it receives **already
fully-qualified** table names because core's `_PluginContext` does the
`{caller}__{table}` prefixing before dispatch (covered separately in
`test_task_executor.py`). Feeding the plugin FQ names here mirrors exactly
what it sees in production and documents that it holds no identity logic.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path
from typing import cast

import pytest

from butter_agent.plugins.database import (
    DatabasePlugin,
    DatabasePluginError,
    build_database_plugin,
)
from butter_agent.storage.sqlite import Database
from tests.support import FakePluginContext

# An arbitrary caller's namespace, as core would have produced it.
_T = 'notes__entries'


@pytest.fixture
async def plugin(tmp_path: Path) -> AsyncIterator[DatabasePlugin]:
    database = await Database.open(tmp_path / 'butter.db')
    try:
        yield DatabasePlugin(database)
    finally:
        await database.close()


def _ctx() -> FakePluginContext:
    # The plugin never touches the context (it is a leaf); any Protocol-
    # satisfying object works.
    return FakePluginContext()


async def _define_default(plugin: DatabasePlugin, table: str = _T) -> None:
    await plugin.execute(
        'define_table',
        {'table': table, 'columns': {'body': {'type': 'text', 'not_null': True}}},
        _ctx(),
    )


# --- Round trip --------------------------------------------------------------


async def test_full_crud_round_trip(plugin: DatabasePlugin) -> None:
    await _define_default(plugin)

    inserted = await plugin.execute('insert', {'table': _T, 'row': {'body': 'first'}}, _ctx())
    assert inserted == {'id': 1}
    second = await plugin.execute('insert', {'table': _T, 'row': {'body': 'second'}}, _ctx())
    assert second == {'id': 2}

    selected = await plugin.execute('select', {'table': _T}, _ctx())
    assert selected == {'rows': [{'id': 1, 'body': 'first'}, {'id': 2, 'body': 'second'}]}

    updated = await plugin.execute(
        'update',
        {'table': _T, 'where': {'id': 1}, 'set': {'body': 'edited'}},
        _ctx(),
    )
    assert updated == {'updated': 1}

    deleted = await plugin.execute('delete', {'table': _T, 'where': {'body': 'second'}}, _ctx())
    assert deleted == {'deleted': 1}

    remaining = await plugin.execute('select', {'table': _T}, _ctx())
    assert remaining == {'rows': [{'id': 1, 'body': 'edited'}]}


async def test_define_table_is_idempotent(plugin: DatabasePlugin) -> None:
    result = await plugin.execute(
        'define_table',
        {'table': _T, 'columns': {'body': {'type': 'text'}}},
        _ctx(),
    )
    assert result == {'table': _T}
    # Second define against an existing table must not raise (CREATE TABLE
    # IF NOT EXISTS) and must not drop data.
    await plugin.execute('insert', {'table': _T, 'row': {'body': 'keep'}}, _ctx())
    await plugin.execute('define_table', {'table': _T, 'columns': {'body': {'type': 'text'}}}, _ctx())
    assert await plugin.execute('select', {'table': _T}, _ctx()) == {'rows': [{'id': 1, 'body': 'keep'}]}


async def test_explicit_primary_key_suppresses_auto_id(plugin: DatabasePlugin) -> None:
    await plugin.execute(
        'define_table',
        {'table': _T, 'columns': {'slug': {'type': 'text', 'primary_key': True}, 'body': {'type': 'text'}}},
        _ctx(),
    )
    await plugin.execute('insert', {'table': _T, 'row': {'slug': 'a', 'body': 'x'}}, _ctx())
    rows = await plugin.execute('select', {'table': _T}, _ctx())
    assert rows == {'rows': [{'slug': 'a', 'body': 'x'}]}  # no auto `id` column


# --- select filtering --------------------------------------------------------


async def test_select_where_is_equality_filter(plugin: DatabasePlugin) -> None:
    await _define_default(plugin)
    for body in ('a', 'b', 'a'):
        await plugin.execute('insert', {'table': _T, 'row': {'body': body}}, _ctx())
    result = await plugin.execute('select', {'table': _T, 'where': {'body': 'a'}}, _ctx())
    assert result == {'rows': [{'id': 1, 'body': 'a'}, {'id': 3, 'body': 'a'}]}


async def test_select_order_by_and_limit(plugin: DatabasePlugin) -> None:
    await _define_default(plugin)
    for body in ('c', 'a', 'b'):
        await plugin.execute('insert', {'table': _T, 'row': {'body': body}}, _ctx())
    result = await plugin.execute('select', {'table': _T, 'order_by': 'body', 'limit': 2}, _ctx())
    assert result == {'rows': [{'id': 2, 'body': 'a'}, {'id': 3, 'body': 'b'}]}


# --- Validation --------------------------------------------------------------


async def test_unknown_capability_raises(plugin: DatabasePlugin) -> None:
    with pytest.raises(DatabasePluginError, match='unknown capability'):
        await plugin.execute('truncate', {'table': _T}, _ctx())


async def test_missing_table_input_raises(plugin: DatabasePlugin) -> None:
    with pytest.raises(DatabasePluginError, match="missing required input 'table'"):
        await plugin.execute('select', {}, _ctx())


async def test_invalid_table_identifier_raises(plugin: DatabasePlugin) -> None:
    # A name that slipped past core (e.g. uppercase) is still refused at
    # the SQL boundary — defence in depth against identifier injection.
    with pytest.raises(DatabasePluginError, match='invalid table name'):
        await plugin.execute('select', {'table': 'Robert);DROP TABLE x;--'}, _ctx())


async def test_invalid_column_identifier_raises(plugin: DatabasePlugin) -> None:
    await _define_default(plugin)
    with pytest.raises(DatabasePluginError, match='invalid column name'):
        await plugin.execute('insert', {'table': _T, 'row': {'bad-col': 1}}, _ctx())


async def test_define_table_rejects_multiple_primary_keys(plugin: DatabasePlugin) -> None:
    with pytest.raises(DatabasePluginError, match='at most one column may declare primary_key'):
        await plugin.execute(
            'define_table',
            {
                'table': _T,
                'columns': {
                    'a': {'type': 'text', 'primary_key': True},
                    'b': {'type': 'text', 'primary_key': True},
                },
            },
            _ctx(),
        )


async def test_define_table_rejects_id_column_colliding_with_auto_key(plugin: DatabasePlugin) -> None:
    """A caller `id` column without primary_key would duplicate the
    auto-injected surrogate — refused with an actionable message rather
    than an opaque SQLite 'duplicate column name' failure."""
    with pytest.raises(DatabasePluginError, match="column 'id' conflicts with the auto-generated primary key"):
        await plugin.execute(
            'define_table',
            {'table': _T, 'columns': {'id': {'type': 'integer'}, 'body': {'type': 'text'}}},
            _ctx(),
        )
    # But a caller that *owns* `id` via primary_key is fine (no auto key).
    await plugin.execute(
        'define_table',
        {'table': _T, 'columns': {'id': {'type': 'text', 'primary_key': True}}},
        _ctx(),
    )


async def test_define_table_rejects_invalid_type(plugin: DatabasePlugin) -> None:
    with pytest.raises(DatabasePluginError, match='invalid type'):
        await plugin.execute(
            'define_table',
            {'table': _T, 'columns': {'c': {'type': 'json'}}},
            _ctx(),
        )


async def test_define_table_requires_columns(plugin: DatabasePlugin) -> None:
    with pytest.raises(DatabasePluginError, match="input 'columns' must not be empty"):
        await plugin.execute('define_table', {'table': _T, 'columns': {}}, _ctx())


async def test_insert_requires_non_empty_row(plugin: DatabasePlugin) -> None:
    await _define_default(plugin)
    with pytest.raises(DatabasePluginError, match="input 'row' must not be empty"):
        await plugin.execute('insert', {'table': _T, 'row': {}}, _ctx())


@pytest.mark.parametrize('capability', ['update', 'delete'])
async def test_update_and_delete_require_non_empty_where(plugin: DatabasePlugin, capability: str) -> None:
    await _define_default(plugin)
    payload: dict[str, object] = {'table': _T, 'where': {}}
    if capability == 'update':
        payload['set'] = {'body': 'x'}
    with pytest.raises(DatabasePluginError, match="input 'where' must not be empty"):
        await plugin.execute(capability, payload, _ctx())


# --- Full-text search --------------------------------------------------------


def _rows(result: dict[str, object]) -> list[dict[str, object]]:
    """Narrow a `search`/`select` result's `rows` for iteration in tests.

    `execute` returns `dict[str, object]`, so `result['rows']` is `object`
    to mypy --strict; the row shape is the documented `{rows: list[dict]}`
    contract these tests exercise."""
    return cast('list[dict[str, object]]', result['rows'])


async def _fts_ready(plugin: DatabasePlugin, *bodies: str) -> None:
    """define_table → insert `bodies` → define_fts on `body`."""
    await _define_default(plugin)
    for body in bodies:
        await plugin.execute('insert', {'table': _T, 'row': {'body': body}}, _ctx())
    result = await plugin.execute('define_fts', {'table': _T, 'columns': ['body']}, _ctx())
    assert result == {'table': _T}


async def test_search_finds_matching_rows(plugin: DatabasePlugin) -> None:
    await _fts_ready(plugin, 'buy butter', 'call the dentist', 'buttered toast')
    result = await plugin.execute('search', {'table': _T, 'query': 'butter'}, _ctx())
    # Porter stemming: "butter" matches "buttered". Whole base rows, same
    # shape as select. Order is bm25 rank (unasserted here — see id test).
    assert {r['body'] for r in _rows(result)} == {'buy butter', 'buttered toast'}


async def test_search_rebuild_backfills_preexisting_rows(plugin: DatabasePlugin) -> None:
    """Rows inserted *before* define_fts are searchable (the migration step)."""
    await _fts_ready(plugin, 'old note about butter')
    result = await plugin.execute('search', {'table': _T, 'query': 'butter'}, _ctx())
    assert [r['body'] for r in _rows(result)] == ['old note about butter']


async def test_search_triggers_keep_index_live(plugin: DatabasePlugin) -> None:
    await _fts_ready(plugin, 'first butter')
    # INSERT after define_fts → indexed by the AFTER INSERT trigger.
    await plugin.execute('insert', {'table': _T, 'row': {'body': 'second butter'}}, _ctx())
    after_insert = await plugin.execute('search', {'table': _T, 'query': 'butter', 'order': 'id'}, _ctx())
    assert [r['body'] for r in _rows(after_insert)] == ['first butter', 'second butter']

    # UPDATE → AFTER UPDATE trigger re-indexes the new content.
    await plugin.execute('update', {'table': _T, 'where': {'id': 1}, 'set': {'body': 'first margarine'}}, _ctx())
    after_update = await plugin.execute('search', {'table': _T, 'query': 'butter'}, _ctx())
    assert [r['body'] for r in _rows(after_update)] == ['second butter']

    # DELETE → AFTER DELETE trigger tombstones the FTS row.
    await plugin.execute('delete', {'table': _T, 'where': {'id': 2}}, _ctx())
    after_delete = await plugin.execute('search', {'table': _T, 'query': 'butter'}, _ctx())
    assert after_delete == {'rows': []}


async def test_search_order_id_is_oldest_first(plugin: DatabasePlugin) -> None:
    await _fts_ready(plugin, 'butter one', 'butter two', 'butter three')
    result = await plugin.execute('search', {'table': _T, 'query': 'butter', 'order': 'id'}, _ctx())
    assert [r['id'] for r in _rows(result)] == [1, 2, 3]


async def test_search_limit_caps_results(plugin: DatabasePlugin) -> None:
    await _fts_ready(plugin, 'butter a', 'butter b', 'butter c')
    result = await plugin.execute('search', {'table': _T, 'query': 'butter', 'order': 'id', 'limit': 2}, _ctx())
    assert [r['body'] for r in _rows(result)] == ['butter a', 'butter b']


async def test_search_treats_fts5_metacharacters_as_literal_terms(plugin: DatabasePlugin) -> None:
    """A query full of FTS5 syntax must not error or inject — just no match."""
    await _fts_ready(plugin, 'plain note')
    result = await plugin.execute('search', {'table': _T, 'query': 'alpha) OR "beta" NEAR(x'}, _ctx())
    assert result == {'rows': []}  # parsed as literal terms, none present


async def test_search_empty_query_raises(plugin: DatabasePlugin) -> None:
    await _fts_ready(plugin, 'something')
    with pytest.raises(DatabasePluginError, match='must contain at least one term'):
        await plugin.execute('search', {'table': _T, 'query': '   '}, _ctx())


async def test_search_rejects_unknown_order(plugin: DatabasePlugin) -> None:
    await _fts_ready(plugin, 'x')
    with pytest.raises(DatabasePluginError, match="'order' must be 'rank' or 'id'"):
        await plugin.execute('search', {'table': _T, 'query': 'x', 'order': 'created_at'}, _ctx())


async def test_define_fts_is_idempotent(plugin: DatabasePlugin) -> None:
    await _fts_ready(plugin, 'butter note')
    # Second define against an existing index must not raise or duplicate
    # triggers (which would double-index every subsequent write).
    await plugin.execute('define_fts', {'table': _T, 'columns': ['body']}, _ctx())
    await plugin.execute('insert', {'table': _T, 'row': {'body': 'more butter'}}, _ctx())
    result = await plugin.execute('search', {'table': _T, 'query': 'butter', 'order': 'id'}, _ctx())
    assert [r['body'] for r in _rows(result)] == ['butter note', 'more butter']


async def test_define_fts_on_missing_table_raises(plugin: DatabasePlugin) -> None:
    with pytest.raises(DatabasePluginError, match='does not exist'):
        await plugin.execute('define_fts', {'table': _T, 'columns': ['body']}, _ctx())


async def test_define_fts_rejects_non_text_column(plugin: DatabasePlugin) -> None:
    await plugin.execute(
        'define_table',
        {'table': _T, 'columns': {'body': {'type': 'text'}, 'score': {'type': 'integer'}}},
        _ctx(),
    )
    with pytest.raises(DatabasePluginError, match='only text columns are full-text indexable'):
        await plugin.execute('define_fts', {'table': _T, 'columns': ['score']}, _ctx())


async def test_define_fts_rejects_unknown_column(plugin: DatabasePlugin) -> None:
    await _define_default(plugin)
    with pytest.raises(DatabasePluginError, match='cannot index unknown column'):
        await plugin.execute('define_fts', {'table': _T, 'columns': ['nope']}, _ctx())


async def test_define_fts_requires_non_empty_columns(plugin: DatabasePlugin) -> None:
    await _define_default(plugin)
    with pytest.raises(DatabasePluginError, match="input 'columns' must be a non-empty list"):
        await plugin.execute('define_fts', {'table': _T, 'columns': []}, _ctx())


# --- Manifest ----------------------------------------------------------------


async def test_built_in_manifest_is_all_internal_local_write(tmp_path: Path) -> None:
    database = await Database.open(tmp_path / 'butter.db')
    try:
        manifest, instance = build_database_plugin(database)
        assert isinstance(instance, DatabasePlugin)
        assert manifest.name == 'database'
        assert manifest.blast_radius.value == 'local-write'
        assert {c.name for c in manifest.capabilities} == {
            'define_table',
            'insert',
            'select',
            'update',
            'delete',
            'define_fts',
            'search',
        }
        # Every capability is internal — invisible to the planner, callable
        # only plugin-to-plugin (spec §5).
        assert all(c.internal for c in manifest.capabilities)
    finally:
        await database.close()
