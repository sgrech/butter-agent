"""Built-in `database` plugin — capability-mediated shared SQLite store.

Task #381 slice 3. Wraps the already-open `storage.sqlite.Database` so
write-plugins (notes, future memory/journal/…) don't each re-implement
connection lifecycle and butter-agent has a single backup target. Spec:
`specs/development/database-plugin.md`.

Hard namespace isolation (invariant #6) is enforced *outside* this module.
Core's `_PluginContext` prefixes the caller-supplied table name with the
calling plugin's identity (`{owner}__{table}`) and rejects a caller name
containing the `__` separator *before* dispatch. This plugin therefore
only ever receives fully-qualified table names and contains **zero**
identity logic — it cannot derive, override, or leak the namespace because
it never sees the un-prefixed name or the caller. That is the whole point
of putting the boundary in core: a third-party plugin swapped in for this
one still could not escape its namespace, because the prefixing already
happened.

All capabilities are `internal: true` — invisible to the planner, callable
only plugin-to-plugin via `PluginContext.call`. The plugin-level
`blast_radius` is `local-write` (the strictest tier any capability needs;
the manifest schema has no per-capability radius yet, so reads inherit it).

Deviation from the spec's draft API: every capability addresses its table
via a single `table` key (the spec's §5 draft used `name` for
`define_table`). Uniform keying keeps core's namespace hook trivial — it
rewrites one key for one blessed plugin — instead of core hardcoding a
per-capability argument map, which would couple core to this plugin's
schema. The spec doc is updated to match.
"""

from __future__ import annotations

import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Final

from butter_agent.core.registry import BlastRadius, Capability, PluginContext, PluginManifest
from butter_agent.storage.sqlite import Database

#: Canonical built-in name. Core (`task_executor`) imports this so the
#: namespace-prefixing hook and the registered plugin agree on one string.
PLUGIN_NAME: Final = 'database'

# Table and column identifiers are interpolated directly into SQL —
# SQLite cannot parameterise them. They are validated to a strict
# lowercase-identifier charset at the trust boundary so interpolation is
# safe. A fully-qualified table name (`owner__table`) matches this too —
# the `__` separator is just underscores — so the same rule covers both
# the core-prefixed table name and caller-supplied column names.
_IDENTIFIER_RE = re.compile(r'^[a-z][a-z0-9_]*$')

#: ColumnSpec `type` → SQLite column type. `datetime` has no SQLite
#: affinity; ISO-8601 text is the conventional storage form.
_TYPE_MAP: Final[dict[str, str]] = {
    'text': 'TEXT',
    'integer': 'INTEGER',
    'real': 'REAL',
    'blob': 'BLOB',
    'datetime': 'TEXT',
}


@dataclass(frozen=True, slots=True)
class _ColumnSpec:
    """A validated ColumnSpec (spec §4 defaults already applied)."""

    type: str
    not_null: bool
    primary_key: bool


class DatabasePluginError(Exception):
    """Raised on any malformed `database.*` call.

    Propagates out of `execute`; the task executor records it as the
    step's `failure_reason` (invariant #6 — plugin code may raise for any
    reason and must not tear the loop down).
    """


class DatabasePlugin:
    """`Plugin` Protocol implementation backed by one shared `Database`."""

    def __init__(self, database: Database) -> None:
        self._db = database

    async def execute(
        self,
        capability: str,
        inputs: dict[str, object],
        context: PluginContext,
    ) -> dict[str, object]:
        # This plugin never calls out — it is a leaf. The context exists
        # only to satisfy the Protocol; namespace identity was already
        # applied by core before dispatch (see module docstring).
        del context
        handler = _HANDLERS.get(capability)
        if handler is None:
            raise DatabasePluginError(
                f'unknown capability {capability!r} (expected one of: {", ".join(sorted(_HANDLERS))})',
            )
        return await handler(self, inputs)

    async def _define_table(self, inputs: dict[str, object]) -> dict[str, object]:
        table = _table(inputs)
        columns = _require_mapping(inputs, 'columns', allow_empty=False)

        column_sql: list[str] = []
        primary_keys = 0
        for raw_name, raw_spec in columns.items():
            name = _ident(raw_name, 'column')
            spec = _require_spec(raw_name, raw_spec)
            sqlite_type = _TYPE_MAP.get(spec.type)
            if sqlite_type is None:
                raise DatabasePluginError(
                    f'column {raw_name!r}: invalid type {spec.type!r} (expected one of: {", ".join(sorted(_TYPE_MAP))})',
                )
            parts = [name, sqlite_type]
            if spec.not_null:
                parts.append('NOT NULL')
            if spec.primary_key:
                parts.append('PRIMARY KEY')
                primary_keys += 1
            # `default` is advisory only per spec §4 — accepted in the
            # ColumnSpec but deliberately NOT emitted into DDL: an
            # arbitrary SQL literal cannot be safely interpolated and
            # SQLite has no placeholder for DEFAULT clauses.
            column_sql.append(' '.join(parts))

        if primary_keys > 1:
            raise DatabasePluginError('at most one column may declare primary_key')
        if primary_keys == 0:
            # Auto surrogate key so every table has a stable rowid alias
            # callers can reference (spec §5). A caller-declared column
            # literally named `id` would collide with this and SQLite
            # would reject the DDL with an opaque "duplicate column name"
            # — refuse early with an actionable message instead.
            if 'id' in columns:
                raise DatabasePluginError(
                    "column 'id' conflicts with the auto-generated primary key — declare it with primary_key: true to own it, or rename it",
                )
            column_sql.insert(0, 'id INTEGER PRIMARY KEY AUTOINCREMENT')

        await self._db.execute_ddl(
            f'CREATE TABLE IF NOT EXISTS {table} ({", ".join(column_sql)})',
        )
        return {'table': table}

    async def _insert(self, inputs: dict[str, object]) -> dict[str, object]:
        table = _table(inputs)
        row = _require_mapping(inputs, 'row', allow_empty=False)
        columns = [_ident(k, 'column') for k in row]
        placeholders = ', '.join('?' * len(columns))
        result = await self._db.execute_write(
            f'INSERT INTO {table} ({", ".join(columns)}) VALUES ({placeholders})',
            tuple(row.values()),
        )
        return {'id': result.last_row_id}

    async def _select(self, inputs: dict[str, object]) -> dict[str, object]:
        table = _table(inputs)
        sql = f'SELECT * FROM {table}'
        params: list[object] = []

        where = _optional_mapping(inputs, 'where')
        if where:
            sql += ' WHERE ' + ' AND '.join(f'{_ident(k, "column")} = ?' for k in where)
            params.extend(where.values())

        order_by = inputs.get('order_by')
        if order_by is not None:
            # v1: a single bare column, ascending. Validated as an
            # identifier so it cannot inject (no parameter form exists
            # for ORDER BY columns).
            sql += f' ORDER BY {_ident(order_by, "order_by")}'

        limit = inputs.get('limit')
        if limit is not None:
            if not isinstance(limit, int) or isinstance(limit, bool) or limit < 0:
                raise DatabasePluginError(f'limit must be a non-negative integer, got {limit!r}')
            sql += ' LIMIT ?'
            params.append(limit)

        return {'rows': await self._db.query(sql, tuple(params))}

    async def _update(self, inputs: dict[str, object]) -> dict[str, object]:
        table = _table(inputs)
        assignments = _require_mapping(inputs, 'set', allow_empty=False)
        # `where` is required and non-empty: an UPDATE with no predicate
        # would rewrite every row in the namespace, almost never intended
        # and unrecoverable. Callers that truly want that can pass an
        # explicit always-true predicate themselves.
        where = _require_mapping(inputs, 'where', allow_empty=False)
        set_sql = ', '.join(f'{_ident(k, "column")} = ?' for k in assignments)
        where_sql = ' AND '.join(f'{_ident(k, "column")} = ?' for k in where)
        result = await self._db.execute_write(
            f'UPDATE {table} SET {set_sql} WHERE {where_sql}',
            (*assignments.values(), *where.values()),
        )
        return {'updated': result.row_count}

    async def _delete(self, inputs: dict[str, object]) -> dict[str, object]:
        table = _table(inputs)
        where = _require_mapping(inputs, 'where', allow_empty=False)  # same guard as update
        where_sql = ' AND '.join(f'{_ident(k, "column")} = ?' for k in where)
        result = await self._db.execute_write(
            f'DELETE FROM {table} WHERE {where_sql}',
            tuple(where.values()),
        )
        return {'deleted': result.row_count}


# Capability name → handler. Defined after the class so the methods
# exist; keeps `execute` a flat dispatch with no if/elif ladder.
_Handler = Callable[['DatabasePlugin', dict[str, object]], Awaitable[dict[str, object]]]

_HANDLERS: Final[dict[str, _Handler]] = {
    'define_table': DatabasePlugin._define_table,
    'insert': DatabasePlugin._insert,
    'select': DatabasePlugin._select,
    'update': DatabasePlugin._update,
    'delete': DatabasePlugin._delete,
}


# --- Input validation --------------------------------------------------------


def _ident(value: object, role: str) -> str:
    """Validate `value` is a safe SQL identifier and return it.

    Identifiers cannot be parameterised, so this is the SQL-injection
    boundary for table/column/order-by names. The fully-qualified table
    name produced by core (`owner__table`) satisfies the same rule.
    """
    if not isinstance(value, str) or not _IDENTIFIER_RE.match(value):
        raise DatabasePluginError(
            f'invalid {role} name {value!r}: must match [a-z][a-z0-9_]* (lowercase letter, then lowercase letters/digits/underscores)',
        )
    return value


def _table(inputs: dict[str, object]) -> str:
    """Validate the (already core-namespaced) `table` input."""
    if 'table' not in inputs:
        raise DatabasePluginError("missing required input 'table'")
    return _ident(inputs['table'], 'table')


def _require_mapping(inputs: dict[str, object], key: str, *, allow_empty: bool) -> dict[str, object]:
    value = inputs.get(key)
    if not isinstance(value, dict):
        raise DatabasePluginError(f'input {key!r} must be a mapping, got {type(value).__name__}')
    if not allow_empty and not value:
        raise DatabasePluginError(f'input {key!r} must not be empty')
    return value


def _optional_mapping(inputs: dict[str, object], key: str) -> dict[str, object]:
    value = inputs.get(key)
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise DatabasePluginError(f'input {key!r} must be a mapping, got {type(value).__name__}')
    return value


def _require_spec(column: object, raw: object) -> _ColumnSpec:
    """Normalise one ColumnSpec, applying spec §4 defaults."""
    if not isinstance(raw, dict):
        raise DatabasePluginError(f'column {column!r}: spec must be a mapping, got {type(raw).__name__}')
    if 'type' not in raw:
        raise DatabasePluginError(f'column {column!r}: spec missing required {"type"!r}')
    col_type = raw['type']
    if not isinstance(col_type, str):
        raise DatabasePluginError(f'column {column!r}: type must be a string, got {type(col_type).__name__}')
    not_null = raw.get('not_null', False)
    primary_key = raw.get('primary_key', False)
    if not isinstance(not_null, bool):
        raise DatabasePluginError(f'column {column!r}: not_null must be a boolean')
    if not isinstance(primary_key, bool):
        raise DatabasePluginError(f'column {column!r}: primary_key must be a boolean')
    return _ColumnSpec(type=col_type, not_null=not_null, primary_key=primary_key)


# --- Manifest + factory ------------------------------------------------------


def _capability(name: str, input_schema: dict[str, object], output_schema: dict[str, object]) -> Capability:
    return Capability(
        name=name,
        description=f'database.{name} (internal — caller namespace applied by core)',
        input_schema=input_schema,
        output_schema=output_schema,
        internal=True,
    )


def build_database_plugin(database: Database) -> tuple[PluginManifest, DatabasePlugin]:
    """Construct the built-in `database` plugin bound to `database`.

    Returns a `(manifest, plugin)` pair ready for `RegistryBuilder.register`.
    The manifest is built in code (not parsed from a `manifest.toml`)
    because this is a built-in, not a fetched source; `entrypoint` is
    informational only — the loader never resolves it.
    """
    manifest = PluginManifest(
        name=PLUGIN_NAME,
        version='1.0.0',
        blast_radius=BlastRadius.LOCAL_WRITE,
        entrypoint='butter_agent.plugins.database:DatabasePlugin',
        capabilities=(
            _capability('define_table', {'table': 'string', 'columns': 'object'}, {'table': 'string'}),
            _capability('insert', {'table': 'string', 'row': 'object'}, {'id': 'integer'}),
            _capability('select', {'table': 'string'}, {'rows': 'array'}),
            _capability('update', {'table': 'string', 'where': 'object', 'set': 'object'}, {'updated': 'integer'}),
            _capability('delete', {'table': 'string', 'where': 'object'}, {'deleted': 'integer'}),
        ),
    )
    return manifest, DatabasePlugin(database)
