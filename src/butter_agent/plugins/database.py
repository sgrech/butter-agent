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
import sqlite3
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


#: Fixed FTS5 tokenizer (spec database-fts.md §4). `porter` stems
#: ("dentists"→"dentist"), `unicode61` is the default word tokenizer,
#: `remove_diacritics 2` folds accents. Not caller-configurable in v1.
_FTS_TOKENIZE: Final = 'porter unicode61 remove_diacritics 2'

#: Declared types (from `define_table`'s ColumnSpec) that land in a TEXT
#: SQLite column and are therefore valid FTS index columns. `_TYPE_MAP`
#: maps both to 'TEXT'; FTS only makes sense over text.
_FTS_INDEXABLE_TYPES: Final = frozenset({'TEXT'})


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

    async def _define_fts(self, inputs: dict[str, object]) -> dict[str, object]:
        """Create (idempotently) an external-content FTS5 index + sync triggers.

        The base table keeps its typed schema, surrogate `id`, and
        NOT NULL columns untouched: the index is a separate
        `{table}_fts` virtual table referencing base rows by rowid
        (`content_rowid='id'`). AFTER INSERT/UPDATE/DELETE triggers keep
        it live so every later `insert`/`update`/`delete` — through this
        plugin or not — stays consistent. A final `'rebuild'` backfills
        rows that predate the index (the migration step for an existing
        table). Spec: database-fts.md §4-5, §8.
        """
        table = _table(inputs)
        columns = _require_fts_columns(inputs)
        await self._assert_indexable(table, columns)

        fts = f'{table}_fts'
        # `CREATE VIRTUAL TABLE IF NOT EXISTS` (and the `IF NOT EXISTS`
        # triggers) would silently no-op if an index already exists with
        # a *different* column set, leaving the requested columns
        # unindexed while the call reports success — a silent
        # correctness bug. Detect a shape mismatch and refuse: changing
        # the indexed columns means dropping and rebuilding the index,
        # which is out of scope for v1 (spec §5).
        existing = await self._existing_fts_columns(fts)
        if existing is not None and existing != columns:
            raise DatabasePluginError(
                f'full-text index for {table!r} already exists over columns {existing!r}; cannot redefine it over {columns!r} (drop-and-rebuild is unsupported in v1)',
            )
        col_sql = ', '.join(columns)
        col_new = ', '.join(f'new.{c}' for c in columns)
        col_old = ', '.join(f'old.{c}' for c in columns)

        try:
            await self._db.execute_ddl(
                f"CREATE VIRTUAL TABLE IF NOT EXISTS {fts} USING fts5({col_sql}, content='{table}', content_rowid='id', tokenize='{_FTS_TOKENIZE}')",
            )
        except sqlite3.OperationalError as exc:
            # The build's SQLite lacks the fts5 module. Fail loudly with
            # an actionable message — never silently fall back to a slow
            # table scan (spec §6: no silent fallback).
            if 'fts5' in str(exc) or 'no such module' in str(exc):
                raise DatabasePluginError(
                    'full-text search unavailable: this SQLite build has no FTS5 module',
                ) from exc
            raise

        # External-content sync triggers (SQLite FTS5 docs §4.4.3). The
        # `'delete'` command row removes the stale index entry before a
        # re-insert on UPDATE.
        await self._db.execute_ddl(
            f'CREATE TRIGGER IF NOT EXISTS {table}_ai AFTER INSERT ON {table} BEGIN INSERT INTO {fts}(rowid, {col_sql}) VALUES (new.id, {col_new}); END',
        )
        await self._db.execute_ddl(
            f"CREATE TRIGGER IF NOT EXISTS {table}_ad AFTER DELETE ON {table} BEGIN INSERT INTO {fts}({fts}, rowid, {col_sql}) VALUES('delete', old.id, {col_old}); END",
        )
        await self._db.execute_ddl(
            f"CREATE TRIGGER IF NOT EXISTS {table}_au AFTER UPDATE ON {table} BEGIN INSERT INTO {fts}({fts}, rowid, {col_sql}) VALUES('delete', old.id, {col_old}); INSERT INTO {fts}(rowid, {col_sql}) VALUES (new.id, {col_new}); END",
        )
        # Backfill rows written before the index existed. Cheap at our
        # single-user scale and safe to repeat, so it runs every call
        # rather than needing first-creation detection (spec §8).
        await self._db.execute(f"INSERT INTO {fts}({fts}) VALUES('rebuild')", ())
        return {'table': table}

    async def _search(self, inputs: dict[str, object]) -> dict[str, object]:
        """Full-text query over a `define_fts`'d table.

        `query` is natural text — never FTS5 syntax. Terms are quoted
        (neutralising every FTS5 metacharacter), prefix-globbed, and
        AND-joined here, then bound as a parameter, so a caller can pass
        a raw user phrase without escaping and it can neither error nor
        inject (spec §5).
        """
        table = _table(inputs)
        fts = f'{table}_fts'
        match = _fts_match_expression(inputs.get('query'))
        order = _fts_order(inputs.get('order'), fts)

        # The FTS table is referenced by name (not aliased): bm25() and
        # the MATCH operator both require the real FTS5 table name.
        sql = f'SELECT b.* FROM {table} b JOIN {fts} ON b.id = {fts}.rowid WHERE {fts} MATCH ? ORDER BY {order}'
        params: list[object] = [match]

        limit = inputs.get('limit')
        if limit is not None:
            if not isinstance(limit, int) or isinstance(limit, bool) or limit < 0:
                raise DatabasePluginError(f'limit must be a non-negative integer, got {limit!r}')
            sql += ' LIMIT ?'
            params.append(limit)

        try:
            rows = await self._db.query(sql, tuple(params))
        except sqlite3.OperationalError as exc:
            # No FTS index for this table — the caller skipped
            # define_fts. Surface the consistent, actionable
            # DatabasePluginError the rest of this plugin raises, not a
            # raw "no such table: …_fts" sqlite3 error.
            if 'no such table' in str(exc) and fts in str(exc):
                raise DatabasePluginError(
                    f'no full-text index for {table!r} — call define_fts first',
                ) from exc
            raise
        return {'rows': rows}

    async def _assert_indexable(self, table: str, columns: list[str]) -> None:
        """Reject FTS over a missing table, missing column, or non-text column.

        Surfaces a descriptive error here rather than letting a later
        `CREATE VIRTUAL TABLE` / trigger fail opaquely. `table` is already
        the `_ident`-validated FQ name so the PRAGMA interpolation is safe.
        """
        info = await self._db.query(f'PRAGMA table_info({table})', ())
        by_name = {row['name']: row for row in info}
        if not by_name:
            raise DatabasePluginError(f'base table {table!r} does not exist — call define_table first')
        if 'id' not in by_name:
            # content_rowid='id' binds the index to the surrogate key;
            # a caller-owned non-id primary key has no FTS support in v1.
            raise DatabasePluginError(f"table {table!r} has no surrogate 'id' column — FTS requires the auto key")
        for column in columns:
            row = by_name.get(column)
            if row is None:
                raise DatabasePluginError(f'cannot index unknown column {column!r} on {table!r}')
            col_type = str(row['type']).upper()
            if col_type not in _FTS_INDEXABLE_TYPES:
                raise DatabasePluginError(
                    f'column {column!r} is {col_type or "untyped"}, not text — only text columns are full-text indexable',
                )

    async def _existing_fts_columns(self, fts: str) -> list[str] | None:
        """Return an existing FTS index's columns in order, or None.

        `PRAGMA table_info` on an FTS5 virtual table lists its indexed
        columns; an empty result means the index does not exist yet.
        Used to detect a redefinition with a changed column set.
        """
        info = await self._db.query(f'PRAGMA table_info({fts})', ())
        if not info:
            return None
        return [str(row['name']) for row in info]


# Capability name → handler. Defined after the class so the methods
# exist; keeps `execute` a flat dispatch with no if/elif ladder.
_Handler = Callable[['DatabasePlugin', dict[str, object]], Awaitable[dict[str, object]]]

_HANDLERS: Final[dict[str, _Handler]] = {
    'define_table': DatabasePlugin._define_table,
    'insert': DatabasePlugin._insert,
    'select': DatabasePlugin._select,
    'update': DatabasePlugin._update,
    'delete': DatabasePlugin._delete,
    'define_fts': DatabasePlugin._define_fts,
    'search': DatabasePlugin._search,
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


def _require_fts_columns(inputs: dict[str, object]) -> list[str]:
    """Validate `columns` is a non-empty list of identifier strings.

    Existence + text-ness against the live base schema is checked
    separately by `_assert_indexable`; this is the shape/charset gate
    (the SQL-injection boundary for the FTS column list, same `_ident`
    rule the rest of the plugin uses).
    """
    value = inputs.get('columns')
    if not isinstance(value, list) or not value:
        raise DatabasePluginError("input 'columns' must be a non-empty list of column names")
    return [_ident(c, 'column') for c in value]


def _fts_match_expression(query: object) -> str:
    """Build a safe FTS5 MATCH expression from natural text (spec §5).

    Each whitespace term is wrapped as a quoted FTS5 string literal
    (internal `"` doubled) with a `*` prefix-glob appended outside the
    quotes — quoting neutralises every FTS5 metacharacter so a term can
    never become an operator or a syntax error. Terms are space-joined
    (implicit AND). The result is always bound as a parameter by the
    caller, never interpolated.
    """
    if not isinstance(query, str):
        raise DatabasePluginError(f"input 'query' must be a string, got {type(query).__name__}")
    terms = query.split()
    if not terms:
        raise DatabasePluginError("input 'query' must contain at least one term")
    return ' '.join(f'"{t.replace(chr(34), chr(34) * 2)}"*' for t in terms)


def _fts_order(order: object, fts: str) -> str:
    """Resolve the `order` input to a safe ORDER BY clause.

    `rank` (default) → `bm25(<fts table>)`, FTS5 relevance, best first.
    The bare `rank` shorthand is deliberately NOT used: when the base
    table happens to have a column named `rank`, `ORDER BY rank` is
    `ambiguous column name: rank` (the JOIN brings both into scope), and
    `bm25()` also requires the real FTS table name — an alias raises
    `no such column`. `id` is oldest-first, matching `select`/`list`.
    Any other value is rejected rather than interpolated. `fts` is the
    `_ident`-validated `{table}_fts` name, safe to interpolate.
    """
    if order is None or order == 'rank':
        return f'bm25({fts})'
    if order == 'id':
        return 'b.id'
    raise DatabasePluginError(f"input 'order' must be 'rank' or 'id', got {order!r}")


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
        version='1.1.0',
        blast_radius=BlastRadius.LOCAL_WRITE,
        entrypoint='butter_agent.plugins.database:DatabasePlugin',
        capabilities=(
            _capability('define_table', {'table': 'string', 'columns': 'object'}, {'table': 'string'}),
            _capability('insert', {'table': 'string', 'row': 'object'}, {'id': 'integer'}),
            _capability('select', {'table': 'string'}, {'rows': 'array'}),
            _capability('update', {'table': 'string', 'where': 'object', 'set': 'object'}, {'updated': 'integer'}),
            _capability('delete', {'table': 'string', 'where': 'object'}, {'deleted': 'integer'}),
            _capability('define_fts', {'table': 'string', 'columns': 'array'}, {'table': 'string'}),
            _capability('search', {'table': 'string', 'query': 'string'}, {'rows': 'array'}),
        ),
    )
    return manifest, DatabasePlugin(database)
