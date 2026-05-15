"""Built-in `notes` plugin — first `local-write` capability consumer.

Task #377. Persistent free-form note capture from the REPL. Notes is the
first plugin to *write* and the first real consumer of the shared
`database` plugin: it owns no SQLite file and never sees raw SQL. It calls
`ctx.call("database.define_table", ...)` once and
`ctx.call("database.insert" | "database.select", ...)` for writes/reads.
Spec: `specs/development/notes-plugin.md`.

Three things this plugin exists to prove end-to-end (spec §1):

- The gate-handler `confirm` path against a real model-emitted plan — the
  manifest declares `blast_radius = local-write`; the planner is expected
  to emit `gate: confirm` on `notes.create`, and the executor enforces it
  before this code ever runs (invariant #5 — gate enforcement is core's,
  never the plugin's).
- The agent-mediated variable-pool data channel: `clock.now → notes.create`
  feeds an ISO-8601 timestamp in as the optional `created_at` input
  (`$t.time`). There is no plugin-to-plugin path for that value — it flows
  only through the executor's variable pool (invariant #6).
- A worked example future write-plugins copy.

Namespace isolation (invariant #6) is core's, not this module's: every
`database.*` call passes the **bare** table name `"entries"`; core's
`_PluginContext` rewrites it to `notes__entries` before dispatch. This
plugin neither sends nor sees the `notes__` prefix, and cannot reach any
other plugin's tables.

Persistence contract discovered in database slice 3 (memory-mcp 1430,
spec §4 callout) that this module is built to:

- The `table` input is a single key carrying the bare name; never `name`,
  never prefixed.
- `ColumnSpec.default` is advisory only — it is **not** emitted into DDL.
  `created_at` therefore has no DB-level default: this plugin always
  supplies it on insert (from the variable pool when chained off
  `clock.now`, else self-generated). `datetime` columns are stored as
  ISO-8601 TEXT.
- `database.select`'s `where` is equality-only AND; reads here only ever
  filter by the surrogate `id`.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import Final

from butter_agent.core.registry import BlastRadius, Capability, PluginContext, PluginManifest

#: Canonical built-in name. `app.build_repl` registers this plugin under
#: it; core prefixes the bare table name with it (`{PLUGIN_NAME}__entries`)
#: before any `database.*` dispatch.
PLUGIN_NAME: Final = 'notes'

#: Bare table name passed to every `database.*` call. Core rewrites it to
#: `notes__entries`; this module never constructs or sees the prefix.
_TABLE: Final = 'entries'

#: Fully-qualified internal capabilities this plugin reaches via
#: `PluginContext.call`. Every entry is also declared in the manifest
#: `requires` (the registry builder rejects a call to anything absent
#: there); keeping one tuple drive both removes the chance of drift.
_REQUIRES: Final = (
    'database.define_table',
    'database.insert',
    'database.select',
)

#: Column schema for the notes table. `not_null` IS emitted into DDL by the
#: database plugin, so every insert must carry both columns — `created_at`
#: is self-populated precisely because no DB default exists for it.
_COLUMNS: Final[dict[str, object]] = {
    'content': {'type': 'text', 'not_null': True},
    'created_at': {'type': 'datetime', 'not_null': True},
}


class NotesPluginError(Exception):
    """Raised on any malformed `notes.*` call.

    Propagates out of `execute`; the task executor catches it on its broad
    plugin-failure path and records it as the step's `failure_reason`
    (invariant #6 — plugin code may raise for any reason and must not tear
    the loop down). Not raised for an empty `notes.list` (an empty list is
    a valid result, not a failure); is raised for `notes.read` of an
    unknown id (the caller asked for a specific note that does not exist).
    """


class NotesPlugin:
    """`Plugin` Protocol implementation backed by the shared `database` plugin.

    Holds no database handle: persistence is entirely via `context.call`
    into `database.*`. The notes table is created lazily on first use and
    only once per process — `define_table` is idempotent (`CREATE TABLE IF
    NOT EXISTS`), so the guard is an optimisation plus a single well-defined
    point where the schema is asserted, not a correctness requirement.
    """

    def __init__(self) -> None:
        self._table_ready = False
        # Plan steps execute sequentially in the executor, but a single
        # plan can legitimately invoke notes twice (e.g. create then list);
        # the lock keeps the check-then-define critical section atomic so
        # the schema is asserted exactly once regardless.
        self._table_lock = asyncio.Lock()

    async def execute(
        self,
        capability: str,
        inputs: dict[str, object],
        context: PluginContext,
    ) -> dict[str, object]:
        if capability == 'create':
            return await self._create(inputs, context)
        if capability == 'list':
            return await self._list(inputs, context)
        if capability == 'read':
            return await self._read(inputs, context)
        raise NotesPluginError(
            f'unknown capability {capability!r} (expected one of: create, list, read)',
        )

    async def _ensure_table(self, context: PluginContext) -> None:
        """Create the notes table on first use, exactly once per process."""
        if self._table_ready:
            return
        async with self._table_lock:
            if self._table_ready:
                return
            await context.call(
                'database.define_table',
                {'table': _TABLE, 'columns': _COLUMNS},
            )
            self._table_ready = True

    async def _create(self, inputs: dict[str, object], context: PluginContext) -> dict[str, object]:
        content = inputs.get('content')
        if not isinstance(content, str) or not content:
            raise NotesPluginError(f"input 'content' must be a non-empty string, got {content!r}")

        # `created_at` is optional: present when the plan chained
        # `clock.now → notes.create` (the `$t.time` variable-pool value),
        # absent for a bare "save a note" plan. Either way the column is
        # NOT NULL with no DB default, so a value is always written.
        created_at = _resolve_created_at(inputs.get('created_at'))

        await self._ensure_table(context)
        inserted = await context.call(
            'database.insert',
            {'table': _TABLE, 'row': {'content': content, 'created_at': created_at}},
        )
        note_id = inserted.get('id')
        if not isinstance(note_id, int):
            # database.insert returns the surrogate rowid; anything else is
            # a contract break in the store, surfaced rather than silently
            # returning a malformed note_id downstream.
            raise NotesPluginError(f'database.insert returned a non-integer id {note_id!r}')
        return {'note_id': note_id, 'created_at': created_at}

    async def _list(self, inputs: dict[str, object], context: PluginContext) -> dict[str, object]:
        await self._ensure_table(context)
        select: dict[str, object] = {'table': _TABLE, 'order_by': 'id'}
        limit = inputs.get('limit')
        if limit is not None:
            if not isinstance(limit, int) or isinstance(limit, bool) or limit < 0:
                raise NotesPluginError(f"input 'limit' must be a non-negative integer, got {limit!r}")
            select['limit'] = limit
        result = await context.call('database.select', select)
        # The store returns rows already shaped {id, content, created_at}
        # — exactly the documented `notes` element shape — so pass them
        # straight through rather than re-projecting field by field.
        return {'notes': result.get('rows', [])}

    async def _read(self, inputs: dict[str, object], context: PluginContext) -> dict[str, object]:
        note_id = inputs.get('note_id')
        if not isinstance(note_id, int) or isinstance(note_id, bool):
            raise NotesPluginError(f"input 'note_id' must be an integer, got {note_id!r}")

        await self._ensure_table(context)
        result = await context.call(
            'database.select',
            {'table': _TABLE, 'where': {'id': note_id}},
        )
        rows = result.get('rows')
        if not isinstance(rows, list) or not rows:
            raise NotesPluginError(f'no note with id {note_id}')
        row = rows[0]
        if not isinstance(row, dict):
            # database.select rows are dict-shaped per its contract;
            # anything else is a store contract break, surfaced rather
            # than indexed blindly (same stance as _create's id check).
            raise NotesPluginError(f'database.select returned a non-mapping row {row!r}')
        content = row.get('content')
        created_at = row.get('created_at')
        if not isinstance(content, str) or not isinstance(created_at, str):
            # The row exists but is missing the columns this plugin
            # defined. Surface it with the same descriptive contract-break
            # error as above rather than letting a bare KeyError /
            # malformed value escape (the executor would otherwise record
            # an opaque `KeyError: 'content'` as the failure_reason).
            raise NotesPluginError(f'database.select returned an incomplete row {row!r}')
        return {'content': content, 'created_at': created_at}


def _resolve_created_at(value: object) -> str:
    """Return the timestamp to store: a supplied ISO-8601 string, or now.

    A `$t.time` from a prior `clock.now` step arrives as a non-empty
    string and is used verbatim — the plugin trusts the chained value
    rather than re-deriving it (the whole point of the variable-pool
    channel). Absent or empty means an un-chained plan: generate a
    timezone-aware ISO-8601 UTC stamp (the column stores TEXT).
    """
    if isinstance(value, str) and value:
        return value
    return datetime.now(UTC).isoformat()


def _capability(
    name: str,
    description: str,
    input_schema: dict[str, object],
    output_schema: dict[str, object],
) -> Capability:
    # `internal` defaults to False: notes capabilities ARE planner-visible
    # (unlike database.*). The model selects them and declares the gate per
    # step; the manifest carries no gate field — gate is a plan concern,
    # enforced by core (invariant #5).
    return Capability(
        name=name,
        description=description,
        input_schema=input_schema,
        output_schema=output_schema,
    )


def build_notes_plugin() -> tuple[PluginManifest, NotesPlugin]:
    """Construct the built-in `notes` plugin.

    Returns a `(manifest, plugin)` pair ready for `RegistryBuilder.register`.
    Takes no arguments — unlike `database`, notes owns no resource; it
    persists entirely through `PluginContext.call`. The manifest is built
    in code (built-in, not a fetched source); `entrypoint` is informational
    only — the loader never resolves it.
    """
    manifest = PluginManifest(
        name=PLUGIN_NAME,
        version='1.0.0',
        blast_radius=BlastRadius.LOCAL_WRITE,
        entrypoint='butter_agent.plugins.notes:NotesPlugin',
        capabilities=(
            _capability(
                'create',
                'Save a free-form note to local storage. `content` is the note body. Optionally accepts `created_at`, an ISO-8601 timestamp (e.g. chained from `clock.now`); when omitted the current time is recorded automatically.',
                {'content': 'string'},
                {'note_id': 'integer', 'created_at': 'string'},
            ),
            _capability(
                'list',
                'List saved notes oldest-first. Optional `limit` caps how many are returned.',
                {},
                {'notes': 'array'},
            ),
            _capability(
                'read',
                'Read one saved note by its `note_id` (as returned by notes.create or notes.list).',
                {'note_id': 'integer'},
                {'content': 'string', 'created_at': 'string'},
            ),
        ),
        requires=_REQUIRES,
    )
    return manifest, NotesPlugin()
