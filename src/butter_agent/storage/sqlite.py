"""SQLite implementation of the storage seams.

Two classes live here:

- `Database` owns one SQLite file. It expands `~`, creates the parent
  directory if missing, opens a thread-safe connection
  (`check_same_thread=False`) and serialises writers with an
  `asyncio.Lock`. All blocking calls are dispatched through
  `asyncio.to_thread` so the async caller is never blocked.
- `SqliteConversationHistory` implements the `ConversationHistory`
  Protocol from `core/context_manager.py` against a single
  `conversation_history` table. Behaviour matches the in-memory
  default: `recent(N)` returns the N most recent entries oldest-first,
  bounded by what is in the table.

What this module does NOT do:

- Define the `ConversationHistory` Protocol — that lives in
  `core/context_manager.py`. This file only implements it.
- Manage plugin state. A future `SqlitePluginState` (or similar) will
  consume the same `Database` but own its own table; consumers do not
  read each other's tables.
- Migrate schemas across versions. The single migration each consumer
  performs today is a `CREATE TABLE IF NOT EXISTS` at startup; a real
  migration runner can land alongside the second consumer.
"""

from __future__ import annotations

import asyncio
import sqlite3
from pathlib import Path
from typing import cast

from butter_agent.core.context_manager import ConversationEntry

# --- Database ---------------------------------------------------------------


class Database:
    """Shared async wrapper around a single SQLite database file."""

    def __init__(self, connection: sqlite3.Connection, lock: asyncio.Lock) -> None:
        # Private constructor — use `Database.open()` so connection lifecycle
        # (path expansion, mkdir, pragmas) is handled in one place.
        self._connection = connection
        self._lock = lock

    @classmethod
    async def open(cls, path: str | Path) -> Database:
        """Open (or create) the SQLite database at `path`.

        `~` is expanded and the parent directory is created if missing
        so the shipped default (`~/.butter-agent/butter.db`) works on a
        fresh install. `check_same_thread=False` lets `asyncio.to_thread`
        dispatch to any worker; concurrent writers are serialised by
        the `asyncio.Lock` owned by this instance.
        """
        resolved = _resolve(path)
        await asyncio.to_thread(resolved.parent.mkdir, parents=True, exist_ok=True)
        connection = await asyncio.to_thread(_connect, resolved)
        return cls(connection=connection, lock=asyncio.Lock())

    async def execute_ddl(self, sql: str) -> None:
        """Run a DDL statement (CREATE TABLE / INDEX). Idempotent by convention.

        Consumers call this once at bootstrap to ensure their table
        exists. DDL is serialised behind the writer lock so two
        consumers bootstrapping in parallel don't race.
        """
        async with self._lock:
            await asyncio.to_thread(self._execute_sync, sql, ())

    async def execute(self, sql: str, params: tuple[object, ...]) -> None:
        """Run a write statement (INSERT / UPDATE / DELETE)."""
        async with self._lock:
            await asyncio.to_thread(self._execute_sync, sql, params)

    async def fetchall(self, sql: str, params: tuple[object, ...]) -> list[tuple[object, ...]]:
        """Run a read statement and return all rows.

        Reads do not take the writer lock — SQLite handles reader/writer
        concurrency itself, and not blocking reads on long writes is
        cheap insurance against a wedged context manager.
        """
        return await asyncio.to_thread(self._fetchall_sync, sql, params)

    async def close(self) -> None:
        await asyncio.to_thread(self._connection.close)

    def _execute_sync(self, sql: str, params: tuple[object, ...]) -> None:
        with self._connection:
            self._connection.execute(sql, params)

    def _fetchall_sync(self, sql: str, params: tuple[object, ...]) -> list[tuple[object, ...]]:
        cursor = self._connection.execute(sql, params)
        try:
            return cursor.fetchall()
        finally:
            cursor.close()


def _resolve(path: str | Path) -> Path:
    return Path(path).expanduser()


def _connect(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
    # WAL gives readers a snapshot while a writer is mid-transaction; foreign_keys
    # is opt-in per connection in SQLite, and we want it on for future tables.
    connection.execute('PRAGMA journal_mode = WAL')
    connection.execute('PRAGMA foreign_keys = ON')
    connection.execute('PRAGMA synchronous = NORMAL')
    return connection


# --- ConversationHistory implementation -------------------------------------


class SqliteConversationHistory:
    """`ConversationHistory` Protocol implementation backed by SQLite.

    Schema is bootstrapped once via `create()`. Subsequent instances
    against the same `Database` are cheap; they share the connection
    and the same table.
    """

    _TABLE = 'conversation_history'
    _DDL = f"""
        CREATE TABLE IF NOT EXISTS {_TABLE} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            turn_id TEXT NOT NULL,
            user_input TEXT NOT NULL,
            assistant_reply TEXT,
            timestamp REAL NOT NULL
        )
    """
    _INSERT = f'INSERT INTO {_TABLE} (turn_id, user_input, assistant_reply, timestamp) VALUES (?, ?, ?, ?)'
    _SELECT_RECENT = f'SELECT turn_id, user_input, assistant_reply, timestamp FROM {_TABLE} ORDER BY id DESC LIMIT ?'

    def __init__(self, database: Database) -> None:
        self._db = database

    @classmethod
    async def create(cls, database: Database) -> SqliteConversationHistory:
        """Bootstrap the schema and return a ready-to-use instance."""
        await database.execute_ddl(cls._DDL)
        return cls(database)

    async def append(self, entry: ConversationEntry) -> None:
        await self._db.execute(
            self._INSERT,
            (entry.turn_id, entry.user_input, entry.assistant_reply, entry.timestamp),
        )

    async def recent(self, limit: int) -> tuple[ConversationEntry, ...]:
        if limit <= 0:
            # Mirror `InMemoryConversationHistory` — non-positive limits
            # are documented as "nothing", not an error.
            return ()
        rows = await self._db.fetchall(self._SELECT_RECENT, (limit,))
        # SELECT ... ORDER BY id DESC gives newest-first; flip to
        # oldest-first within the window so the order matches the
        # in-memory default. SQLite's type affinity + our schema
        # guarantee the column types; cast tells mypy that.
        typed_rows = cast('list[tuple[str, str, str | None, float]]', rows)
        return tuple(ConversationEntry(turn_id=turn_id, user_input=user_input, assistant_reply=reply, timestamp=ts) for turn_id, user_input, reply, ts in reversed(typed_rows))
