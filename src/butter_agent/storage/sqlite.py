"""SQLite implementation of the storage seams.

`Database` owns one SQLite file. It expands `~`, creates the parent
directory if missing, opens a thread-safe connection
(`check_same_thread=False`) and serialises writers with an
`asyncio.Lock`. All blocking calls are dispatched through
`asyncio.to_thread` so the async caller is never blocked.

What this module does NOT do:

- Define the storage seams' Protocols — those live alongside their
  consumers (`ConversationHistory` in `core/context_manager.py`).
- Migrate schemas across versions. A real migration runner can land
  when the first persistent consumer arrives.

Conversation history is intentionally not persisted today: every
butter invocation starts a fresh chat session. When a session concept
lands, the persisted history class lives next to it (recoverable from
git history).
"""

from __future__ import annotations

import asyncio
import sqlite3
from pathlib import Path

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
        exists. Serialised behind the lock so two consumers
        bootstrapping in parallel don't race.
        """
        async with self._lock:
            await asyncio.to_thread(self._execute_sync, sql, ())

    async def execute(self, sql: str, params: tuple[object, ...]) -> None:
        """Run a write statement (INSERT / UPDATE / DELETE)."""
        async with self._lock:
            await asyncio.to_thread(self._execute_sync, sql, params)

    async def fetchall(self, sql: str, params: tuple[object, ...]) -> list[tuple[object, ...]]:
        """Run a read statement and return all rows.

        Reads take the same lock as writes. With `check_same_thread=False`
        the sqlite3 docs require the caller to serialise all access to
        the connection — otherwise an overlapping reader/writer pair
        can hit `ProgrammingError` or `OperationalError` intermittently.
        The lock is local to one async task at a time anyway, so the
        cost is negligible at our single-user scale.
        """
        async with self._lock:
            return await asyncio.to_thread(self._fetchall_sync, sql, params)

    async def close(self) -> None:
        """Close the underlying connection.

        Acquires the lock so any in-flight `execute` / `fetchall`
        completes before the connection is torn down — otherwise the
        in-flight thread could surface `sqlite3.ProgrammingError:
        Cannot operate on a closed database`. Idempotent: closing twice
        is harmless because `Connection.close()` itself is.
        """
        async with self._lock:
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
