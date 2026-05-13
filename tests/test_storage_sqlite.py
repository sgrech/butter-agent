"""Tests for the SQLite storage layer.

The storage module owns persistence behind seams declared in
`core/context_manager.py`. Two responsibilities under test:

- `Database` lifecycle: path expansion, parent-dir creation, async
  wrapping, idempotent DDL, and resilience to repeated open against
  the same file.
- `SqliteConversationHistory`: round-trip append → recent, ordering
  (oldest-first within the window, matching the in-memory default),
  limit semantics, null `assistant_reply` handling, and persistence
  across reopen.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from butter_agent.core.context_manager import ConversationEntry
from butter_agent.storage.sqlite import (
    Database,
    SqliteConversationHistory,
)


def _entry(turn_id: str, *, user: str = 'hello', reply: str | None = 'hi', ts: float = 1.0) -> ConversationEntry:
    return ConversationEntry(turn_id=turn_id, user_input=user, assistant_reply=reply, timestamp=ts)


@pytest.fixture
async def db(tmp_path: Path) -> AsyncIterator[Database]:
    database = await Database.open(tmp_path / 'butter.db')
    try:
        yield database
    finally:
        # Close on teardown so SQLite releases the file handle before
        # tmp_path cleanup runs (matters on Windows + WAL aux files).
        await database.close()


# --- Database lifecycle -----------------------------------------------------


async def test_database_creates_parent_directory(tmp_path: Path) -> None:
    nested = tmp_path / 'deep' / 'nested' / 'butter.db'
    assert not nested.parent.exists()
    database = await Database.open(nested)
    try:
        assert nested.parent.exists()
        assert nested.exists()
    finally:
        await database.close()


async def test_database_expands_home_directory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Point `~` at tmp_path so we can assert expansion without touching real $HOME.
    monkeypatch.setenv('HOME', str(tmp_path))
    db = await Database.open('~/butter.db')
    expected = tmp_path / 'butter.db'
    assert expected.exists()
    await db.close()


async def test_database_idempotent_ddl(db: Database) -> None:
    # Running the same CREATE TABLE IF NOT EXISTS twice must succeed.
    ddl = 'CREATE TABLE IF NOT EXISTS x (id INTEGER PRIMARY KEY)'
    await db.execute_ddl(ddl)
    await db.execute_ddl(ddl)
    # And the table is actually there.
    await db.execute('INSERT INTO x DEFAULT VALUES', ())
    rows = await db.fetchall('SELECT COUNT(*) FROM x', ())
    assert rows == [(1,)]


async def test_database_reopen_sees_prior_writes(tmp_path: Path) -> None:
    path = tmp_path / 'butter.db'
    first = await Database.open(path)
    try:
        await first.execute_ddl('CREATE TABLE t (v TEXT)')
        await first.execute('INSERT INTO t VALUES (?)', ('persisted',))
    finally:
        await first.close()

    second = await Database.open(path)
    try:
        rows = await second.fetchall('SELECT v FROM t', ())
        assert rows == [('persisted',)]
    finally:
        await second.close()


# --- SqliteConversationHistory ----------------------------------------------


async def test_history_round_trip(db: Database) -> None:
    history = await SqliteConversationHistory.create(db)
    entry = _entry('t1', user='hi', reply='hello', ts=12.5)
    await history.append(entry)
    assert await history.recent(10) == (entry,)


async def test_history_returns_oldest_first_within_window(db: Database) -> None:
    history = await SqliteConversationHistory.create(db)
    entries = [_entry(f't{i}', ts=float(i)) for i in range(5)]
    for entry in entries:
        await history.append(entry)
    # All five, in insertion order.
    assert await history.recent(10) == tuple(entries)


async def test_history_limit_caps_window(db: Database) -> None:
    history = await SqliteConversationHistory.create(db)
    entries = [_entry(f't{i}', ts=float(i)) for i in range(5)]
    for entry in entries:
        await history.append(entry)
    # Last 3 — oldest-first within the window.
    recent = await history.recent(3)
    assert recent == tuple(entries[-3:])


@pytest.mark.parametrize('limit', [0, -1, -100])
async def test_history_non_positive_limit_returns_empty(db: Database, limit: int) -> None:
    history = await SqliteConversationHistory.create(db)
    await history.append(_entry('t1'))
    assert await history.recent(limit) == ()


async def test_history_preserves_null_assistant_reply(db: Database) -> None:
    history = await SqliteConversationHistory.create(db)
    entry = _entry('t1', reply=None)
    await history.append(entry)
    (got,) = await history.recent(10)
    assert got.assistant_reply is None


async def test_history_persists_across_reopen(tmp_path: Path) -> None:
    path = tmp_path / 'butter.db'
    first_db = await Database.open(path)
    try:
        first_history = await SqliteConversationHistory.create(first_db)
        await first_history.append(_entry('t1', user='hi'))
    finally:
        await first_db.close()

    second_db = await Database.open(path)
    try:
        second_history = await SqliteConversationHistory.create(second_db)
        (got,) = await second_history.recent(10)
        assert got.turn_id == 't1'
        assert got.user_input == 'hi'
    finally:
        await second_db.close()


async def test_history_create_is_idempotent(db: Database) -> None:
    # Two consumers bootstrapping against the same DB must both succeed
    # without dropping data.
    first = await SqliteConversationHistory.create(db)
    await first.append(_entry('t1'))
    second = await SqliteConversationHistory.create(db)
    assert await second.recent(10) == (_entry('t1'),)


async def test_database_close_waits_for_in_flight_ops(tmp_path: Path) -> None:
    # close() acquires the writer lock — so any in-flight execute/fetchall
    # completes before the connection is torn down. Without the fix, the
    # in-flight thread can hit 'Cannot operate on a closed database'.
    database = await Database.open(tmp_path / 'butter.db')
    await database.execute_ddl('CREATE TABLE t (v INTEGER)')
    # Launch writes and a close in the same scheduling tick; the lock
    # decides who goes first but neither should crash.
    writes = [database.execute('INSERT INTO t VALUES (?)', (i,)) for i in range(10)]
    await asyncio.gather(*writes, database.close())


async def test_history_concurrent_appends_all_land(db: Database) -> None:
    # Writer lock should serialise concurrent appends without losing any.
    history = await SqliteConversationHistory.create(db)
    entries = [_entry(f't{i}', ts=float(i)) for i in range(20)]
    await asyncio.gather(*(history.append(e) for e in entries))
    got = await history.recent(50)
    assert len(got) == 20
    # Order within a batch is not contract — set comparison is enough.
    assert {e.turn_id for e in got} == {e.turn_id for e in entries}
