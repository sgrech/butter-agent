"""Tests for the SQLite storage layer.

The storage module owns persistence behind seams declared in
`core/context_manager.py`. Today only `Database` lifecycle is under
test — conversation history is intentionally in-memory; a persisted
history class lands with the session concept.

Covered: path expansion, parent-dir creation, async wrapping,
idempotent DDL, writer-lock serialisation, and clean close-with-
in-flight-ops behaviour.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from butter_agent.storage.sqlite import Database


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


async def test_database_writer_lock_serialises_concurrent_writes(db: Database) -> None:
    # Writer lock serialises concurrent executes without dropping any.
    await db.execute_ddl('CREATE TABLE t (v INTEGER)')
    await asyncio.gather(*(db.execute('INSERT INTO t VALUES (?)', (i,)) for i in range(20)))
    rows = await db.fetchall('SELECT v FROM t', ())
    assert len({row[0] for row in rows}) == 20


# --- execute_write / query (slice 3 primitives) -----------------------------


async def test_execute_write_reports_insert_rowid(db: Database) -> None:
    await db.execute_ddl('CREATE TABLE t (id INTEGER PRIMARY KEY AUTOINCREMENT, v TEXT)')
    first = await db.execute_write('INSERT INTO t (v) VALUES (?)', ('a',))
    second = await db.execute_write('INSERT INTO t (v) VALUES (?)', ('b',))
    assert (first.last_row_id, first.row_count) == (1, 1)
    assert (second.last_row_id, second.row_count) == (2, 1)


async def test_execute_write_reports_affected_rowcount(db: Database) -> None:
    await db.execute_ddl('CREATE TABLE t (id INTEGER PRIMARY KEY AUTOINCREMENT, k TEXT)')
    for k in ('x', 'x', 'y'):
        await db.execute_write('INSERT INTO t (k) VALUES (?)', (k,))
    updated = await db.execute_write('UPDATE t SET k = ? WHERE k = ?', ('z', 'x'))
    assert updated.row_count == 2
    deleted = await db.execute_write('DELETE FROM t WHERE k = ?', ('z',))
    assert deleted.row_count == 2


async def test_query_returns_column_keyed_dicts(db: Database) -> None:
    await db.execute_ddl('CREATE TABLE t (id INTEGER PRIMARY KEY AUTOINCREMENT, v TEXT)')
    await db.execute_write('INSERT INTO t (v) VALUES (?)', ('hello',))
    rows = await db.query('SELECT * FROM t', ())
    assert rows == [{'id': 1, 'v': 'hello'}]


async def test_query_empty_result_is_empty_list(db: Database) -> None:
    await db.execute_ddl('CREATE TABLE t (id INTEGER PRIMARY KEY AUTOINCREMENT)')
    assert await db.query('SELECT * FROM t', ()) == []
