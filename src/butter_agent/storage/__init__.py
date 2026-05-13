"""SQLite-backed persistence for butter-agent.

Each consumer of storage (conversation history, future plugin state,
future memory cache) owns its own table. The shared `Database` only
manages the connection, schema bootstrap, and async wrapping — it does
not arbitrate cross-consumer access, and consumers do not read each
other's tables.

Lives outside `core/` so swapping storage backends (e.g. a future
provider for a different SQL dialect) does not touch the loop or the
executor — invariant #1 (loop shape never changes) is preserved.
"""
