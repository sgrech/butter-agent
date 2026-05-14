# Database Plugin Spec

## 1. Purpose

Provide a capability-mediated shared SQLite store so write-plugins (notes, future memory/journal/etc.) don't each re-implement connection lifecycle, and butter-agent has a single backup target. Isolation between plugins moves from filesystem-level (per-plugin DB files) to capability-level (per-caller namespace enforced by core). Keeps the "plugins are the only IO surface" invariant.

## 2. Scope

### In Scope

- Single SQLite file under XDG data dir (`~/.local/share/butter-agent/butter.db`).
- Structured CRUD API — no raw SQL across the capability boundary.
- Per-caller namespace, with caller identity supplied **by core**, not by the caller.
- Idempotent table definition (`CREATE TABLE IF NOT EXISTS …`) keyed on `{caller}__{table}`.
- Hard isolation: a plugin can only ever access its own namespace. No cross-namespace reads, ever. The only inter-plugin data channel is the variable pool, written by the agent's plan.
- "Internal" capability flag so the database plugin's surface is invisible to the planner — only other plugins call it. Reserved for infrastructure plugins (database, future log/fs/http).
- Plugin-calling-plugin pattern: core provides each plugin handler with a client carrying caller identity.

### Out of Scope

- Migrations beyond `CREATE TABLE IF NOT EXISTS` — schema evolution lands in a follow-up.
- Transactions spanning multiple plugins — each call is its own implicit txn for v1.
- Full-text / vector search — separate capability if ever needed.
- Backup / export tooling — separate core concern, never a plugin capability.
- Concurrency beyond SQLite's default WAL — no connection pooling.

## 3. User Stories

- As a plugin author, I want to declare e.g. `requires = ["database.define_table", "database.insert", "database.select"]` in my manifest and reach SQLite through `ctx.call(...)` so I can persist state without managing connections myself.
- As a butter-agent user, I want all local writes from all plugins in one inspectable file so I can back up or wipe butter-agent state in one move.
- As a butter-agent operator, I want plugins to be unable to read each other's rows under any circumstance so installing a third-party plugin can never exfiltrate data from another plugin.

## 4. Data Model

Physical schema (single file, all tables prefixed by owning plugin):

```
{plugin_name}__{table_name}
```

Example: `notes__entries`, `clock__alarms`.

Per-call namespace is derived from the caller's plugin name at the registry/executor layer — never read from the call arguments.

| Concept | Where it lives | Description |
|---------|----------------|-------------|
| caller_plugin | core (registry) | Identity of the plugin invoking a `database.*` capability. Injected into the client. |
| namespace | derived | `caller_plugin` — used as the table-name prefix. Always equals the caller; never an argument. |
| table schema | calling plugin | Plugin declares table shape via `db.define_table(name, columns)` on first use. |

## 5. API Surface

All capabilities are flagged `internal: true` — invisible to the planner, callable only by other plugin handlers via `PluginContext.call`. The plugin-level `blast_radius = "local-write"` covers the strictest tier reached by any capability; the manifest schema does not (yet) support per-capability blast radii, so reads inherit the plugin value.

| Capability | Inputs | Outputs | Notes |
|------------|--------|---------|-------|
| `database.define_table` | `name: str`, `columns: dict[str, ColumnSpec]` | `{table: str}` | Idempotent. Returns the fully-qualified table name. |
| `database.insert` | `table: str`, `row: dict` | `{id: int}` | Gate sits on the calling capability (e.g. `notes.create`), not here. |
| `database.select` | `table: str`, `where?: dict`, `limit?: int`, `order_by?: str` | `{rows: list[dict]}` | `where` is equality-only for v1. |
| `database.update` | `table: str`, `where: dict`, `set: dict` | `{updated: int}` | Same gate-on-caller rule. |
| `database.delete` | `table: str`, `where: dict` | `{deleted: int}` | Same gate-on-caller rule. |

`ColumnSpec` (v1, minimal):

```python
{"type": "text" | "integer" | "real" | "blob" | "datetime",
 "not_null": bool,            # default False
 "default": str | None,       # SQL literal, advisory only
 "primary_key": bool}         # default False; one column max
```

`id` columns are auto-added as `INTEGER PRIMARY KEY AUTOINCREMENT` unless the caller declares a `primary_key` column explicitly.

### Why no gate on `database.*` writes

Gates are enforced at the task-plan step level. `database.insert` is never a plan step — it's an inner call. The user-visible capability (`notes.create`) carries the gate. Putting a second gate on `database.insert` would either double-prompt or require core to suppress one — both worse than the rule "gates live on the outermost step in the plan."

## 6. Interactions

**Plugin-calling-plugin pattern (new):**

1. The `Plugin` Protocol gains a third parameter on `execute()`: `context: PluginContext`. This is a breaking change to the existing Protocol in `core/registry.py` — pre-1.0, acceptable. The clock plugin is updated alongside.
2. Manifests gain capability-granular `requires` (e.g. `requires = ["database.insert", "database.select"]`) and per-capability `internal: bool`. Only `internal: true` capabilities are legal targets of `requires`.
3. `RegistryBuilder.build()` topo-sorts plugins by `requires`, rejects cycles, and validates each non-internal capability's declared blast radius covers the transitive union of its reachable internal capabilities' radii.
4. At step execution the task executor constructs a `PluginContext` whose `.call(capability, args)` is restricted to the caller's `requires` set and whose caller identity is closed over (sourced from the registry, never from arguments).
5. Inside a calling plugin (e.g. `notes`), `await ctx.call("database.insert", {"table": "entries", ...})` is rewritten by `PluginContext` into a `database.insert` invocation against `notes__entries`. The database plugin handler sees only the fully-qualified table name; namespace prefixing happens in `PluginContext` construction. Any caller-supplied table name containing `__` is rejected.

**Hard namespace isolation:**

- A plugin's namespace is its own name. The `DatabaseClient` has no `namespace` argument — there is nothing for a plugin to set, override, or lie about.
- Cross-plugin data flow is exclusively agent-mediated via the variable pool (e.g. plan step `clock.now → notes.create($t.time)`). Plugins never reach into each other's data.
- The database plugin code itself only ever sees fully-qualified table names. It does not parse plugin identity. All identity logic lives in the client construction inside core.

**Storage location:**

- `XDG_DATA_HOME` if set, else `~/.local/share/butter-agent/butter.db`.
- SQLite opened with `PRAGMA journal_mode=WAL` for concurrent reads during writes.

**Reuse the existing `Database` class:**

- `butter_agent.storage.sqlite.Database` already wraps one SQLite file with an async lock and `to_thread` dispatch. The `database` plugin is the *capability boundary* on top of that connection — it does not replace `Database`, it wraps it. Two different concerns: `Database` is the IO layer, the plugin is the identity-enforcing API surface.

## 7. Migration Strategy

- New plugin, no existing data to migrate.
- Lands **before** `notes` so `notes` can target the contract directly.
- v1 ships without schema migration tooling — `define_table` is `CREATE TABLE IF NOT EXISTS` and that's it. Schema changes are a follow-up spec.
- Live-acceptance: a smoke-test plugin (or `notes` itself once it lands) defines a table, inserts a row, reads it back. A second plugin attempting to name another plugin's table is structurally impossible — the client API has no surface for it.
