# Notes Plugin Spec

> **Status: externalized.** Notes was implemented as a core built-in
> (PR #22) then moved to its own repo per
> `plugin-externalization.md` — it now lives at
> [`butter-plugin-notes`](https://github.com/sgrech/butter-plugin-notes)
> (`v0.1.0`), loaded via a `config.toml` `[[plugin]]` declaration. This
> spec remains the authoritative description of the *capability* (data
> model, API surface, gate/chain behaviour); it is no longer a
> description of where the code lives. The §4 `database` persistence
> constraints still hold — they are a `database` contract, not a
> notes-location concern.

## 1. Purpose

Notes is butter-agent's first `local-write` plugin: persistent free-form note capture from the REPL. It exists to prove the gate-handler `confirm` path end-to-end against a real model-emitted plan, to exercise the agent-mediated variable-pool data channel via `clock.now → notes.create`, and to serve as the worked example future write-plugins copy.

## 2. Scope

### In Scope

- Notes plugin capability surface (`create` / `list` / `read`)
- Persistence via the `database` plugin (see `database-plugin.md`) — namespace-isolated to `notes__entries`
- Manifest declaration: plugin-level `blast_radius=local-write`, `requires = ["database.define_table", "database.insert", "database.select"]`, per-capability `gate`
- Live exercise of the gate-handler `confirm` path against a real model-emitted plan
- Integration with `clock` plugin for chained plans (`clock.now → notes.create` using `$t.time`)

### Out of Scope

- Cross-session note search / semantic indexing (deferred)
- Note editing / versioning beyond create + read (deferred)
- Multi-user / sharing concerns — butter-agent is single-user, local-first

## 3. User Stories

- As a butter-agent user, I want to ask "save a note that says X" and have the REPL confirm before writing to disk so that local writes are never silent.
- As a butter-agent user, I want to chain `clock.now` → `notes.create` so that "save the current time as a note" produces a single confirmed plan.
- As a plugin author, I want a worked example of a `local-write` plugin so that future write-side plugins follow the same manifest + gate pattern.

## 4. Data Model

| Field | Type | Constraints | Description |
|-------|------|-------------|-------------|
| id | int | PK, autoincrement | Note row id |
| content | text | not null | Note body |
| created_at | datetime | not null, default now | Creation timestamp |

Schema lives in the shared `database` plugin under the namespaced table `notes__entries`. The notes plugin never sees raw SQL; it calls `ctx.call("database.define_table", ...)` once and `ctx.call("database.insert" | "database.select", ...)` for writes/reads. See `database-plugin.md` for the namespace-isolation rules.

> **Constraints discovered in database slice 3 (memory-mcp 1430) — the next session must build to these, not the draft above:**
> - Every `database.*` call addresses its table via a single **`table`** key (not `name`), passing the **bare** name (`"entries"`); core prefixes it to `notes__entries`. Notes never sends or sees the `notes__` prefix.
> - ColumnSpec `default` is **advisory only — NOT emitted into DDL**. So `created_at default now` does **not** exist at the DB layer: `notes.create` must populate `created_at` itself (this is exactly what the `clock.now → notes.create($t.time)` variable-pool chain is for; without an upstream `clock.now`, notes generates the timestamp). `datetime` columns are stored as ISO-8601 **TEXT**.
> - `update`/`delete` require a non-empty equality `where`; `select` `where` is optional, equality-only AND. A caller-supplied `table` containing `__` is rejected by core.

## 5. API Surface

### Capabilities

Plugin-level `blast_radius = "local-write"` (the strictest tier required across the plugin's write capabilities; reads inherit the plugin-level value — there is no per-capability blast radius in the manifest schema today). Per-capability `gate` is declared by the planner per plan step; the values below are the recommended defaults the model should emit.

| Capability | Inputs | Outputs | gate (recommended) |
|------------|--------|---------|--------------------|
| `notes.create` | `content: str` | `note_id: int`, `created_at: str` | `confirm` |
| `notes.list` | `limit?: int` | `notes: list[{id, content, created_at}]` | `none` |
| `notes.read` | `note_id: int` | `content: str`, `created_at: str` | `none` |

## 6. Interactions

- Task executor enforces the `confirm` gate before invoking `notes.create`; REPL renders the prompt (already wired in PR #18).
- Variable pool: `notes.create` accepts `$t.time` from a prior `clock.now` step — exercises plan-level `$variable` resolution end-to-end with a real write.
- Persistence: notes calls into `database` via `PluginContext.call`. The namespace is closed over by the executor (always `notes`); notes cannot read or write any other plugin's tables. Cross-plugin data flow remains agent-mediated via the variable pool.

## 7. Migration Strategy

- New plugin, no migration. First run creates the notes DB on demand.
- No feature flag — gate-handler already lives in core, REPL already suspends spinner during prompts.
- Live-REPL acceptance: model emits `clock.now → notes.create`, REPL prompts for confirm, write happens, `notes.list` reads it back in the same session.
