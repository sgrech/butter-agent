# Notes Plugin Spec

## 1. Purpose

{2-3 sentences describing why this feature exists and what problem it solves. Frame it from the user's perspective.}

## 2. Scope

### In Scope

- Notes plugin capability surface (create / list / read / ?delete)
- Persistence model — does the plugin own its own SQLite store, or share `Database` with core?
- Manifest declaration: inputs, outputs, `blast_radius=local-write`, gate strategy
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

Open question: schema lives in plugin-owned SQLite file under e.g. `~/.butter-agent/plugins/notes.db`, OR in a shared `Database` injected by core. Decide in §6.

## 5. API Surface

### Capabilities

| Capability | Inputs | Outputs | blast_radius | gate |
|------------|--------|---------|--------------|------|
| `notes.create` | `content: str` | `note_id: int`, `created_at: str` | `local-write` | `confirm` |
| `notes.list` | `limit?: int` | `notes: list[{id, content, created_at}]` | `read-only` | none |
| `notes.read` | `note_id: int` | `content: str`, `created_at: str` | `read-only` | none |

## 6. Interactions

- Task executor enforces the `confirm` gate before invoking `notes.create`; REPL renders the prompt (already wired in PR #18).
- Variable pool: `notes.create` accepts `$t.time` from a prior `clock.now` step — exercises plan-level `$variable` resolution end-to-end with a real write.
- **Decision required:** Database ownership.
  - Option A — plugin owns its SQLite file under XDG data dir. Pro: blast-radius isolation, no shared schema migrations. Con: every write-plugin re-implements connection lifecycle.
  - Option B — core exposes a `Database` capability plugins request in their manifest. Pro: shared lifecycle, single backup target. Con: weakens "plugins cannot read each other's internal state" invariant unless namespaced.
  - Recommend A for v1; revisit when second write-plugin lands.

## 7. Migration Strategy

- New plugin, no migration. First run creates the notes DB on demand.
- No feature flag — gate-handler already lives in core, REPL already suspends spinner during prompts.
- Live-REPL acceptance: model emits `clock.now → notes.create`, REPL prompts for confirm, write happens, `notes.list` reads it back in the same session.
