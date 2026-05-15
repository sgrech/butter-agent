# Database Plugin — Full-Text Search Spec

## 1. Purpose

Extend the built-in `database` plugin with full-text search so write-plugins can offer real search over their own rows without leaving the local-first contract. This is the follow-up the database-plugin spec explicitly deferred (`database-plugin.md` §2 Out of Scope: *"Full-text / vector search — separate capability if ever needed"*). The motivating consumer is `butter-plugin-notes`: a notes store with only `list` is a write-only pile that gets harder to navigate every time it grows.

This spec covers **lexical** full-text search only (SQLite FTS5). **Semantic** search (embeddings, meaning/synonym matching) is deliberately a separate, later spec because it crosses the `network` blast-radius boundary and needs a model — see §7.

## 2. Scope

### In Scope

- Two new `database` capabilities: `database.define_fts` and `database.search`, both `internal: true` (planner-invisible, plugin-to-plugin only) like the rest of the `database` surface.
- **External-content FTS5** index (`content='{table}'`, `content_rowid='id'`) plus AFTER INSERT/UPDATE/DELETE sync triggers, so the base table keeps its typed schema, surrogate `id`, and `NOT NULL` columns unchanged. The existing `define_table`/`insert`/`update`/`delete` contract is untouched.
- Idempotent index definition (`CREATE VIRTUAL TABLE IF NOT EXISTS` + `CREATE TRIGGER IF NOT EXISTS`), mirroring `define_table`'s idempotency.
- One-time backfill of rows that predate the index, via FTS5's `'rebuild'` command, run inside `define_fts`.
- Injection-proof, syntax-error-proof query handling: the caller passes natural text; the plugin builds the FTS5 MATCH expression. The caller never hand-writes FTS5 query syntax.
- Stays `blast_radius = "local-write"` — DDL + local index maintenance, no new IO surface, no invariant #7 expansion.
- Namespacing reuses core's **existing** single-`table`-key rewrite with **zero core changes** (see §6).

### Out of Scope

- **Semantic / vector search** — separate spec; different blast radius (§7).
- Snippet / highlight extraction (`snippet()`, `highlight()`), match-offset metadata — a later enhancement; v1 returns whole rows.
- Configurable bm25 column weighting, custom tokenizers/stemmers per caller — v1 fixes one sensible tokenizer (§4).
- Exposing FTS5 boolean/`NEAR`/column-filter query syntax to callers or the planner — the plugin owns expression construction (§5).
- Multi-table / join search — search is scoped to one namespaced table.
- Migration tooling beyond the one-shot `'rebuild'` backfill.

## 3. User Stories

- As a plugin author, I want to declare `requires = ["…", "database.define_fts", "database.search"]`, call `define_fts` once at table-ensure time, and then `search(table, query)` — and get ranked matching rows back in the same `{rows: [...]}` shape as `select`, so my search capability is a thin pass-through.
- As a butter-agent user, when I ask "find my note about the dentist" the agent's `notes.search` step returns the right note even though I didn't remember its exact wording or id.
- As a butter-agent operator, I want full-text search to stay inside the single local SQLite file with no network egress, so enabling search on a plugin never changes its blast radius or my backup/wipe story.
- As a plugin author, I want to pass the user's raw phrase as `query` without escaping FTS5 syntax, and never have a stray `)` or `"` turn into a query error or an unintended boolean.

## 4. Data Model

For a base table `{ns}` (already core-namespaced to `{caller}__{table}`, e.g. `notes__entries`) the plugin derives, owns, and never exposes:

| Object | Name | Created by | Purpose |
|--------|------|-----------|---------|
| FTS index | `{ns}_fts` | `define_fts` | `CREATE VIRTUAL TABLE … USING fts5(<cols>, content='{ns}', content_rowid='id')` |
| insert trigger | `{ns}_ai` | `define_fts` | AFTER INSERT → mirror new row into `{ns}_fts` |
| update trigger | `{ns}_au` | `define_fts` | AFTER UPDATE → delete-then-reinsert the FTS row |
| delete trigger | `{ns}_ad` | `define_fts` | AFTER DELETE → tombstone the FTS row |

- The `_fts` suffix and trigger names are **derived inside the plugin** from the already-namespaced `{ns}`. Core does not know about them and does not need to (§6). A caller-supplied table name containing `__` is already rejected upstream, so the derived names cannot collide across namespaces.
- `content_rowid='id'` binds the FTS index to the base table's existing auto-surrogate `id` (`INTEGER PRIMARY KEY AUTOINCREMENT` per `database-plugin.md` §5). External-content FTS5 stores no document copy — it indexes terms and references base rows by rowid, so storage overhead is the index only.
- **Tokenizer (fixed, v1):** `unicode61` with `remove_diacritics 2` plus the `porter` stemming wrapper — i.e. `tokenize = "porter unicode61 remove_diacritics 2"`. Rationale: diacritic-insensitive, and stemming lets "dentist" match "dentists" / "buttered" match "butter", which is the forgiving behaviour the notes use case wants. Per-caller tokenizer choice is out of scope.
- Columns indexed are caller-declared (`define_fts(table, columns=[...])`); each must be an existing `text`/`datetime` column on the base table (validated against the same identifier rule the rest of the plugin uses). Non-text columns are rejected with a descriptive `DatabasePluginError`.

## 5. API Surface

Both capabilities are `internal: true`; `blast_radius` stays the plugin-level `local-write`.

| Capability | Inputs | Outputs | Notes |
|------------|--------|---------|-------|
| `database.define_fts` | `table: str`, `columns: list[str]` | `{table: str}` | Idempotent. Base table must already exist (`define_table` first) — else `DatabasePluginError`. Creates the FTS index + 3 triggers `IF NOT EXISTS`, then runs the one-time `'rebuild'` backfill. Returns the fully-qualified base table name. |
| `database.search` | `table: str`, `query: str`, `limit?: int`, `order?: "rank" \| "id"` | `{rows: list[dict]}` | Same row shape as `select` (whole base rows, `SELECT b.*`). `order` defaults to `"rank"` (bm25, most-relevant first); `"id"` gives oldest-first parity with `select`/`list`. `limit` is a non-negative int, validated exactly as `select`'s. |

### Query construction (the safety boundary)

`query` is **natural text, never FTS5 syntax.** The plugin builds the MATCH expression:

1. Split `query` on whitespace into terms; drop empties.
2. Empty result (blank/whitespace-only query) → `DatabasePluginError` ("query must contain at least one term"). Callers validate their own user-facing emptiness too, but the store refuses to run a matchless FTS query rather than scanning everything.
3. Each term is wrapped as a quoted FTS5 string literal (internal `"` doubled) with a `*` prefix-suffix appended **outside** the quotes: `"term"*`. Quoting neutralises every FTS5 metacharacter (`(`, `)`, `:`, `^`, `OR`, `NEAR`, …) so a term can never become an operator or a syntax error; the trailing `*` keeps matching forgiving (prefix match, complementing the porter stemmer).
4. Terms are joined by a space → implicit FTS5 AND. `find butter shop` ⇒ all of `"find"* "butter"* "shop"*`.
5. The expression is bound as a **parameter** (`WHERE {ns}_fts MATCH ?`), never interpolated — defence in depth on top of the quoting.

SQL shape:

```sql
SELECT b.* FROM {ns} b
JOIN {ns}_fts f ON b.id = f.rowid
WHERE {ns}_fts MATCH ?
ORDER BY <bm25({ns}_fts) | b.id>
LIMIT ?            -- only when limit given
```

`{ns}` is the validated, core-namespaced identifier (same `_ident` boundary as the rest of the plugin); `_fts` is appended to that validated string, so it inherits the injection-safe charset. Only the MATCH expression and `limit` are bound params.

### Why no gate

Identical reasoning to `database-plugin.md` §5: `database.search` is an inner call, never a plan step. The gate (if any) sits on the user-visible caller capability (e.g. `notes.search`). v1: search is read-only, typically ungated by the planner anyway.

## 6. Interactions

**Zero core changes — this is the "cheap" claim, made precise:**

- Core's `_PluginContext` namespacing rewrites exactly one input key — `table` — for the one blessed `database` plugin, using closed-over caller identity (`database-plugin.md` §6.5). `define_fts` and `search` both take `table` and nothing else identity-bearing, so they flow through that existing hook **unmodified**. Core gains two capability *names* in the `database` manifest; it gains no new code, no new argument map, no awareness of `_fts`.
- `requires` validation is already generic: a consumer adding `"database.define_fts"`/`"database.search"` to its manifest is validated by the same registry logic that handles `database.insert` today. The transitive blast-radius check still passes because both new capabilities are `local-write`.
- Invariant #6 (namespace isolation) holds without new enforcement: the plugin derives `{ns}_fts` and trigger names *from* the already-prefixed `{ns}` it receives. It cannot derive another namespace's index because it never sees an un-prefixed or foreign name — same property the base CRUD already relies on.
- Invariant #7 (blast radius only restricted, never expanded): unchanged. FTS5 is local SQLite DDL + triggers. No network, no model, no new process. A consumer's declared `local-write` still covers everything reachable.

**`storage.sqlite.Database` reuse:** no changes. `execute_ddl` already runs arbitrary `CREATE VIRTUAL TABLE` / `CREATE TRIGGER`; `query` already returns column-keyed dicts from arbitrary `SELECT`. `define_fts`'s DDL goes through `execute_ddl` (idempotent-by-convention, lock-serialised); the `'rebuild'` backfill is a write via `execute`/`execute_ddl`; `search` is a `query`. The FTS layer is pure SQL the plugin emits — the IO layer needs nothing new.

**FTS5 availability gate:** asserted in `build_database_plugin` (or first `define_fts`) by attempting a throwaway `CREATE VIRTUAL TABLE … USING fts5` in a savepoint, or reading `PRAGMA compile_options`. If FTS5 is absent, `define_fts` raises a descriptive `DatabasePluginError` — **no silent fallback to a slower scan.** A consumer whose `define_fts` fails records it as that step's `failure_reason` (invariant #6 — plugin may raise, loop still synthesises) and its `search` capability is simply unavailable; it does not silently degrade to returning everything. (Bundled SQLite confirmed FTS5-enabled: `sqlite_version 3.51.0`.)

**Downstream consumer (separate change, tracked separately):** `butter-plugin-notes` v0.3.0 —

- `requires` gains `database.define_fts`, `database.search`.
- `_ensure_table` additionally calls `define_fts(table='entries', columns=['content'])` under the same once-per-process lock (idempotent, so safe).
- `_search` drops the full-table pull + Python `needle in content.lower()` loop and calls `database.search(table='entries', query=..., limit=...)`, returning `{notes: rows}`.
- **Behavioural change to document in the notes repo:** matching shifts from arbitrary infix substring to porter-stemmed prefix-term AND (e.g. `search("utter")` no longer matches "butter"; `search("dentist")` now matches "dentists"). Ordering: notes decides `order="id"` (keep its documented oldest-first contract) vs `order="rank"` (relevance). `notes.search` shipped in 0.2.0 with no pinned consumers, so the contract change is low-blast but must be a deliberate, documented choice in the notes follow-up — out of scope here.

## 7. Why semantic search is a different spec

Stated so the boundary is explicit and not silently conflated:

| | FTS5 (this spec) | Semantic (future spec) |
|---|---|---|
| Blast radius | `local-write` (unchanged) | `network` or bundled model — **expands** the contract |
| Where it lives | host `database` plugin extension | likely a distinct plugin / host memory layer, composed via the variable pool |
| Match quality | lexical: stem + prefix, ranked | meaning/synonym ("grocery" ⇄ "buy butter and milk") |
| Cost to ship | two capabilities, zero core change | embedding model + vector store + invariant-7 review |

FTS5 is the high-value, low-cost tier and the only one in scope. Semantic search is not a bigger version of this — it is a different architectural decision and gets its own spec when justified.

## 8. Migration Strategy

- **New capabilities, additive.** No existing `database` behaviour changes; no stored data is rewritten. Existing tables without an FTS index are entirely unaffected until a consumer calls `define_fts`.
- **Pre-existing rows are searchable.** External-content FTS5 does not auto-index rows written before the index existed. `define_fts` runs `INSERT INTO {ns}_fts({ns}_fts) VALUES('rebuild')` after ensuring the virtual table — at single-user scale this is cheap and always-safe to re-run, so it executes every `define_fts` call rather than needing creation detection. This is the entire migration step for a consumer like `notes` that already has `notes__entries` rows.
- **Lands before the notes follow-up** so notes can target the final contract directly (same sequencing discipline as `database-plugin.md` §7: the store capability ships before its consumer).
- **Live-acceptance:**
  1. `define_table` a table, `insert` 3 rows, `define_fts` it, `search` a term present in 1 row → exactly that row, `{rows:[...]}` shape identical to `select`.
  2. `insert` a 4th row post-`define_fts`, `search` a term only in it → returned (triggers keep the index live).
  3. `delete` a matching row, re-`search` → gone (delete trigger).
  4. `search("alpha)beta")` (FTS5 metachars) → treated as literal terms, no error, no injection.
  5. `define_fts` twice → idempotent, no error, no duplicate triggers.
  6. A second plugin cannot name another plugin's `_fts` table — structurally impossible, same proof as the base CRUD (no un-prefixed-name surface).
- **Test layers** mirror `tests/test_database_plugin.py`: unit tests against a real in-memory `Database` for trigger sync + query construction + the metachar/empty-query guards; the manifest round-trip gains the two new capability names. The notes-side end-to-end (real host executor, `notes.search` through this) lives in the butter-agent integration suite when the notes follow-up lands.
