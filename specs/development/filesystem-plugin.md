# Filesystem Plugin Spec

> **Status: implemented & externalised.** Lives in its own repo
> [`butter-plugin-filesystem`](https://github.com/sgrech/butter-plugin-filesystem)
> (`v0.1.0`), loaded via a `config.toml` `[[plugin]]` declaration. This
> spec is the authoritative description of the *capability* (path model,
> API surface, gate/safety behaviour), not of where the code lives.
> Requires a host providing `PluginContext.config` —
> `plugin-config-injection.md`, butter-agent ≥ `v0.1.0`.

## 1. Purpose

Filesystem is butter-agent's reference for the **operator-config gate**
pattern: a `local-write` plugin whose most destructive capability
(`delete`) is gated by an operator-declared, model-invisible config flag
rather than by model discretion. It gives the agent first-class local
disk access (navigate, inspect, read, write, edit, search) and proves
that a plugin can hold a hard policy line the planner cannot argue past.

## 2. Scope

### In Scope

- Navigation/inspection (`pwd`, `cd`, `list_dir`, `stat`), reads
  (`read_file`), mutation (`write_file`, `edit_file`), search
  (`find_files`, `search_content`), and operator-gated `delete`.
- A mutable plugin-held working directory (`_cwd`) the model moves only
  via `cd`; relative paths resolve against it.
- Opportunistic `fd`/`ripgrep` delegation with a pure-stdlib fallback —
  zero third-party runtime dependencies.
- `delete` safety: operator `allow_delete` / `allow_recursive_delete`
  config flags, `dry_run` preview, unconditional protected-path
  backstops, trash-by-default.
- Consuming `PluginContext.config` (the first plugin to do so).

### Out of Scope

- A root jail / path prefix sandbox — by explicit design the model may
  `cd` anywhere; containment is on the destructive side only.
- Diff/patch application — `edit_file` is exact unique-match replace;
  whole-file replacement is `write_file`.
- `.gitignore`-aware search in the **stdlib fallback** (only the
  `fd`/`ripgrep` backends are; the fallback prunes a fixed VCS/build set).
- Binary-file editing, encodings other than UTF-8, recursive intermediate
  directory creation on write.

## 3. User Stories

- As a butter-agent user, I want to ask "find every TODO under src and
  show me the file:line" and get a grounded answer from the real tree.
- As a butter-agent user, I want "delete the build dir" to be **refused
  outright** unless I, the operator, enabled deletion in config — the
  model cannot turn it on for me.
- As an operator, I want a deleted file to land in a recoverable
  `.butter-trash/` by default, and a recursive wipe to require an
  explicit second opt-in.
- As a plugin author, I want a worked example of reading
  `PluginContext.config` to gate a capability.

## 4. Working-Directory & Path Model

- The plugin holds `_cwd`, initialised to the process CWD, mutated
  **only** by the `cd` capability. Every relative `path` input resolves
  against it; `~` expands; `..` normalises.
- There is **no root jail**. Boundaries are drawn by the operator
  (config flags + the planner's `human` gate), not a path prefix —
  recorded as an explicit scope decision.
- `delete` resolves its target **without following a final symlink**
  (`_resolve_unfollowed`): deleting a link removes the link, never its
  target. All other capabilities fully resolve (`_resolve`).
- No persistent store, no `database` dependency, no `requires`: the
  filesystem *is* the state. Cross-plugin data flow stays agent-mediated
  via the variable pool.

## 5. API Surface

### Capabilities

Plugin-level `blast_radius = "local-write"` (the strictest tier across
the plugin's capabilities; reads/search inherit it — there is no
per-capability blast radius in the manifest schema). Per-capability
`gate` is declared by the planner per plan step; the values below are the
recommended defaults the model should emit.

| Capability | Inputs | Outputs | gate (recommended) |
|------------|--------|---------|--------------------|
| `pwd` | — | `cwd: str` | `none` |
| `cd` | `path: str` | `cwd: str` | `none` |
| `list_dir` | `path?: str` | `path: str`, `entries: [{name,type,size}]` | `none` |
| `stat` | `path: str` | `path, type, size: int, mtime: str` | `none` |
| `read_file` | `path: str`, `offset?: int`, `limit?: int` | `path, content: str, lines: int, truncated: bool` | `none` |
| `write_file` | `path: str`, `content: str` | `path, bytes_written: int` | `confirm` |
| `edit_file` | `path: str`, `old: str`, `new: str` | `path, replaced: int` | `confirm` |
| `find_files` | `pattern: str`, `path?: str`, `limit?: int` | `paths: [str], backend: str` | `none` |
| `search_content` | `query: str`, `path?: str`, `glob?: str`, `limit?: int` | `matches: [{path,line,text}], backend: str` | `none` |
| `delete` | `path: str`, `recursive?: bool`, `dry_run?: bool` | `path, trashed: bool, recursive: bool, dry_run: bool` | **`human`** |

- `edit_file` requires `old` to occur **exactly once** (raises on 0 or
  >1, file untouched); `new` may be empty to delete the matched text.
- `find_files`/`search_content` set `backend` to `fd`/`ripgrep` when the
  binary is on PATH, else `stdlib`. `search_content` `query` is a regex
  (pre-validated with Python `re`).
- `delete` is **disabled** unless operator config `allow_delete = true`;
  a non-empty directory additionally needs `recursive: true` **and**
  `allow_recursive_delete = true`; `dry_run: true` returns the resolved
  absolute target + `entries`/`is_dir` with no mutation. Default action
  is move-to-`.butter-trash/` (recoverable); hard delete only when
  `allow_recursive_delete` is set. Unconditionally refuses the fs root /
  shallow system paths, `$HOME` or any ancestor, the cwd or any ancestor,
  and the trash dir itself — no flag re-enables these.

## 6. Interactions

- **Operator config:** `delete` reads `context.config['allow_delete']` /
  `['allow_recursive_delete']` (strict bool `True`). Operator-declared in
  the `[[plugin]]` `config` table, surfaced read-only by core, invisible
  to the model — see `plugin-config-injection.md`.
- **Gates:** the planner should emit `confirm` for `write_file`/
  `edit_file` and **`human`** for `delete` so the executor displays the
  resolved absolute target (and `dry_run` preview, if chained) before the
  irreversible step. Gate enforcement is core's, not the plugin's.
- **Variable pool:** `find_files`/`search_content`/`stat` outputs feed
  later steps (e.g. `search_content → read_file($hit.path)`); `delete`
  chains `dry_run` → human-gated real `delete` on the same `$target`.
- **No `requires`:** filesystem calls no other plugin; `PluginContext`
  is consulted only for `config`.

## 7. Migration & Acceptance

- New plugin, no migration. Opt in via a `config.toml` `[[plugin]]`
  entry (local `path` for dev, pinned `source` + `config` table for
  production). Omitting the `config` table simply leaves `delete`
  disabled.
- Live-REPL acceptance: with `path = "~/Workspace/butter-plugin-filesystem"`
  and `config = { allow_delete = true }`, the model can `search_content`
  then `read_file` a hit; `delete` without the flag is refused; with it,
  `delete` triggers the `human` gate showing the resolved absolute path;
  a recursive dir delete is refused without `allow_recursive_delete`; a
  trashed file is recoverable from `.butter-trash/`. A second loaded
  plugin cannot observe filesystem's `config` (isolation, asserted in the
  core suite).
