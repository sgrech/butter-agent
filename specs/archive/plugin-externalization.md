# Plugin Externalization Spec

**Status**: completed — archived. The notes plugin was extracted to
[`butter-plugin-notes`](https://github.com/sgrech/butter-plugin-notes)
and core's bundled copy removed (commits `cd30f55`, `56ee50e`). Retained
as the historical record of the extraction procedure and the
bundled-vs-external rule; not a live roadmap.

## 1. Purpose

Establish the rule for **what ships in butter-agent core versus what lives
in a standalone plugin repo**, and the reusable procedure for moving an
opinionated capability out. The immediate application is `notes` (merged
into core in PR #22, task #377); the same procedure applies unchanged to
`reminders` and `search` when they are built.

The driving principle, decided 2026-05-15:

> **Core ships only the machinery that drives the ecosystem, plus the one
> bundled infrastructure plugin (`database`). Every opinionated,
> user-facing capability is a separate public repo the operator opts into
> via `config.toml`.**

Rationale: a user who wants their own `notes` (different schema, different
storage policy, tags, encryption, whatever) must be able to *not load*
butter-agent's and load theirs instead, with no core changes. A capability
baked into core is not replaceable; a `[[plugin]]` declaration is. This is
the "flexibility at the edges, strictness at the core" architecture
principle made literal.

## 2. Scope

### In Scope

- The bundled-vs-external decision rule (§3) and its rationale.
- The standalone plugin repo template (§4), mirroring the existing
  `butter-plugin-clock` (`git@github.com:sgrech/butter-plugin-clock.git`).
- The reusable extraction checklist (§5), reusable verbatim for
  `reminders` / `search`.
- The first application: extract `notes` into `butter-plugin-notes` and
  remove it from butter-agent core (§6).
- Documentation/state updates the extraction must make: knowledgebase
  `ai-butterbot`, memory-mcp `butter-agent`, and the stale
  `app.build_repl` docstring.

### Out of Scope

- **`database` stays a core built-in.** It is not an opinionated
  capability — it is the invariant-#6 namespace-isolation enforcement
  point, the single backup target, and the `requires` substrate every
  write-plugin depends on. It is the *one* bundled infrastructure plugin,
  registered before any external plugin so a third-party repo cannot
  shadow it.
- External→external `requires` resolution and "which loaded plugin
  provides a shared internal capability" — not needed while the only
  internal-capability provider (`database`) is a core built-in. Revisit
  only if a second infrastructure plugin is ever externalized (it should
  not be, per §3).
- The actual extraction of `reminders` / `search` — those plugins do not
  exist yet. This spec is the procedure they will follow when they do.
- Dependency installation for external plugins — `core/plugin_source.py`
  already documents this as deliberately unsupported in v1 (stdlib-only
  plugins, or packages already in butter's environment).

## 3. The Bundled-vs-External Rule

A plugin is a **core built-in** if and only if it is *infrastructure the
ecosystem depends on to function*. Today that set is exactly one:
`database`. Properties that put it there:

- All capabilities are `internal: true` — invisible to the planner,
  callable only plugin-to-plugin.
- It is the enforcement point for a core invariant (#6: caller-namespaced
  tables; the prefix is applied in core's `_PluginContext`, not the
  plugin).
- Other plugins name it in `requires`; it has no `requires` of its own.
- Removing it would mean every write-plugin re-implements connection
  lifecycle and there is no single backup target.

Every other plugin is **external**. Heuristics that mean "external":

- It is planner-visible (`internal: false`) — i.e. it is a *capability a
  user asks for*, not infrastructure.
- It encodes an opinion a user might reasonably want to replace (schema,
  storage policy, output format, external service choice).
- Core runs correctly with it absent (empty registry is valid by design).

`notes`, `reminders`, `search` are all external by this rule. `clock`
already is (`butter-plugin-clock`).

## 4. Standalone Plugin Repo Template

Mirror `butter-plugin-clock` exactly. Required layout:

```
butter-plugin-<name>/
  manifest.toml          # parsed by core/registry.parse_manifest
  pyproject.toml          # UV project, src layout
  uv.lock
  justfile                # check / fix / test recipes
  CLAUDE.md               # plugin-local instructions
  README.md               # what it does, config snippet to load it
  LICENSE
  src/<package>/…         # the Plugin implementation
  tests/…                 # the plugin's own unit tests
```

`manifest.toml` is the source of truth core reads (built-ins build the
manifest in code; external plugins ship it as a file). It MUST declare:

- `[plugin]` `name`, `version`, `blast_radius`, `entrypoint`
  (`module:Class`, resolved by `_import_entrypoint` against the repo's
  `src/`).
- `requires = [...]` for any internal capability it calls
  (`database.define_table`, etc. for a write-plugin).
- `[[capability]]` blocks with `name`, `description`, `input_schema`,
  `output_schema` (and `internal` only if infrastructure — external
  plugins are planner-visible, so omit it).

Constraints inherited from core (do not re-litigate in the plugin):

- The top-level package name must be unique across all loaded plugins —
  `_import_entrypoint` rejects a module that resolves outside its plugin
  root (sys.modules collision guard).
- `execute` must be `async def execute(self, capability, inputs, context)`
  — three params after `self`; the loader fails fast otherwise.
- Plugins are stdlib-only (v1) unless the dependency is already in
  butter's environment.
- The plugin never sees the `database` namespace prefix: it passes the
  **bare** table name; core rewrites it to `<plugin-name>__<table>` using
  the owner identity from the *loaded manifest's* `name`. Externalizing
  does not change this — the owner identity is the manifest name whether
  the plugin is built-in or loaded.

## 5. Extraction Checklist (reusable)

For any built-in `X` being externalized into `butter-plugin-X`:

1. **Create the repo** from the §4 template. Copy `src/.../X.py` →
   `src/butter_plugin_X/plugin.py` (unique package name), adapting the
   built-in's in-code manifest into `manifest.toml`. Move the built-in's
   isolated unit tests into the new repo's `tests/`.
2. **Pin a release**: tag `v0.1.0`; production config uses `repo@v0.1.0`,
   never `@main` (pinned-ref invariant).
3. **Remove from core**: delete `src/butter_agent/plugins/X.py`, its
   `build_X_plugin` import and `builder.register(...)` call in
   `app.build_repl`, and `tests/test_X_plugin.py`.
4. **Keep the core integration test** but assert via an in-repo recording
   stub (the `clock` pattern in `tests/integration/test_scenarios.py`):
   the scenario must still prove the chain/gate end-to-end without
   depending on the external repo being checked out.
5. **Config wiring**: external plugins are *always* opt-in — there is no
   default-on path. The shipped `config.toml` has no active `[[plugin]]`
   entries and `Config.plugins` defaults to `()`; loading `X` means the
   operator adds a `[[plugin]]` block. Document that block in the new
   repo's README — `PluginPath` (`path = "~/Workspace/butter-plugin-X"`)
   for local dev, `PluginSource` (`repo`/`ref`) for production — and
   update the commented example in butter-agent's `config.toml` to point
   at the real repo.
6. **Fix the stale docstring**: `app.build_repl` still says "Empty
   registry by design (v1)" — update it to state that built-ins
   (`database`) register first, then external `[[plugin]]` declarations.
7. **Update knowledge state**: knowledgebase `ai-butterbot` (the "Plugin
   System Design" chunk currently says "Built-ins: notes, reminders,
   search … bundled not repo-loaded" — now wrong: only `database` is
   bundled). Add a memory-mcp `butter-agent` entry recording the
   reversal. Update the relevant development spec(s).
8. **Verify**: `just check` green in both repos; live-REPL the capability
   loaded via `PluginPath` to confirm the requires/gate/namespace path
   works identically to the built-in.

## 6. First Application: `notes`

- New repo `butter-plugin-notes` (`git@github.com:sgrech/butter-plugin-notes.git`),
  layout per §4. `src/butter_plugin_notes/plugin.py` is the current
  `NotesPlugin` verbatim; `manifest.toml` encodes what `build_notes_plugin`
  builds in code today (`blast_radius = "local-write"`,
  `requires = ["database.define_table", "database.insert",
  "database.select"]`, capabilities `create`/`list`/`read`,
  planner-visible).
- Remove `src/butter_agent/plugins/notes.py`, the `build_notes_plugin`
  wiring in `app.py`, and `tests/test_notes_plugin.py` from core.
- `tests/integration/test_notes_scenario.py` stays in butter-agent but is
  reworked to register an in-repo recording notes stub alongside the real
  `database` plugin + recording clock — same self-contained pattern as
  the existing clock scenario. The *real* notes plugin's unit tests live
  in `butter-plugin-notes`.
- No on/off-by-default question exists: external plugins are opt-in by
  `[[plugin]]` declaration and the shipped config declares none. Post
  extraction, `notes` is simply absent until the operator adds the block;
  the only config touch is updating the commented example to reference
  `butter-plugin-notes@v0.1.0`.

## 7. Migration Strategy

- Extraction is additive (new repo) then subtractive (core removal) — no
  data migration: `notes` persists via the unchanged built-in `database`,
  so an existing `notes__entries` table is read by the externalized
  plugin identically (owner identity = manifest name = `notes`,
  unchanged).
- No feature flag. The migration *is* the config change: stop bundling,
  start declaring. An operator who had notes before adds one `[[plugin]]`
  block to keep it.
- Sequencing: this spec lands first (planning of record). The notes
  extraction is its own PR pair (new repo + core-removal PR) executed
  against this checklist. `reminders` / `search`, when specced, reference
  §4–§5 instead of redefining them.

## 8. References

- `specs/development/notes-plugin.md` — the capability being externalized
  (its §4 slice-3 persistence constraints still hold; they are a
  `database` contract, not a notes-location concern).
- `specs/development/database-plugin.md` — why `database` is the one
  bundled infra plugin (§6 namespace isolation, invariant #6).
- `src/butter_agent/core/plugin_source.py` — the already-implemented
  loader (`PluginLoader`, `GitFetcher`, `PluginPath`); externalization is
  unblocked, not new infrastructure.
- `butter-plugin-clock` — the reference repo this template mirrors.
- memory-mcp `butter-agent`: 1427 (plugin/persistence model), 1432
  (notes implemented/merged), and the externalization-decision entry the
  extraction will add.
