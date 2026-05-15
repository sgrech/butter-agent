# butter-agent

Local-first open-source conversational agent runtime. "I pass butter."

Full design lives in `knowledgebase-mcp` domain **`ai-butterbot`** (source key `butter-agent-scope-2026-05-12`). Search there before any new spec or implementation work.

## Architecture invariants (non-negotiable)

1. Core loop never changes shape at runtime
2. Plugin registry immutable after startup
3. Task plans validated atomically before any execution begins
4. Variable resolution is dict lookup — never model inference
5. Gate enforcement lives in core (task executor), not model, not plugin
6. Plugins cannot read each other's internal state — only declared outputs via core variable pool
7. Blast radius can only be restricted at core config level — never expanded by a plugin

## Build order

1. `core/loop.py` — agent loop (~200 lines)
2. `core/registry.py` — plugin loader, manifest validation, frozen registry
3. `core/task_executor.py` — plan validation, step execution, gate handling, `$variable` resolution
4. `core/context_manager.py` — small-context-footprint enforcer
5. `repl.py` — REPL interface (first-class, not dev-tool)
6. `config.toml` — default config, plugin sources declared here

## Implementation Guidelines

All new code must follow the patterns documented in `specs/guidelines/`. Consult the relevant spec before writing new models, services, endpoints, or tests.

| Spec | When to consult |
|------|----------------|
| `specs/development/plugin-config-injection.md` | Before reading operator settings in a plugin, adding a `[[plugin]].config` key, or touching `PluginContext.config` / config parse + dump |
| `specs/development/filesystem-plugin.md` | Before changing the filesystem plugin's capability surface, path/cwd model, or `delete` safety layering (the reference operator-config gate) |
| `specs/development/capability-discovery.md` | Before changing how capabilities are surfaced to the model — `CapabilityFilter`, `context_manager` selection, or adding a discovery/loop phase |

Development specs (roadmaps for WHAT to build) live in `specs/development/`. Completed or superseded specs are moved to `specs/archive/`.
