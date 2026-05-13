# butter-agent

> "I pass butter."

Local-first, open-source conversational agent runtime. Operational on first install — Ollama + Qwen3 8B, zero config. Not a coding agent, not a framework, not a cloud service.

## Design

Full architecture and scope live in the `ai-butterbot` domain of `knowledgebase-mcp` (source key `butter-agent-scope-2026-05-12`). See `CLAUDE.md` for the seven non-negotiable invariants and the build order.

## Status

Day-1 in progress. Build order from `CLAUDE.md`:

| # | Module | Status |
|---|--------|--------|
| 1 | `core/loop.py` — shape-fixed agent loop, Protocol seams | ✅ implemented |
| 2 | `core/registry.py` — plugin manifest parsing, frozen registry, blast-radius gating | ✅ implemented |
| 3 | `core/task_executor.py` — atomic plan validation, `$variable` resolution, gate enforcement | ⏳ next |
| 4 | `core/context_manager.py` — small-context-footprint enforcer | ⏳ deferred |
| 5 | `repl.py` — first-class REPL adapter | ⏳ deferred |
| 6 | `config.toml` — default config and plugin source list | ⏳ deferred |

No model adapter, no plugins, no entrypoint script wired yet.

## Development

```bash
uv sync
just check       # ruff + mypy + pytest
just fix         # auto-fix lint and format
```
