# butter-agent

> "I pass butter."

Local-first, open-source conversational agent runtime. Operational on first install — Ollama + Qwen3 8B, zero config. Not a coding agent, not a framework, not a cloud service.

## Design

Full architecture and scope live in the `ai-butterbot` domain of `knowledgebase-mcp` (source key `butter-agent-scope-2026-05-12`). See `CLAUDE.md` for the seven non-negotiable invariants and the build order.

## Status

Day-1 scaffolding. Nothing implemented yet.

## Development

```bash
uv sync
just check       # ruff + mypy + pytest
just fix         # auto-fix lint and format
```
