# Runnable baseline — `butter` CLI + first-run UX

**Status**: planned
**Created**: 2026-05-13

## Goal

Get butter-agent from "composable in-process" (today) to "`pip install -e . && butter configure && butter start` is a real first-run experience" — usable with zero plugins, model self-aware of its capabilities.

User expectation (verbatim from session):

> A single entrypoint, the `butter` command. Once invoked, a configuration slash command to configure the butter runtime with sensible defaults (model, database path, etc). Once configured, chat with the model even with zero plugins, where the model is aware of what it is and what it can or cannot do — e.g. asking for a calendar event surfaces "no calendar plugin available". Runs on local models via Ollama, with `qwen3:8b` as the default (Ollama Cloud free tier covers users without local GPU; smaller default would mislead about what scales).

## Non-goals

- Built-in plugins (notes, reminders, search) — separate scope, lands after baseline works.
- External plugin source-fetch (cloning pinned repos) — flagged in `registry.py` as future work.
- Memory retrieval backend — `NullMemoryRetriever` is fine for v1.
- Ollama JSON schema mode upgrade — the current `format: "json"` works at the 8B default; revisit only if/when we drop the floor.

## Architecture decisions locked in

- **Config path**: `~/.config/butter-agent/config.toml` (XDG-style). Falls back to shipped repo defaults if absent.
- **CLI framework**: stdlib `argparse`. No new runtime dependencies.
- **Slash commands live in REPL**, not in core. Lines starting with `/` route through a `CommandRegistry`; everything else falls through to the model unchanged. This keeps invariant #1 (loop shape never changes) — the REPL is an adapter, free to add interaction sugar.
- **`/configure` writes to disk; runtime config stays immutable** (invariant: changes require restart). The command surfaces this explicitly — "restart `butter` to apply".
- **Empty-registry chat works**: `KeywordCapabilityFilter` already returns `()` when nothing is registered. The system prompt is the missing piece; see PR C.

## Plan — 3 PRs, in order

### PR A — `butter` CLI entrypoint + bootstrap composition

**Files**:
- `src/butter_agent/app.py` — pure async composition function `build_repl(config: Config) -> Repl`. Wires Database → SqliteConversationHistory → DefaultContextManager → OllamaModelClient → ReplGateHandler → DefaultTaskExecutor → AgentLoop → Repl. Empty registry for v1.
- `src/butter_agent/cli.py` — `argparse` entrypoint. Subcommands: `butter start` (default), `butter configure` (alias for /configure but pre-REPL).
- `pyproject.toml` — add `[project.scripts] butter = "butter_agent.cli:main"`.

**Behaviour**:
- `butter start` reads `~/.config/butter-agent/config.toml` if present, else uses in-code defaults (matches shipped `config.toml`).
- Prints a one-line banner identifying the resolved model + storage path before the REPL takes over.

**Tests**: composition function returns a wired Repl; CLI parses subcommands; missing config file falls back to defaults without error.

### PR B — Slash command dispatcher + `/configure`

**Files**:
- `src/butter_agent/core/repl.py` — add `Command` Protocol and `CommandRegistry`. `Repl.run` intercepts lines starting with `/` and dispatches.
- New `src/butter_agent/cli_commands/` package — `configure.py`, `help.py`, `quit.py`, `status.py`.

**Commands**:
- `/help` — list commands and one-line descriptions
- `/quit` — same as Ctrl-D
- `/status` — show resolved config (model, host, storage path, registered plugin count)
- `/configure` — interactive walk-through of model.provider/model/host, storage.path, core.max_blast_radius. Writes to `~/.config/butter-agent/config.toml`. Prints "Restart `butter` to apply" — does not hot-reload (immutable runtime invariant).

**Tests**: command dispatch (slash → command, non-slash → model), each command's behaviour with stub IO seams, `/configure` writes a round-trippable file (load_config reproduces what was written).

### PR C — System prompt: identity + capability honesty

**Files**:
- `src/butter_agent/model/ollama.py` — update `_SYSTEM_PROMPT`.

**New prompt content** (additive, not destructive):
- One-line identity: "You are butter-agent, a local-first personal assistant. You can chat directly and, when matching capabilities are available, plan multi-step actions using plugins."
- Explicit rule: "If the user asks for something that would require a capability and no matching capability is listed above, return a reply explaining what's missing — do not invent a plugin name or fabricate a plan."

**Tests**: render the prompt with empty capabilities, assert the no-plan rule is present. Optional integration test (skipped if Ollama unavailable) that asks "add a calendar event" against an empty registry and asserts the reply mentions the gap.

## Acceptance criteria

After all three PRs merge:

1. `pip install -e . && butter start` boots into a REPL that connects to Ollama (or fails with a clear diagnostic).
2. Asking a conversational question gets a conversational reply.
3. Asking "add a calendar event" gets a reply that names the missing capability instead of a fabricated plan.
4. Running `/configure` from inside the REPL writes a valid `config.toml` that loads cleanly on next start.
5. Full test suite green, ruff + mypy --strict clean.

## Out of scope follow-ups (do not bundle)

- First built-in plugin (`notes`) — exercises the registry path end-to-end with a real plugin
- `plugin_source` module — clone repos declared in `[[plugin]]` at startup
- Skill system (summarise/reformat/plan/converse)
- Ollama JSON schema mode upgrade (only if we lower the default model size)
