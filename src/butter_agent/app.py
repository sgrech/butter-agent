"""Application composition — wire core seams into a runnable `Repl`.

`build_repl(config)` is the single composition point. It is pure async,
takes a validated `Config`, and returns a fully wired `Repl` ready for
`.run()`. The CLI calls this; tests call this; future adapters (Telegram,
web) will replace it with their own composition while reusing the same
core seams.

Config resolution lives here too: `resolve_config_path()` returns the
XDG-style location (`~/.config/butter-agent/config.toml`, or
`$XDG_CONFIG_HOME/butter-agent/config.toml` when set), and
`load_or_default_config()` reads it if present or falls back to the
documented in-code defaults — an empty TOML document is valid by design
(see `core/config.py`).
"""

from __future__ import annotations

import os
import sys
from collections.abc import Callable
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass
from pathlib import Path

from butter_agent.cli_commands import build_default_commands
from butter_agent.core.config import Config, load_config
from butter_agent.core.context_manager import DefaultContextManager, InMemoryConversationHistory
from butter_agent.core.loop import AgentLoop, ModelClient
from butter_agent.core.plugin_source import PluginLoader, PluginLoadError
from butter_agent.core.registry import RegistryBuilder, RegistryError
from butter_agent.core.repl import InputSource, Output, Repl, ReplGateHandler, StdioInputSource, StdioOutput
from butter_agent.core.task_executor import DefaultTaskExecutor
from butter_agent.model.ollama import OllamaModelClient
from butter_agent.plugins.database import build_database_plugin
from butter_agent.repl_prompt_toolkit import InferenceIndicator, PromptToolkitInputSource
from butter_agent.storage.sqlite import Database

# --- Config path resolution --------------------------------------------------


def resolve_config_path() -> Path:
    """Return the XDG-style config file location.

    Honours `$XDG_CONFIG_HOME` when set; otherwise defaults to
    `~/.config/butter-agent/config.toml`.
    """
    # Per the XDG Base Directory spec, an empty value is treated as unset.
    xdg = os.environ.get('XDG_CONFIG_HOME') or None
    base = Path(xdg).expanduser() if xdg else Path.home() / '.config'
    return base / 'butter-agent' / 'config.toml'


def load_or_default_config(path: Path | None = None) -> Config:
    """Load `config.toml` from `path` if it exists, else return defaults.

    A missing file is not an error — the first-run experience is "no
    config needed, sensible defaults apply". Anything else (malformed
    TOML, type mismatch) bubbles up as `ConfigError` from `load_config`.
    """
    target = path if path is not None else resolve_config_path()
    text = target.read_text() if target.is_file() else ''
    return load_config(text)


# --- Composition -------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class App:
    """Composed application handle — a `Repl` plus the lifecycle resources it owns.

    Returned from `build_repl` so callers can close the underlying SQLite
    connection deterministically after the REPL exits rather than relying
    on interpreter-shutdown GC ordering.
    """

    repl: Repl
    database: Database

    async def close(self) -> None:
        """Idempotently close owned resources."""
        await self.database.close()


async def build_repl(
    config: Config,
    *,
    input_source: InputSource | None = None,
    output: Output | None = None,
    config_path: Path | None = None,
    plugin_loader: PluginLoader | None = None,
    model: ModelClient | None = None,
) -> App:
    """Compose a runnable `App` (Repl + database handle) from a validated `Config`.

    The registry is built with built-ins first: the shared `database`
    infrastructure plugin is registered before any external `[[plugin]]`
    declaration, so a third-party plugin cannot shadow it and plugins
    that `require` it resolve at build. Opinionated, planner-visible
    capabilities (notes, reminders, search) are NOT bundled — they are
    standalone repos the operator opts into via `config.toml`
    (`PluginLoader`); see `specs/archive/plugin-externalization.md`.
    With no `[[plugin]]` declared, only `database` (all-internal) is
    registered and the model receives an empty user-facing capability
    list, which the system prompt teaches it to handle honestly.

    `input_source` / `output` default to stdio; tests inject stubs so the
    full wiring can be exercised without touching the terminal. Passing
    `model` injects a scripted/recording `ModelClient` for end-to-end
    scenario tests (`tests/integration/test_scenarios.py`) — every
    other seam (registry, executor, gates, plugins, REPL) stays real so
    the test exercises the same code path the live REPL does.
    """
    io_out = output if output is not None else StdioOutput()

    # `Database` is opened so the storage path is created/validated and a
    # connection handle is ready for future consumers (plugin state, etc.).
    # Conversation history is deliberately in-memory: every butter
    # invocation starts a fresh chat session. A persisted-history design
    # paired with explicit sessions is future work — see scope notes.
    database = await Database.open(config.storage.path)
    history = InMemoryConversationHistory()
    loader = plugin_loader if plugin_loader is not None else PluginLoader()
    loaded = loader.load_all(config.plugins)
    builder = RegistryBuilder(max_blast_radius=config.core.max_blast_radius)
    # RegistryBuilder enforces invariants (blast-radius ceiling, unique
    # names). Re-raise its errors as PluginLoadError so the CLI surfaces
    # them through the same friendly `[error] plugin: ...` path instead of
    # dumping a raw traceback.
    try:
        # Built-in infrastructure first: the shared `database` store wraps
        # the connection opened above. Registering it before external
        # plugins means a third-party plugin claiming the name `database`
        # is rejected as a duplicate — core infrastructure can't be
        # shadowed (invariants #6/#7).
        db_manifest, db_plugin = build_database_plugin(database)
        builder.register(db_manifest, db_plugin)
        # External plugins after the built-in infra so their `requires`
        # (e.g. notes → database.*) resolve and they cannot shadow it.
        for entry in loaded:
            builder.register(entry.manifest, entry.plugin)
        registry = builder.build()
    except RegistryError as exc:
        raise PluginLoadError(f'registry rejected plugin: {exc}') from exc

    commands = build_default_commands(
        config=config,
        config_path=config_path if config_path is not None else resolve_config_path(),
        plugin_count=len(registry),
    )
    # IO selection happens after `commands` are built so the
    # prompt_toolkit completer can be seeded with the registered slash
    # names. Caller-supplied `input_source` always wins (tests). When
    # stdin is a TTY, promote to prompt_toolkit + spinner; otherwise
    # stick with stdio so piped / automation runs stay log-friendly.
    io_in: InputSource
    indicator_factory: Callable[[], AbstractAsyncContextManager[object]] | None = None
    # Banner is redundant when the bottom toolbar shows the same info,
    # so suppress it for interactive runs. Piped / non-TTY callers keep
    # the banner so they can see model/host/storage in the log.
    banner: str
    if input_source is not None:
        io_in = input_source
        banner = _format_banner(config)
    elif sys.stdin.isatty():
        io_in = PromptToolkitInputSource(
            history_path=_history_path(),
            bottom_toolbar_text=_format_toolbar(config),
            slash_commands=tuple(cmd.name for cmd in commands.all()),
        )
        indicator_factory = InferenceIndicator
        banner = ''
    else:
        io_in = StdioInputSource()
        banner = _format_banner(config)

    context_manager = DefaultContextManager(registry, history)
    model_client: ModelClient = (
        model
        if model is not None
        else OllamaModelClient(
            host=config.model.host,
            model=config.model.model,
            timeout_seconds=config.model.timeout_seconds,
            think=config.model.think,
        )
    )
    gate_handler = ReplGateHandler(io_in, io_out)
    executor = DefaultTaskExecutor(registry, gate_handler)
    loop = AgentLoop(context_manager, model_client, executor, history=history)

    repl = Repl(
        loop,
        io_in,
        io_out,
        banner=banner,
        commands=commands,
        indicator_factory=indicator_factory,
    )
    return App(repl=repl, database=database)


def _format_banner(config: Config) -> str:
    return f'butter-agent. model={config.model.model} host={config.model.host} storage={config.storage.path}\n'


def _format_toolbar(config: Config) -> str:
    """Status line shown in the prompt_toolkit bottom toolbar.

    Short enough to fit one terminal row at typical widths. Surfaces
    the active model/host so the user can verify what they're talking
    to without re-reading the banner.
    """
    return f' butter-agent · {config.model.model} @ {config.model.host} · Ctrl-D to quit '


def _history_path() -> Path:
    """Persistent input history location, mirroring the storage default.

    `~/.butter-agent/history` sits beside the SQLite database so the
    whole runtime state lives under one user-owned directory. The
    parent dir is created eagerly so prompt_toolkit's `FileHistory`
    can write without surprising the user on first run.
    """
    path = Path.home() / '.butter-agent' / 'history'
    path.parent.mkdir(parents=True, exist_ok=True)
    return path
