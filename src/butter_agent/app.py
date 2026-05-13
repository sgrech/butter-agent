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
from dataclasses import dataclass
from pathlib import Path

from butter_agent.cli_commands import build_default_commands
from butter_agent.core.config import Config, load_config
from butter_agent.core.context_manager import DefaultContextManager
from butter_agent.core.loop import AgentLoop
from butter_agent.core.plugin_source import PluginLoader, PluginLoadError
from butter_agent.core.registry import RegistryBuilder, RegistryError
from butter_agent.core.repl import InputSource, Output, Repl, ReplGateHandler, StdioInputSource, StdioOutput
from butter_agent.core.task_executor import DefaultTaskExecutor
from butter_agent.model.ollama import OllamaModelClient
from butter_agent.storage.sqlite import Database, SqliteConversationHistory

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
) -> App:
    """Compose a runnable `App` (Repl + database handle) from a validated `Config`.

    Empty registry by design (v1) — `[[plugin]]` source-fetch lands in a
    later scope. The model still receives an (empty) capabilities list,
    which PR C's system-prompt update teaches it to handle honestly.

    `input_source` / `output` default to stdio; tests inject stubs so the
    full wiring can be exercised without touching the terminal.
    """
    io_in = input_source if input_source is not None else StdioInputSource()
    io_out = output if output is not None else StdioOutput()

    database = await Database.open(config.storage.path)
    history = await SqliteConversationHistory.create(database)
    loader = plugin_loader if plugin_loader is not None else PluginLoader()
    loaded = loader.load_all(config.plugins)
    builder = RegistryBuilder(max_blast_radius=config.core.max_blast_radius)
    # RegistryBuilder enforces invariants (blast-radius ceiling, unique
    # names). Re-raise its errors as PluginLoadError so the CLI surfaces
    # them through the same friendly `[error] plugin: ...` path instead of
    # dumping a raw traceback.
    try:
        for entry in loaded:
            builder.register(entry.manifest, entry.plugin)
        registry = builder.build()
    except RegistryError as exc:
        raise PluginLoadError(f'registry rejected plugin: {exc}') from exc

    context_manager = DefaultContextManager(registry, history)
    model = OllamaModelClient(
        host=config.model.host,
        model=config.model.model,
        timeout_seconds=config.model.timeout_seconds,
    )
    gate_handler = ReplGateHandler(io_in, io_out)
    executor = DefaultTaskExecutor(registry, gate_handler)
    loop = AgentLoop(context_manager, model, executor, history=history)

    commands = build_default_commands(
        config=config,
        config_path=config_path if config_path is not None else resolve_config_path(),
        plugin_count=len(registry),
    )
    repl = Repl(loop, io_in, io_out, banner=_format_banner(config), commands=commands)
    return App(repl=repl, database=database)


def _format_banner(config: Config) -> str:
    return f'butter-agent. model={config.model.model} host={config.model.host} storage={config.storage.path}\n'
