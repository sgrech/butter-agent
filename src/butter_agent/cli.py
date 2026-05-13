"""`butter` CLI entrypoint — argparse, no runtime dependencies.

Two subcommands today:

- `butter start` (default) — load config, compose the REPL, run it.
- `butter configure` — placeholder for the interactive flow that lands
  in PR B (slash-command dispatcher + `/configure`). Today it points the
  user at the config file path and exits cleanly.

The CLI is intentionally thin: composition lives in `app.build_repl`, so
this module only handles argument parsing, top-level error rendering,
deterministic resource cleanup, and the `asyncio.run` boundary.
"""

from __future__ import annotations

import argparse
import asyncio
import sqlite3
import sys
from collections.abc import Sequence
from pathlib import Path

from butter_agent import __version__
from butter_agent.app import build_repl, load_or_default_config, resolve_config_path
from butter_agent.core.config import ConfigError
from butter_agent.core.plugin_source import PluginLoadError


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level argument parser with `start` / `configure` subcommands."""
    parser = argparse.ArgumentParser(prog='butter', description='butter-agent — local-first conversational agent runtime.')
    parser.add_argument('--version', action='version', version=f'butter-agent {__version__}')

    subparsers = parser.add_subparsers(dest='command', metavar='{start,configure}')

    start = subparsers.add_parser('start', help='Start the interactive REPL (default).')
    start.add_argument('--config', '-c', type=Path, default=None, help='Path to config.toml (overrides the XDG-resolved default).')

    subparsers.add_parser('configure', help='Configure butter-agent (interactive flow lands in a follow-up; today this prints the config path).')
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint. Returns a process exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)
    command = args.command or 'start'

    if command == 'configure':
        return _cmd_configure()
    config_override: Path | None = getattr(args, 'config', None)
    return _cmd_start(config_override)


def _cmd_start(config_path: Path | None) -> int:
    target = config_path if config_path is not None else resolve_config_path()
    try:
        config = load_or_default_config(target)
    except ConfigError as exc:
        sys.stderr.write(f'[error] config: {exc}\n')
        return 2

    async def _run() -> None:
        app = await build_repl(config, config_path=target)
        try:
            await app.repl.run()
        finally:
            await app.close()

    try:
        asyncio.run(_run())
    except PluginLoadError as exc:
        # Declared in config but couldn't be resolved/imported. The user
        # wrote the [[plugin]] entry, so this is their config to fix.
        sys.stderr.write(f'[error] plugin: {exc}\n')
        return 2
    except (OSError, sqlite3.Error) as exc:
        # First-run install paths can hit PermissionError on mkdir, ENOSPC on
        # the SQLite write, or a corrupted DB at `storage.path`. Surface those
        # through the same friendly channel as ConfigError instead of dumping
        # a raw traceback.
        sys.stderr.write(f'[error] startup: {exc}\n')
        return 2
    return 0


def _cmd_configure() -> int:
    path = resolve_config_path()
    sys.stdout.write(
        f'Run `butter start` and use the `/configure` slash command to edit settings interactively.\nConfig file: {path}\n',
    )
    return 0


if __name__ == '__main__':  # pragma: no cover - module exec path
    raise SystemExit(main())
