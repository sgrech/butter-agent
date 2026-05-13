"""`butter` CLI entrypoint — argparse, no runtime dependencies.

Two subcommands today:

- `butter start` (default) — load config, compose the REPL, run it.
- `butter configure` — placeholder for the interactive flow that lands
  in PR B (slash-command dispatcher + `/configure`). Today it points the
  user at the config file path and exits cleanly.

The CLI is intentionally thin: composition lives in `app.build_repl`, so
this module only handles argument parsing, top-level error rendering,
and the `asyncio.run` boundary.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from collections.abc import Sequence

from butter_agent import __version__
from butter_agent.app import build_repl, load_or_default_config, resolve_config_path
from butter_agent.core.config import ConfigError


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level argument parser with `start` / `configure` subcommands."""
    parser = argparse.ArgumentParser(prog='butter', description='butter-agent — local-first conversational agent runtime.')
    parser.add_argument('--version', action='version', version=f'butter-agent {__version__}')

    subparsers = parser.add_subparsers(dest='command', metavar='{start,configure}')
    subparsers.add_parser('start', help='Start the interactive REPL (default).')
    subparsers.add_parser('configure', help='Configure butter-agent (interactive flow lands in a follow-up; today this prints the config path).')
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint. Returns a process exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)
    command = args.command or 'start'

    if command == 'configure':
        return _cmd_configure()
    return _cmd_start()


def _cmd_start() -> int:
    try:
        config = load_or_default_config(resolve_config_path())
    except ConfigError as exc:
        sys.stderr.write(f'[error] config: {exc}\n')
        return 2

    async def _run() -> None:
        repl = await build_repl(config)
        await repl.run()

    asyncio.run(_run())
    return 0


def _cmd_configure() -> int:
    path = resolve_config_path()
    sys.stdout.write(
        f'Interactive configuration lands with the `/configure` slash command in a follow-up.\nFor now, edit this file directly (it is created on first save):\n  {path}\n',
    )
    return 0


if __name__ == '__main__':  # pragma: no cover - module exec path
    raise SystemExit(main())
