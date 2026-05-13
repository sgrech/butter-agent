"""Slash commands exposed by the `Repl`.

This package owns the adapter-side commands (`/help`, `/quit`,
`/status`, `/configure`). The dispatcher itself lives in
`core/repl.py` — these modules supply the `Command` implementations and
`build_default_commands()` wires them up at startup.

Slash commands belong to the REPL adapter, not core. They never mutate
runtime state — `/configure` writes config to disk and asks the user
to restart, preserving the immutable-runtime invariant.
"""

from __future__ import annotations

from pathlib import Path

from butter_agent.cli_commands.configure import ConfigureCommand
from butter_agent.cli_commands.help import HelpCommand
from butter_agent.cli_commands.quit import QuitCommand
from butter_agent.cli_commands.status import StatusCommand
from butter_agent.core.config import Config
from butter_agent.core.repl import Command, CommandRegistry


def build_default_commands(
    *,
    config: Config,
    config_path: Path,
    plugin_count: int,
) -> CommandRegistry:
    """Construct the default `CommandRegistry` shipped with the REPL.

    `/help` needs the full command list, so it is built last and given a
    callable that closes over the assembled tuple — this keeps the
    registry frozen without exposing a back-edge for mutation.
    """
    others: list[Command] = [
        QuitCommand(),
        StatusCommand(config=config, plugin_count=plugin_count),
        ConfigureCommand(config=config, config_path=config_path),
    ]
    help_command = HelpCommand(lambda: (*others, help_command))
    return CommandRegistry((help_command, *others))


__all__ = [
    'ConfigureCommand',
    'HelpCommand',
    'QuitCommand',
    'StatusCommand',
    'build_default_commands',
]
