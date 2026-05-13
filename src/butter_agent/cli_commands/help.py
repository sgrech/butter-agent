"""`/help` — list registered slash commands with one-line descriptions."""

from __future__ import annotations

from collections.abc import Callable

from butter_agent.core.repl import Command, CommandResult, InputSource, Output


class HelpCommand:
    """Print the available commands and their descriptions.

    The list of peer commands is supplied lazily via a callable. This
    lets the help command be inserted into the same `CommandRegistry`
    that owns it without resorting to mutable state.
    """

    name = 'help'
    description = 'List available slash commands.'

    def __init__(self, commands_provider: Callable[[], tuple[Command, ...]]) -> None:
        self._provider = commands_provider

    async def run(self, args: str, io_in: InputSource, output: Output) -> CommandResult:
        del args, io_in
        commands = self._provider()
        if not commands:
            output.write('(no slash commands registered)\n')
            return CommandResult()
        width = max(len(c.name) for c in commands)
        output.write('Available commands:\n')
        for command in commands:
            output.write(f'  /{command.name.ljust(width)}  {command.description}\n')
        return CommandResult()
