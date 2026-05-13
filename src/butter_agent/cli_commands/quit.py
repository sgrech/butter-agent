"""`/quit` — equivalent to Ctrl-D, halts the REPL cleanly."""

from __future__ import annotations

from butter_agent.core.repl import CommandResult, InputSource, Output


class QuitCommand:
    """Set `exit=True` on `CommandResult` so the REPL stops accepting input."""

    name = 'quit'
    description = 'Exit the REPL (same as Ctrl-D).'

    async def run(self, args: str, io_in: InputSource, output: Output) -> CommandResult:
        del args, io_in, output
        return CommandResult(exit=True)
