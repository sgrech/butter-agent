"""`/status` — print the resolved config plus the registered-plugin count."""

from __future__ import annotations

from butter_agent.core.config import Config
from butter_agent.core.repl import CommandResult, InputSource, Output


class StatusCommand:
    """Show the resolved config used by the running REPL.

    Reads from the frozen `Config` snapshot captured at startup —
    `/configure` writes changes to disk but does not hot-reload, so a
    running session's `/status` always reflects what is actually in use.
    """

    name = 'status'
    description = 'Show the resolved runtime config (model, host, storage, plugin count).'

    def __init__(self, *, config: Config, plugin_count: int) -> None:
        self._config = config
        self._plugin_count = plugin_count

    async def run(self, args: str, io_in: InputSource, output: Output) -> CommandResult:
        del args, io_in
        cfg = self._config
        output.write('butter-agent status:\n')
        output.write(f'  model.provider         = {cfg.model.provider}\n')
        output.write(f'  model.model            = {cfg.model.model}\n')
        output.write(f'  model.host             = {cfg.model.host}\n')
        output.write(f'  storage.provider       = {cfg.storage.provider}\n')
        output.write(f'  storage.path           = {cfg.storage.path}\n')
        output.write(f'  core.max_blast_radius  = {cfg.core.max_blast_radius.value}\n')
        output.write(f'  plugins.registered     = {self._plugin_count}\n')
        return CommandResult()
