"""`/configure` — interactive walk-through that writes `config.toml`.

The walkthrough prompts for the model, storage, and core fields that
matter for a first-run install. Existing values are shown as defaults
and reused on blank input. Plugin sources are preserved verbatim from
the in-memory snapshot — the interactive flow does not manage them.

Writing is to disk only — the running session keeps the snapshot it was
constructed with. The REPL prints a "restart `butter` to apply" notice
so the operator knows their edits are persisted but not yet live, which
preserves invariant #1's "runtime never reshapes" guarantee.
"""

from __future__ import annotations

import math
from pathlib import Path

from butter_agent.core.config import Config, ConfigError, CoreConfig, ModelConfig, StorageConfig, dump_config
from butter_agent.core.registry import BlastRadius
from butter_agent.core.repl import CommandResult, InputSource, Output


class ConfigureCommand:
    """Run the interactive walkthrough and write the result to disk."""

    name = 'configure'
    description = 'Interactively update config.toml (requires restart to apply).'

    def __init__(self, *, config: Config, config_path: Path) -> None:
        self._config = config
        self._path = config_path

    async def run(self, args: str, io_in: InputSource, output: Output) -> CommandResult:
        del args
        output.write(f'Editing {self._path}. Press Enter to keep the current value.\n')

        try:
            provider = await _prompt(io_in, output, 'model.provider', self._config.model.provider)
            model = await _prompt(io_in, output, 'model.model', self._config.model.model)
            host = await _prompt(io_in, output, 'model.host', self._config.model.host)
            timeout = await _prompt_positive_float(io_in, output, 'model.timeout_seconds', self._config.model.timeout_seconds)
            storage_path = await _prompt(io_in, output, 'storage.path', self._config.storage.path)
            radius = await _prompt_radius(io_in, output, self._config.core.max_blast_radius)
        except EOFError:
            output.write('\n[abort] /configure cancelled — config not written.\n')
            return CommandResult()

        new_config = Config(
            core=CoreConfig(max_blast_radius=radius, network_allowlist=self._config.core.network_allowlist),
            model=ModelConfig(provider=provider, model=model, host=host, timeout_seconds=timeout),
            storage=StorageConfig(provider=self._config.storage.provider, path=storage_path),
            plugins=self._config.plugins,
        )

        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._path.write_text(dump_config(new_config))
        except OSError as exc:
            output.write(f'[error] could not write {self._path}: {exc}\n')
            return CommandResult()

        output.write(f'Wrote {self._path}. Restart `butter` to apply.\n')
        return CommandResult()


async def _prompt(io_in: InputSource, output: Output, label: str, current: str) -> str:
    """Prompt for `label`; return the user's input or `current` on blank."""
    del output  # the prompt itself is rendered via the InputSource
    answer = await io_in.read_line(f'  {label} [{current}]: ')
    stripped = answer.strip()
    return stripped if stripped else current


async def _prompt_positive_float(io_in: InputSource, output: Output, label: str, current: float) -> float:
    """Prompt for a positive finite float; re-prompt on invalid input."""
    while True:
        raw = await _prompt(io_in, output, label, str(current))
        try:
            value = float(raw)
        except ValueError:
            output.write(f'  [invalid] {raw!r} is not a number\n')
            continue
        # `nan`/`inf` parse successfully via float() but make no sense as
        # a timeout — and NaN slips past the positivity check because every
        # comparison with NaN is False.
        if not math.isfinite(value):
            output.write(f'  [invalid] {label} must be a finite number, got {value}\n')
            continue
        if value <= 0:
            output.write(f'  [invalid] {label} must be positive, got {value}\n')
            continue
        return value


async def _prompt_radius(io_in: InputSource, output: Output, current: BlastRadius) -> BlastRadius:
    """Prompt for `core.max_blast_radius` and re-prompt on invalid input."""
    valid = ', '.join(r.value for r in BlastRadius)
    while True:
        raw = await _prompt(io_in, output, f'core.max_blast_radius (one of: {valid})', current.value)
        try:
            return BlastRadius(raw)
        except ValueError:
            output.write(f'  [invalid] {raw!r} is not one of: {valid}\n')


# Re-exported so callers that want to write a Config without invoking the
# interactive flow (tests, future `butter configure` non-interactive mode)
# do not need to reach into `core.config` directly.
__all__ = ['ConfigError', 'ConfigureCommand', 'dump_config']
