"""Tests for slash commands and the REPL dispatcher.

Covers:

- `Repl` dispatch: slash → command, non-slash → model, unknown command
  prints an error, `/quit` exits cleanly.
- Each built-in command's behaviour against stub IO seams.
- `/configure` writes a round-trippable file (`load_config(dump_config(...))`
  reproduces the in-memory `Config`).
- `dump_config` round-trip for the documented defaults.
"""

from __future__ import annotations

import io
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from butter_agent.cli_commands import build_default_commands
from butter_agent.cli_commands.configure import ConfigureCommand
from butter_agent.cli_commands.help import HelpCommand
from butter_agent.cli_commands.quit import QuitCommand
from butter_agent.cli_commands.status import StatusCommand
from butter_agent.core.config import Config, CoreConfig, ModelConfig, PluginSource, StorageConfig, dump_config, load_config
from butter_agent.core.loop import AgentLoop, ExecutionResult, ModelContext, ModelOutput, ModelReply, TaskPlan, Turn, TurnResult
from butter_agent.core.registry import BlastRadius
from butter_agent.core.repl import CommandRegistry, CommandResult, Repl

# --- IO stubs ---------------------------------------------------------------


class _ScriptedInput:
    """Yields canned lines, raising `EOFError` once exhausted."""

    def __init__(self, lines: list[str]) -> None:
        self._lines: deque[str] = deque(lines)

    async def read_line(self, prompt: str) -> str:
        del prompt
        if not self._lines:
            raise EOFError
        return self._lines.popleft()


class _Recording:
    """Collects writes for assertion."""

    def __init__(self) -> None:
        self.buffer = io.StringIO()

    def write(self, text: str) -> None:
        self.buffer.write(text)


# --- dump_config / round-trip ----------------------------------------------


def test_dump_config_roundtrips_defaults() -> None:
    original = load_config('')
    assert load_config(dump_config(original)) == original


def test_dump_config_roundtrips_custom_with_plugins() -> None:
    original = Config(
        core=CoreConfig(max_blast_radius=BlastRadius.LOCAL_WRITE, network_allowlist=('example.com',)),
        model=ModelConfig(provider='ollama', model='qwen3:4b', host='http://localhost:11434'),
        storage=StorageConfig(provider='sqlite', path='/var/butter/db'),
        plugins=(PluginSource(repo='github.com/example/notes', ref='v0.1.0'),),
    )
    assert load_config(dump_config(original)) == original


def test_dump_config_escapes_special_chars() -> None:
    original = Config(storage=StorageConfig(path='C:\\Users\\butter\\"weird"\\db.sqlite'))
    assert load_config(dump_config(original)) == original


# --- /quit ------------------------------------------------------------------


async def test_quit_command_sets_exit_flag() -> None:
    result = await QuitCommand().run('', _ScriptedInput([]), _Recording())
    assert result == CommandResult(exit=True)


# --- /help ------------------------------------------------------------------


async def test_help_lists_commands() -> None:
    out = _Recording()
    quit_cmd = QuitCommand()
    help_cmd = HelpCommand(lambda: (quit_cmd,))
    result = await help_cmd.run('', _ScriptedInput([]), out)
    assert result == CommandResult()
    rendered = out.buffer.getvalue()
    assert '/quit' in rendered
    assert QuitCommand.description in rendered


async def test_help_handles_empty_registry() -> None:
    out = _Recording()
    help_cmd = HelpCommand(lambda: ())
    await help_cmd.run('', _ScriptedInput([]), out)
    assert 'no slash commands registered' in out.buffer.getvalue()


# --- /status ----------------------------------------------------------------


async def test_status_prints_resolved_config() -> None:
    out = _Recording()
    config = Config(
        model=ModelConfig(model='qwen3:8b', host='http://localhost:11434'),
        storage=StorageConfig(path='/tmp/butter.db'),
    )
    await StatusCommand(config=config, plugin_count=3).run('', _ScriptedInput([]), out)
    rendered = out.buffer.getvalue()
    assert 'qwen3:8b' in rendered
    assert 'http://localhost:11434' in rendered
    assert '/tmp/butter.db' in rendered
    assert 'plugins.registered' in rendered
    assert 'core.max_blast_radius' in rendered
    # The legacy abbreviated label must not leak back in.
    assert 'core.max_radius' not in rendered


# --- /configure -------------------------------------------------------------


async def test_configure_writes_roundtrippable_file(tmp_path: Path) -> None:
    config_path = tmp_path / 'config.toml'
    original = Config(
        plugins=(PluginSource(repo='github.com/example/notes', ref='v0.1.0'),),
    )
    inputs = _ScriptedInput(
        [
            '',  # model.provider — keep
            'qwen3:4b',  # model.model
            '',  # model.host — keep
            '/custom/butter.db',  # storage.path
            'local-write',  # core.max_blast_radius
        ]
    )
    out = _Recording()
    await ConfigureCommand(config=original, config_path=config_path).run('', inputs, out)

    assert config_path.exists()
    written = load_config(config_path.read_text())
    assert written.model.model == 'qwen3:4b'
    assert written.storage.path == '/custom/butter.db'
    assert written.core.max_blast_radius is BlastRadius.LOCAL_WRITE
    # Existing plugins survive a /configure pass.
    assert written.plugins == original.plugins
    assert 'Restart `butter` to apply' in out.buffer.getvalue()


async def test_configure_keeps_unchanged_values_on_blank_input(tmp_path: Path) -> None:
    config_path = tmp_path / 'config.toml'
    original = Config(model=ModelConfig(model='qwen3:8b', host='http://localhost:11434'))
    inputs = _ScriptedInput(['', '', '', '', ''])
    await ConfigureCommand(config=original, config_path=config_path).run('', inputs, _Recording())
    assert load_config(config_path.read_text()) == original


async def test_configure_reprompts_on_invalid_radius(tmp_path: Path) -> None:
    config_path = tmp_path / 'config.toml'
    original = Config()
    inputs = _ScriptedInput(['', '', '', '', 'not-a-radius', 'read-only'])
    out = _Recording()
    await ConfigureCommand(config=original, config_path=config_path).run('', inputs, out)
    rendered = out.buffer.getvalue()
    assert '[invalid]' in rendered
    written = load_config(config_path.read_text())
    assert written.core.max_blast_radius is BlastRadius.READ_ONLY


async def test_configure_aborts_cleanly_on_eof(tmp_path: Path) -> None:
    config_path = tmp_path / 'config.toml'
    out = _Recording()
    await ConfigureCommand(config=Config(), config_path=config_path).run('', _ScriptedInput([]), out)
    assert not config_path.exists()
    assert 'cancelled' in out.buffer.getvalue()


async def test_configure_reports_write_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config_path = tmp_path / 'sub' / 'config.toml'
    out = _Recording()

    def boom(self: Path, *args: object, **kwargs: object) -> None:
        raise PermissionError('denied')

    monkeypatch.setattr(Path, 'write_text', boom)
    await ConfigureCommand(config=Config(), config_path=config_path).run('', _ScriptedInput(['', '', '', '', '']), out)
    assert 'could not write' in out.buffer.getvalue()


# --- Dispatcher in Repl -----------------------------------------------------


async def test_repl_dispatches_slash_command() -> None:
    out = _Recording()
    quit_cmd = QuitCommand()
    registry = CommandRegistry((quit_cmd,))
    repl = Repl(
        _StubAgentLoop(),
        _ScriptedInput(['/quit', 'never reached']),
        out,
        commands=registry,
    )
    await repl.run()
    rendered = out.buffer.getvalue()
    # The stub agent loop would have appended a 'model reply' marker — its
    # absence confirms /quit short-circuited before model dispatch.
    assert 'never reached' not in rendered


async def test_repl_falls_through_non_slash_to_model() -> None:
    out = _Recording()
    agent = _StubAgentLoop()
    repl = Repl(
        agent,
        _ScriptedInput(['hello there']),
        out,
        commands=CommandRegistry((QuitCommand(),)),
    )
    await repl.run()
    assert agent.calls == ['hello there']


async def test_repl_dispatch_splits_on_any_whitespace() -> None:
    """`/status\\t--verbose` and `/status --verbose` both route to `status`."""
    out = _Recording()
    captured: list[str] = []

    class _Recorder:
        name = 'status'
        description = 'recorder'

        async def run(self, args: str, io_in, output) -> CommandResult:  # type: ignore[no-untyped-def]
            captured.append(args)
            return CommandResult()

    repl = Repl(
        _StubAgentLoop(),
        _ScriptedInput(['/status\t--verbose', '/status --verbose']),
        out,
        commands=CommandRegistry((_Recorder(),)),
    )
    await repl.run()
    assert captured == ['--verbose', '--verbose']


async def test_repl_unknown_command_renders_error() -> None:
    out = _Recording()
    repl = Repl(
        _StubAgentLoop(),
        _ScriptedInput(['/whoami']),
        out,
        commands=CommandRegistry(()),
    )
    await repl.run()
    assert '[error] unknown command: /whoami' in out.buffer.getvalue()


# --- build_default_commands -------------------------------------------------


def test_build_default_commands_includes_all_four() -> None:
    registry = build_default_commands(config=Config(), config_path=Path('/tmp/cfg.toml'), plugin_count=0)
    names = {cmd.name for cmd in registry.all()}
    assert names == {'help', 'quit', 'status', 'configure'}


def test_command_registry_rejects_duplicates() -> None:
    with pytest.raises(ValueError, match='duplicate slash command'):
        CommandRegistry((QuitCommand(), QuitCommand()))


# --- Stub agent loop --------------------------------------------------------


@dataclass
class _StubContextManager:
    payload: dict[str, object] = field(default_factory=dict)

    async def assemble(self, turn: Turn) -> ModelContext:
        return ModelContext(turn=turn, payload=dict(self.payload))


@dataclass
class _StubModel:
    async def generate(self, context: ModelContext) -> ModelOutput:
        return ModelReply(text=f'echo:{context.turn.user_input}')


class _StubExecutor:
    async def execute(self, plan: TaskPlan) -> ExecutionResult:
        raise AssertionError('executor should not be reached in these tests')


class _StubAgentLoop(AgentLoop):
    """`AgentLoop` subclass that records inputs without invoking a real model."""

    def __init__(self) -> None:
        super().__init__(_StubContextManager(), _StubModel(), _StubExecutor())
        self.calls: list[str] = []

    async def run_turn(self, user_input: str) -> TurnResult:
        self.calls.append(user_input)
        return await super().run_turn(user_input)


# Silence unused-import warnings for Mapping (kept for parity with sibling tests).
_ = Mapping
