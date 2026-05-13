"""Tests for the `butter` CLI entrypoint.

Covers argparse wiring (no-arg defaults to `start`, explicit subcommand
parsing), the `configure` placeholder output, and the start-command's
error path when config loading fails.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

import pytest

from butter_agent import cli
from butter_agent.core.config import Config


def test_parser_defaults_to_no_command() -> None:
    args = cli.build_parser().parse_args([])
    assert args.command is None


def test_parser_accepts_start_subcommand() -> None:
    args = cli.build_parser().parse_args(['start'])
    assert args.command == 'start'


def test_parser_accepts_configure_subcommand() -> None:
    args = cli.build_parser().parse_args(['configure'])
    assert args.command == 'configure'


def test_main_configure_prints_config_path(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setenv('XDG_CONFIG_HOME', str(tmp_path))
    exit_code = cli.main(['configure'])
    captured = capsys.readouterr()
    assert exit_code == 0
    assert str(tmp_path / 'butter-agent' / 'config.toml') in captured.out


def test_main_start_runs_repl_with_loaded_config(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    config_path = tmp_path / 'config.toml'
    config_path.write_text('[model]\nmodel = "qwen3:4b"\n')
    monkeypatch.setattr(cli, 'resolve_config_path', lambda: config_path)

    captured: dict[str, Any] = {'ran': False, 'closed': False}

    async def fake_build_repl(config: Config, **_: object) -> _RecordingApp:
        captured['config'] = config
        return _RecordingApp(captured)

    monkeypatch.setattr(cli, 'build_repl', fake_build_repl)
    exit_code = cli.main(['start'])

    assert exit_code == 0
    assert captured['ran'] is True
    assert captured['closed'] is True
    cfg = captured['config']
    assert isinstance(cfg, Config)
    assert cfg.model.model == 'qwen3:4b'


def test_main_start_honours_explicit_config_flag(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    flag_path = tmp_path / 'override.toml'
    flag_path.write_text('[model]\nmodel = "qwen3:override"\n')

    def _unexpected() -> Path:  # pragma: no cover - guard
        raise AssertionError('resolve_config_path should not be called when --config is given')

    monkeypatch.setattr(cli, 'resolve_config_path', _unexpected)

    captured: dict[str, Any] = {'ran': False, 'closed': False}

    async def fake_build_repl(config: Config, **_: object) -> _RecordingApp:
        captured['config'] = config
        return _RecordingApp(captured)

    monkeypatch.setattr(cli, 'build_repl', fake_build_repl)
    assert cli.main(['start', '--config', str(flag_path)]) == 0
    cfg = captured['config']
    assert isinstance(cfg, Config)
    assert cfg.model.model == 'qwen3:override'


def test_main_start_renders_oserror_as_friendly_diagnostic(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr(cli, 'resolve_config_path', lambda: tmp_path / 'missing.toml')

    async def fake_build_repl(_config: Config, **_: object) -> _RecordingApp:
        raise PermissionError('cannot create storage dir')

    monkeypatch.setattr(cli, 'build_repl', fake_build_repl)
    exit_code = cli.main(['start'])
    captured = capsys.readouterr()
    assert exit_code == 2
    assert 'cannot create storage dir' in captured.err


def test_main_start_renders_sqlite_error_as_friendly_diagnostic(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr(cli, 'resolve_config_path', lambda: tmp_path / 'missing.toml')

    async def fake_build_repl(_config: Config, **_: object) -> _RecordingApp:
        raise sqlite3.OperationalError('database file is corrupted')

    monkeypatch.setattr(cli, 'build_repl', fake_build_repl)
    exit_code = cli.main(['start'])
    captured = capsys.readouterr()
    assert exit_code == 2
    assert 'database file is corrupted' in captured.err


def test_main_start_returns_error_code_on_bad_config(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    config_path = tmp_path / 'config.toml'
    config_path.write_text('not [valid toml')
    monkeypatch.setattr(cli, 'resolve_config_path', lambda: config_path)

    exit_code = cli.main(['start'])

    captured = capsys.readouterr()
    assert exit_code == 2
    assert 'config' in captured.err.lower()


def test_main_no_args_defaults_to_start(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(cli, 'resolve_config_path', lambda: tmp_path / 'missing.toml')

    captured: dict[str, Any] = {'ran': False, 'closed': False}

    async def fake_build_repl(config: Config, **_: object) -> _RecordingApp:
        return _RecordingApp(captured)

    monkeypatch.setattr(cli, 'build_repl', fake_build_repl)
    assert cli.main([]) == 0
    assert captured['ran'] is True
    assert captured['closed'] is True


class _RecordingApp:
    """Stand-in for `app.App` — tracks `repl.run()` and `close()` invocations."""

    def __init__(self, captured: dict[str, Any]) -> None:
        self._captured = captured
        self.repl = _RecordingRepl(captured)

    async def close(self) -> None:
        self._captured['closed'] = True


class _RecordingRepl:
    def __init__(self, captured: dict[str, Any]) -> None:
        self._captured = captured

    async def run(self) -> None:
        self._captured['ran'] = True
