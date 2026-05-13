"""Tests for `butter_agent.app` — config path resolution and composition.

Covers:

- `resolve_config_path` honours `$XDG_CONFIG_HOME` and falls back to
  `~/.config/butter-agent/config.toml` otherwise.
- `load_or_default_config` returns the documented defaults when the
  target file does not exist, and parses it when it does.
- `build_repl` wires Database + history + registry + context manager +
  Ollama client + executor into a runnable `Repl` against a real on-disk
  SQLite path under `tmp_path`.
"""

from __future__ import annotations

import io
from pathlib import Path

import pytest

from butter_agent.app import build_repl, load_or_default_config, resolve_config_path
from butter_agent.core.config import Config, ConfigError, ModelConfig, StorageConfig, load_config
from butter_agent.core.repl import Repl, StdioInputSource


def test_resolve_config_path_uses_xdg_config_home(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv('XDG_CONFIG_HOME', str(tmp_path))
    assert resolve_config_path() == tmp_path / 'butter-agent' / 'config.toml'


def test_resolve_config_path_falls_back_to_home_dotconfig(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv('XDG_CONFIG_HOME', raising=False)
    monkeypatch.setattr(Path, 'home', classmethod(lambda _cls: tmp_path))
    assert resolve_config_path() == tmp_path / '.config' / 'butter-agent' / 'config.toml'


def test_load_or_default_config_returns_defaults_when_missing(tmp_path: Path) -> None:
    config = load_or_default_config(tmp_path / 'missing.toml')
    assert config == load_config('')


def test_load_or_default_config_reads_existing_file(tmp_path: Path) -> None:
    path = tmp_path / 'config.toml'
    path.write_text('[model]\nmodel = "qwen3:4b"\n')
    config = load_or_default_config(path)
    assert config.model.model == 'qwen3:4b'


def test_load_or_default_config_propagates_config_error(tmp_path: Path) -> None:
    path = tmp_path / 'broken.toml'
    path.write_text('not [valid toml')
    with pytest.raises(ConfigError):
        load_or_default_config(path)


async def test_build_repl_returns_wired_repl(tmp_path: Path) -> None:
    db_path = tmp_path / 'butter.db'
    config = Config(
        model=ModelConfig(model='qwen3:8b', host='http://localhost:11434'),
        storage=StorageConfig(path=str(db_path)),
    )
    in_buf = io.StringIO('')
    out_buf = io.StringIO()
    repl = await build_repl(
        config,
        input_source=StdioInputSource(in_buf, prompt_stream=out_buf),
        output=_RecordingOutput(out_buf),
    )
    assert isinstance(repl, Repl)
    # The Repl runs to EOF immediately on empty input, exercising the wiring.
    await repl.run()
    assert 'butter-agent' in out_buf.getvalue()
    assert str(db_path) in out_buf.getvalue()
    assert db_path.exists()


class _RecordingOutput:
    def __init__(self, stream: io.StringIO) -> None:
        self._stream = stream

    def write(self, text: str) -> None:
        self._stream.write(text)
