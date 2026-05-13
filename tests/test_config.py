"""Tests for the core config loader.

The loader is the only place raw `config.toml` becomes typed values. The
contract checked here:

- Defaults: an empty document yields the documented out-of-box config
  (Ollama / Qwen3 8B, local SQLite, network ceiling, no plugins).
- Atomic validation: malformed TOML, wrong types, unknown enums, and
  unpinned plugin refs raise `ConfigError` without producing a partial
  Config.
- Pin enforcement: branch refs (`main` / `master` / `HEAD`,
  case-insensitive) are rejected — the "pinned refs only" rule from the
  scope is enforced at parse time, not at fetch time.
- Section independence: defaults only apply to omitted sections; an
  override leaves siblings alone.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from butter_agent.core.config import (
    Config,
    ConfigError,
    CoreConfig,
    ModelConfig,
    PluginSource,
    StorageConfig,
    load_config,
)
from butter_agent.core.registry import BlastRadius

# --- Defaults ---------------------------------------------------------------


def test_empty_document_yields_documented_defaults() -> None:
    cfg = load_config('')
    assert cfg == Config(
        core=CoreConfig(max_blast_radius=BlastRadius.NETWORK, network_allowlist=()),
        model=ModelConfig(provider='ollama', model='qwen3:8b', host='http://localhost:11434'),
        storage=StorageConfig(provider='sqlite', path='~/.butter-agent/butter.db'),
        plugins=(),
    )


def test_shipped_default_config_loads_cleanly() -> None:
    # The repo-root config.toml is the documented first-run experience —
    # it must always parse and reproduce the in-code defaults.
    repo_root = Path(__file__).resolve().parent.parent
    text = (repo_root / 'config.toml').read_text()
    assert load_config(text) == Config()


# --- Core section -----------------------------------------------------------


def test_core_section_overrides_ceiling_and_allowlist() -> None:
    cfg = load_config(
        """
        [core]
        max_blast_radius = "local-write"
        network_allowlist = ["api.example.com", "*.internal"]
        """,
    )
    assert cfg.core.max_blast_radius is BlastRadius.LOCAL_WRITE
    assert cfg.core.network_allowlist == ('api.example.com', '*.internal')


def test_core_rejects_unknown_blast_radius() -> None:
    with pytest.raises(ConfigError, match='invalid value'):
        load_config('[core]\nmax_blast_radius = "yolo"')


def test_core_rejects_non_string_allowlist_item() -> None:
    with pytest.raises(ConfigError, match=r'allowlist\[1\]'):
        load_config('[core]\nnetwork_allowlist = ["ok", 42]')


def test_core_rejects_non_array_allowlist() -> None:
    with pytest.raises(ConfigError, match='network_allowlist'):
        load_config('[core]\nnetwork_allowlist = "nope"')


def test_core_rejects_non_string_radius() -> None:
    with pytest.raises(ConfigError, match='expected string'):
        load_config('[core]\nmax_blast_radius = 3')


# --- Model section ----------------------------------------------------------


def test_model_section_partial_override_leaves_defaults() -> None:
    cfg = load_config('[model]\nmodel = "llama3:8b"')
    assert cfg.model == ModelConfig(provider='ollama', model='llama3:8b', host='http://localhost:11434')


def test_model_section_full_override() -> None:
    cfg = load_config(
        """
        [model]
        provider = "anthropic"
        model = "claude-haiku-4-5"
        host = "https://api.anthropic.com"
        """,
    )
    assert cfg.model == ModelConfig(
        provider='anthropic',
        model='claude-haiku-4-5',
        host='https://api.anthropic.com',
    )


def test_model_rejects_non_string_field() -> None:
    with pytest.raises(ConfigError, match=r'model\.provider'):
        load_config('[model]\nprovider = 7')


# --- Storage section --------------------------------------------------------


def test_storage_section_override() -> None:
    cfg = load_config('[storage]\npath = "/var/lib/butter/state.db"')
    assert cfg.storage.path == '/var/lib/butter/state.db'
    assert cfg.storage.provider == 'sqlite'


# --- Plugin sources ---------------------------------------------------------


def test_plugin_with_tag_ref() -> None:
    cfg = load_config(
        """
        [[plugin]]
        source = "github.com/example/notes@v0.2.1"
        """,
    )
    assert cfg.plugins == (PluginSource(repo='github.com/example/notes', ref='v0.2.1'),)


def test_plugin_with_commit_sha_ref() -> None:
    sha = 'a' * 40
    cfg = load_config(f'[[plugin]]\nsource = "github.com/example/notes@{sha}"')
    assert cfg.plugins == (PluginSource(repo='github.com/example/notes', ref=sha),)


def test_multiple_plugins_preserve_order() -> None:
    cfg = load_config(
        """
        [[plugin]]
        source = "github.com/a/one@v1"
        [[plugin]]
        source = "github.com/b/two@v2"
        """,
    )
    assert [p.repo for p in cfg.plugins] == ['github.com/a/one', 'github.com/b/two']


def test_plugin_missing_at_rejected() -> None:
    with pytest.raises(ConfigError, match='missing a pinned ref'):
        load_config('[[plugin]]\nsource = "github.com/example/notes"')


def test_plugin_empty_ref_rejected() -> None:
    with pytest.raises(ConfigError, match='empty ref'):
        load_config('[[plugin]]\nsource = "github.com/example/notes@"')


def test_plugin_empty_repo_rejected() -> None:
    with pytest.raises(ConfigError, match='empty repo'):
        load_config('[[plugin]]\nsource = "@v0.1.0"')


@pytest.mark.parametrize('ref', ['main', 'master', 'HEAD', 'MAIN', 'Master', 'head'])
def test_plugin_branch_refs_rejected(ref: str) -> None:
    with pytest.raises(ConfigError, match='unpinned ref'):
        load_config(f'[[plugin]]\nsource = "github.com/example/notes@{ref}"')


def test_plugin_missing_source_key() -> None:
    with pytest.raises(ConfigError, match='missing or empty string `source`'):
        load_config('[[plugin]]\nname = "notes"')


def test_plugin_non_string_source() -> None:
    with pytest.raises(ConfigError, match='missing or empty string `source`'):
        load_config('[[plugin]]\nsource = 42')


def test_plugin_entries_must_be_array() -> None:
    # `plugin = "x"` would put a string under `plugin`, which we reject.
    with pytest.raises(ConfigError, match=r'\[\[plugin\]\] must be an array'):
        load_config('plugin = "oops"')


# --- Top-level / TOML errors -----------------------------------------------


def test_malformed_toml_is_config_error() -> None:
    with pytest.raises(ConfigError, match='invalid TOML'):
        load_config('this is not [valid')


def test_section_must_be_table() -> None:
    with pytest.raises(ConfigError, match=r'\[core\] must be a table'):
        load_config('core = "oops"')


# --- Pin ref retains case (only the rejection check is case-insensitive) ----


def test_plugin_ref_case_preserved() -> None:
    cfg = load_config('[[plugin]]\nsource = "github.com/example/notes@V1.0.0"')
    assert cfg.plugins[0].ref == 'V1.0.0'
