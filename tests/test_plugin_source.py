"""Tests for the plugin source loader.

Covers the public load path (manifest + entrypoint resolution against
real on-disk fixtures), the failure modes a plugin author / operator
will hit (missing manifest, malformed entrypoint, bad import, bad
instance), the `Fetcher` seam (stubbed — never hits the network), the
default `GitFetcher`'s caching behaviour, and `PluginPath` resolution.

Two fixtures live under `tests/fixtures/`:

- `fake_plugin/` — documented src/ layout. Most tests use this.
- `fake_plugin_flat/` — flat layout (package at repo root, no src/).
  Exercises the fallback path in `_import_entrypoint`.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from butter_agent.core.config import PluginPath, PluginSource
from butter_agent.core.plugin_source import (
    GitFetcher,
    LoadedPlugin,
    PluginLoader,
    PluginLoadError,
)
from butter_agent.core.registry import BlastRadius

FIXTURES = Path(__file__).resolve().parent / 'fixtures'
SRC_FIXTURE = FIXTURES / 'fake_plugin'
FLAT_FIXTURE = FIXTURES / 'fake_plugin_flat'


# --- Stub fetcher -----------------------------------------------------------


class _StubFetcher:
    """Maps known `(repo, ref)` pairs to fixture paths. Fails loudly on miss.

    Tests inject this so loader logic exercises the same code paths as
    production without touching the network.
    """

    def __init__(self, mapping: dict[tuple[str, str], Path]) -> None:
        self._mapping = mapping

    def fetch(self, source: PluginSource) -> Path:
        try:
            return self._mapping[(source.repo, source.ref)]
        except KeyError as exc:
            raise PluginLoadError(f'stub fetcher: no fixture for {source.repo}@{source.ref}') from exc


# --- PluginPath happy paths -------------------------------------------------


def test_load_local_path_returns_manifest_and_instance() -> None:
    loader = PluginLoader(fetcher=_StubFetcher({}))
    loaded = loader.load_all([PluginPath(path=str(SRC_FIXTURE))])
    assert len(loaded) == 1
    entry = loaded[0]
    assert isinstance(entry, LoadedPlugin)
    assert entry.manifest.name == 'fake'
    assert entry.manifest.blast_radius is BlastRadius.READ_ONLY


async def test_loaded_plugin_executes() -> None:
    """Plugin instance returned by the loader is callable as documented."""
    loader = PluginLoader(fetcher=_StubFetcher({}))
    (entry,) = loader.load_all([PluginPath(path=str(SRC_FIXTURE))])
    result = await entry.plugin.execute('ping', {})
    assert result == {'reply': 'pong'}


def test_flat_layout_plugin_loads() -> None:
    """A plugin with no `src/` directory falls back to the manifest dir."""
    loader = PluginLoader(fetcher=_StubFetcher({}))
    (entry,) = loader.load_all([PluginPath(path=str(FLAT_FIXTURE))])
    assert entry.manifest.name == 'fake_flat'


# --- PluginPath failure modes -----------------------------------------------


def test_missing_path_raises(tmp_path: Path) -> None:
    loader = PluginLoader(fetcher=_StubFetcher({}))
    with pytest.raises(PluginLoadError, match='does not exist'):
        loader.load_all([PluginPath(path=str(tmp_path / 'nope'))])


def test_missing_manifest_raises(tmp_path: Path) -> None:
    # Empty directory — exists but has no manifest.toml.
    loader = PluginLoader(fetcher=_StubFetcher({}))
    with pytest.raises(PluginLoadError, match=r'missing manifest\.toml'):
        loader.load_all([PluginPath(path=str(tmp_path))])


def test_malformed_manifest_raises(tmp_path: Path) -> None:
    (tmp_path / 'manifest.toml').write_text('this is not valid toml [[[')
    loader = PluginLoader(fetcher=_StubFetcher({}))
    with pytest.raises(PluginLoadError, match='invalid manifest'):
        loader.load_all([PluginPath(path=str(tmp_path))])


def test_bad_entrypoint_format_raises(tmp_path: Path) -> None:
    (tmp_path / 'manifest.toml').write_text(
        """
[plugin]
name = "broken"
version = "0.1.0"
entrypoint = "no_colon_here"
blast_radius = "read-only"

[[capability]]
name = "x"
description = "x"
input_schema = {}
output_schema = {}
"""
    )
    loader = PluginLoader(fetcher=_StubFetcher({}))
    with pytest.raises(PluginLoadError, match="must be 'module:Class'"):
        loader.load_all([PluginPath(path=str(tmp_path))])


def test_unimportable_module_raises(tmp_path: Path) -> None:
    (tmp_path / 'manifest.toml').write_text(
        """
[plugin]
name = "broken"
version = "0.1.0"
entrypoint = "does_not_exist_anywhere_xyzzy:Plugin"
blast_radius = "read-only"

[[capability]]
name = "x"
description = "x"
input_schema = {}
output_schema = {}
"""
    )
    loader = PluginLoader(fetcher=_StubFetcher({}))
    with pytest.raises(PluginLoadError, match='could not import'):
        loader.load_all([PluginPath(path=str(tmp_path))])


def test_missing_class_attribute_raises(tmp_path: Path) -> None:
    pkg_dir = tmp_path / 'src' / 'mod_missing_cls'
    pkg_dir.mkdir(parents=True)
    (pkg_dir / '__init__.py').write_text('# no Plugin class here\n')
    (tmp_path / 'manifest.toml').write_text(
        """
[plugin]
name = "broken"
version = "0.1.0"
entrypoint = "mod_missing_cls:Plugin"
blast_radius = "read-only"

[[capability]]
name = "x"
description = "x"
input_schema = {}
output_schema = {}
"""
    )
    loader = PluginLoader(fetcher=_StubFetcher({}))
    with pytest.raises(PluginLoadError, match='not found'):
        loader.load_all([PluginPath(path=str(tmp_path))])


def test_instance_without_execute_raises(tmp_path: Path) -> None:
    pkg_dir = tmp_path / 'src' / 'mod_no_execute'
    pkg_dir.mkdir(parents=True)
    (pkg_dir / '__init__.py').write_text('class Plugin:\n    pass\n')
    (tmp_path / 'manifest.toml').write_text(
        """
[plugin]
name = "broken"
version = "0.1.0"
entrypoint = "mod_no_execute:Plugin"
blast_radius = "read-only"

[[capability]]
name = "x"
description = "x"
input_schema = {}
output_schema = {}
"""
    )
    loader = PluginLoader(fetcher=_StubFetcher({}))
    with pytest.raises(PluginLoadError, match='does not implement the Plugin Protocol'):
        loader.load_all([PluginPath(path=str(tmp_path))])


# --- PluginSource path (via stub fetcher) -----------------------------------


def test_load_source_via_fetcher() -> None:
    """Source-mode declarations route through the fetcher seam."""
    fetcher = _StubFetcher({('github.com/example/fake', 'v0.1.0'): SRC_FIXTURE})
    loader = PluginLoader(fetcher=fetcher)
    (entry,) = loader.load_all([PluginSource(repo='github.com/example/fake', ref='v0.1.0')])
    assert entry.manifest.name == 'fake'


def test_fetcher_failure_surfaces_as_load_error() -> None:
    loader = PluginLoader(fetcher=_StubFetcher({}))
    with pytest.raises(PluginLoadError, match='no fixture'):
        loader.load_all([PluginSource(repo='github.com/example/missing', ref='v0.1.0')])


# --- GitFetcher caching + git-missing -------------------------------------


def test_git_fetcher_uses_cached_clone(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Cache hits must skip the clone subprocess entirely.

    Pre-populates the expected cache path; if `git clone` were invoked,
    it would fail on the bogus repo URL.
    """
    fetcher = GitFetcher(cache_root=tmp_path)
    source = PluginSource(repo='github.com/example/cached', ref='v0.1.0')
    expected = tmp_path / 'plugins' / 'github_com_example_cached-v0.1.0'
    expected.mkdir(parents=True)

    calls: list[list[str]] = []

    def _record(args: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        calls.append(args)
        raise AssertionError('cache hit must not invoke git clone')

    monkeypatch.setattr(subprocess, 'run', _record)
    assert fetcher.fetch(source) == expected
    assert calls == []


def test_git_fetcher_reports_missing_git_binary(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fetcher = GitFetcher(cache_root=tmp_path)

    def _no_git(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        raise FileNotFoundError("[Errno 2] No such file or directory: 'git'")

    monkeypatch.setattr(subprocess, 'run', _no_git)
    with pytest.raises(PluginLoadError, match='git executable not found'):
        fetcher.fetch(PluginSource(repo='github.com/example/missing', ref='v0.1.0'))


def test_git_fetcher_reports_clone_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fetcher = GitFetcher(cache_root=tmp_path)

    def _clone_fails(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        # `cmd` is purely informational for `__str__`; the loader reads `stderr`.
        raise subprocess.CalledProcessError(returncode=128, cmd=['git'], stderr='fatal: ref not found\n')

    monkeypatch.setattr(subprocess, 'run', _clone_fails)
    with pytest.raises(PluginLoadError, match=r'git clone failed.*ref not found'):
        fetcher.fetch(PluginSource(repo='github.com/example/missing', ref='v999.0.0'))


# --- Multiple plugins -------------------------------------------------------


def test_loads_multiple_declarations_in_order() -> None:
    fetcher = _StubFetcher({('github.com/example/fake', 'v0.1.0'): SRC_FIXTURE})
    loader = PluginLoader(fetcher=fetcher)
    loaded = loader.load_all(
        [
            PluginPath(path=str(FLAT_FIXTURE)),
            PluginSource(repo='github.com/example/fake', ref='v0.1.0'),
        ]
    )
    assert [e.manifest.name for e in loaded] == ['fake_flat', 'fake']


# --- sys.path hygiene -------------------------------------------------------


def test_loading_does_not_pollute_sys_path_with_duplicates() -> None:
    """Re-loading the same plugin twice should not append `src/` to sys.path twice."""
    loader = PluginLoader(fetcher=_StubFetcher({}))
    loader.load_all([PluginPath(path=str(SRC_FIXTURE))])
    before = sys.path.count(str(SRC_FIXTURE / 'src'))
    loader.load_all([PluginPath(path=str(SRC_FIXTURE))])
    after = sys.path.count(str(SRC_FIXTURE / 'src'))
    assert before == after
