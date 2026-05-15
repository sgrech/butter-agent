"""Plugin source loader — resolves `[[plugin]]` config entries to live `Plugin` instances.

Each declaration becomes a tuple `(PluginManifest, Plugin)` ready for
`RegistryBuilder.register()`. Two declaration shapes are supported:

- `PluginSource(repo, ref)` — production form. The configured `Fetcher`
  clones the pinned ref into an XDG cache directory; subsequent runs
  reuse the cache.
- `PluginPath(path)` — local-dev form. The path is used as-is, no
  fetcher involvement. Use this while iterating on a plugin alongside
  butter-agent.

Loading is fail-fast: any error — missing manifest, malformed
entrypoint, import failure, instantiation failure — aborts the whole
startup. The user declared the plugin in their config; if it doesn't
work, the right answer is to surface the diagnostic and stop, not to
silently boot with a partial registry.

What this module deliberately does NOT do (v1):

- Install plugin dependencies. Plugins must be stdlib-only or rely on
  packages already present in butter's environment. A future revision
  may add `pip install --target` into a per-plugin site-packages
  directory.
- Verify the cloned ref's signature or contents beyond what `git
  clone --branch <ref>` enforces. Plugin authors and operators are
  responsible for trusting their declared sources.
- Hot-reload. Invariant #2 (frozen registry) holds: changes require a
  butter-agent restart.

The `Fetcher` Protocol is the seam tests inject against. Production
uses `GitFetcher`; tests use a stub returning a known fixture path.
"""

from __future__ import annotations

import importlib
import inspect
import os
import subprocess
import sys
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, cast

from butter_agent.core.config import PluginDeclaration, PluginPath, PluginSource
from butter_agent.core.registry import Plugin, PluginManifest, parse_manifest
from butter_agent.plugin_api import MANIFEST_FILENAME

# --- Errors ------------------------------------------------------------------


class PluginLoadError(Exception):
    """Raised when a declared plugin cannot be loaded.

    Surfaces a single uniform error type so the CLI's startup-error
    path can render it consistently (alongside `ConfigError` and
    `OSError`).
    """


# --- Fetcher seam ------------------------------------------------------------


class Fetcher(Protocol):
    """Resolves a `PluginSource` to a local directory containing the plugin.

    Implementations MUST raise `PluginLoadError` on any failure — the
    loader does not catch generic exceptions, so wrap network / process
    errors here.
    """

    def fetch(self, source: PluginSource) -> Path: ...


class GitFetcher:
    """Default `Fetcher`: clones pinned refs via `git clone --depth 1 --branch`.

    Cache layout: `{cache_root}/plugins/{slug}-{ref}/`. Slugs are
    derived by replacing non-alphanumeric characters in the repo
    identifier with `_`. Cache hits skip the clone entirely — refs are
    pinned by config so the same `(repo, ref)` pair always resolves to
    the same revision.

    `cache_root` defaults to `$XDG_CACHE_HOME/butter-agent` (or
    `~/.cache/butter-agent`). Tests inject a `tmp_path` here.
    """

    def __init__(self, cache_root: Path | None = None) -> None:
        self._cache_root = cache_root if cache_root is not None else _default_cache_root()

    def fetch(self, source: PluginSource) -> Path:
        slug = _slugify(source.repo)
        dest = self._cache_root / 'plugins' / f'{slug}-{source.ref}'
        if dest.exists():
            return dest
        dest.parent.mkdir(parents=True, exist_ok=True)
        url = f'https://{source.repo}.git'
        try:
            subprocess.run(
                ['git', 'clone', '--depth', '1', '--branch', source.ref, url, str(dest)],
                check=True,
                capture_output=True,
                text=True,
            )
        except FileNotFoundError as exc:
            raise PluginLoadError('git executable not found on PATH — required to fetch remote plugins') from exc
        except subprocess.CalledProcessError as exc:
            raise PluginLoadError(
                f'git clone failed for {source.repo}@{source.ref}: {exc.stderr.strip() or exc}',
            ) from exc
        return dest


def _default_cache_root() -> Path:
    xdg = os.environ.get('XDG_CACHE_HOME') or None
    base = Path(xdg).expanduser() if xdg else Path.home() / '.cache'
    return base / 'butter-agent'


def _slugify(repo: str) -> str:
    return ''.join(ch if ch.isalnum() else '_' for ch in repo)


# --- Loader ------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class LoadedPlugin:
    """One successfully loaded plugin — manifest plus a live instance ready to register.

    `config` is carried straight through from the declaration so the
    bootstrap can hand it to `RegistryBuilder.register` without re-reading
    `config.toml`.
    """

    manifest: PluginManifest
    plugin: Plugin
    config: Mapping[str, object] = field(default_factory=dict)


class PluginLoader:
    """Resolves declarations to `LoadedPlugin` entries.

    Holds a `Fetcher` for remote sources. Path declarations bypass the
    fetcher and are used directly.
    """

    def __init__(self, fetcher: Fetcher | None = None) -> None:
        self._fetcher: Fetcher = fetcher if fetcher is not None else GitFetcher()

    def load_all(self, declarations: Iterable[PluginDeclaration]) -> tuple[LoadedPlugin, ...]:
        return tuple(self._load_one(d) for d in declarations)

    def _load_one(self, declaration: PluginDeclaration) -> LoadedPlugin:
        plugin_dir = self._resolve(declaration)
        manifest = _read_manifest(plugin_dir)
        plugin = _import_entrypoint(plugin_dir, manifest)
        return LoadedPlugin(manifest=manifest, plugin=plugin, config=declaration.config)

    def _resolve(self, declaration: PluginDeclaration) -> Path:
        if isinstance(declaration, PluginPath):
            resolved = Path(declaration.path).expanduser()
            if not resolved.is_dir():
                raise PluginLoadError(f'plugin path {str(resolved)!r} does not exist or is not a directory')
            return resolved
        return self._fetcher.fetch(declaration)


def _read_manifest(plugin_dir: Path) -> PluginManifest:
    manifest_path = plugin_dir / MANIFEST_FILENAME
    if not manifest_path.is_file():
        raise PluginLoadError(f'plugin at {plugin_dir} is missing {MANIFEST_FILENAME}')
    try:
        text = manifest_path.read_text()
    except OSError as exc:
        raise PluginLoadError(f'could not read {manifest_path}: {exc}') from exc
    try:
        return parse_manifest(text)
    except Exception as exc:
        # parse_manifest raises ManifestError; re-wrap as PluginLoadError so
        # the CLI's startup-error renderer sees a single uniform type.
        raise PluginLoadError(f'invalid manifest at {manifest_path}: {exc}') from exc


def _import_entrypoint(plugin_dir: Path, manifest: PluginManifest) -> Plugin:
    """Resolve the manifest's `module:Class` entrypoint to a live instance.

    Adds the plugin's `src/` directory to `sys.path` when present (the
    documented layout); falls back to the plugin directory itself for
    plugins that ship a flat layout. Reuses any already-imported module
    so reloading the same plugin twice in one process doesn't surface
    stale state.
    """
    if ':' not in manifest.entrypoint:
        raise PluginLoadError(
            f"plugin {manifest.name!r}: entrypoint {manifest.entrypoint!r} must be 'module:Class'",
        )
    module_path, _, attr = manifest.entrypoint.partition(':')
    if not module_path or not attr:
        raise PluginLoadError(
            f"plugin {manifest.name!r}: entrypoint {manifest.entrypoint!r} must be 'module:Class'",
        )

    src_dir = plugin_dir / 'src'
    import_root = src_dir if src_dir.is_dir() else plugin_dir
    root_str = str(import_root.resolve())
    if root_str not in sys.path:
        sys.path.insert(0, root_str)

    try:
        module = importlib.import_module(module_path)
    except ImportError as exc:
        raise PluginLoadError(
            f'plugin {manifest.name!r}: could not import {module_path!r} from {import_root}: {exc}',
        ) from exc

    # Guard against sys.modules cache collisions: `import_module` returns the
    # already-loaded module if the same top-level name was imported earlier,
    # which would silently route to the wrong plugin if two plugins shared a
    # package name. Verify the resolved module actually lives under our
    # import root before trusting it.
    module_file = getattr(module, '__file__', None)
    if module_file is None:
        raise PluginLoadError(
            f'plugin {manifest.name!r}: imported module {module_path!r} has no __file__ — cannot verify it belongs to this plugin (namespace package or builtin?)',
        )
    try:
        Path(module_file).resolve().relative_to(import_root.resolve())
    except ValueError as exc:
        raise PluginLoadError(
            f'plugin {manifest.name!r}: module {module_path!r} resolved to {module_file} which is outside the plugin root {import_root} — another plugin or package likely shadows this name. Rename your top-level package to something unique.',
        ) from exc

    try:
        cls = getattr(module, attr)
    except AttributeError as exc:
        raise PluginLoadError(
            f'plugin {manifest.name!r}: {module_path}.{attr} not found',
        ) from exc

    if not callable(cls):
        raise PluginLoadError(
            f'plugin {manifest.name!r}: entrypoint {manifest.entrypoint!r} is not callable',
        )

    try:
        instance = cls()
    except Exception as exc:
        raise PluginLoadError(
            f'plugin {manifest.name!r}: constructing {manifest.entrypoint} raised {type(exc).__name__}: {exc}',
        ) from exc

    execute = getattr(instance, 'execute', None)
    if execute is None or not callable(execute):
        raise PluginLoadError(
            f'plugin {manifest.name!r}: {manifest.entrypoint} does not implement the Plugin Protocol (missing async `execute`)',
        )
    # The runtime contract awaits `execute`; a sync def would only blow up at
    # the first invocation. Catch that at load time so the plugin author sees
    # a clear diagnostic instead of a downstream `TypeError`.
    if not inspect.iscoroutinefunction(execute):
        raise PluginLoadError(
            f'plugin {manifest.name!r}: {manifest.entrypoint}.execute must be `async def` — synchronous execute methods are not supported by the Plugin Protocol',
        )
    # The Plugin Protocol takes three arguments after `self`:
    # `(capability, inputs, context)`. A legacy plugin written against the
    # old two-arg signature would import successfully and only crash on the
    # first plan step with a TypeError — surface that as a load-time error
    # so the operator can pin or update the plugin before booting.
    expected_params = 3  # capability, inputs, context (excluding self)
    sig = inspect.signature(execute)
    actual_params = len(sig.parameters)
    if actual_params != expected_params:
        raise PluginLoadError(
            f'plugin {manifest.name!r}: {manifest.entrypoint}.execute must accept (capability, inputs, context) — got {actual_params} positional parameter(s). This plugin is likely written against an older Plugin Protocol version.',
        )
    # Structural Plugin Protocol — cast through `object` to satisfy mypy
    # since `cls()` returned `Any`. The runtime hasattr check above is the
    # actual contract gate.
    return cast(Plugin, instance)
