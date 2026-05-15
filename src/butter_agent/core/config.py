"""Core config — parsed `config.toml` consumed at startup.

This module owns the **declarative** half of the runtime contract: what
the operator wrote in `config.toml`. It is the only place where raw TOML
becomes typed values for the rest of `core/` to consume.

Three responsibilities:

- Parse `config.toml` into a frozen `Config` value with typed sections
  (`core`, `model`, `storage`, `plugins`).
- Validate atomically. Unknown blast radii, malformed plugin sources, or
  type mismatches raise `ConfigError` before any startup wiring proceeds.
- Apply sensible defaults when a section or key is omitted, so a brand-new
  install with an empty `config.toml` still boots into the documented
  out-of-box experience: Ollama + Qwen3 8B, local SQLite, network plugins
  allowed but gated.

What this module does NOT do:

- Read `config.toml` from disk. The caller picks the path (project-local,
  XDG, env override) and passes the text in. This keeps the loader pure
  and trivially testable.
- Clone plugin repos. `PluginSource` is parsed data only; the actual
  fetch is a separate future I/O module (see the note in `registry.py`).
- Construct the registry. The `Config` is fed to a builder at startup;
  the builder owns the invariant-#7 ceiling enforcement.
"""

from __future__ import annotations

import math
import re
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Final

from butter_agent.core.registry import BlastRadius

# --- Errors ------------------------------------------------------------------


class ConfigError(Exception):
    """Raised when `config.toml` fails to parse or validate.

    Distinct from `ManifestError` (per-plugin) — this is core config only.
    """


# --- Value types -------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CoreConfig:
    """Core-side policy: blast-radius ceiling, network allowlist, discovery.

    `max_blast_radius` is the invariant-#7 ceiling — `RegistryBuilder`
    rejects any plugin whose declared radius exceeds it. `network_allowlist`
    is a future-facing hook for restricting which hosts a `NETWORK` plugin
    may reach; it is parsed and surfaced here but not yet enforced (the
    plugin source-fetch module will read it).

    `capability_discovery` switches the loop from the single keyword-filtered
    intent pass to two-tier progressive disclosure (Tier-1 plugin index →
    model picks plugins → Tier-2 schemas → plan). Default `false` so the
    feature can be A/B'd on local models without a flag-day; see
    `specs/development/capability-discovery.md`. `KeywordCapabilityFilter`
    remains the implementation when this is off.

    `discovery_capability_threshold` is the skip-when-trivial cut-off: when
    discovery is on but the registry exposes this many *or fewer*
    user-facing capabilities, the whole menu already fits without
    truncation, so the extra discovery round-trip buys nothing and is
    skipped (the keyword path runs instead). The default mirrors
    `KeywordCapabilityFilter`'s `top_k` (8) — at or below it the keyword
    filter would surface everything anyway, so the cut-off is *derived*
    from existing behaviour, not a guessed constant. Operators tuning for
    a measured token budget on a large install override it here.
    """

    max_blast_radius: BlastRadius = BlastRadius.NETWORK
    network_allowlist: tuple[str, ...] = ()
    capability_discovery: bool = False
    discovery_capability_threshold: int = 8


@dataclass(frozen=True, slots=True)
class ModelConfig:
    """Model provider selection — Ollama / Qwen3 8B is the documented default.

    `timeout_seconds` bounds a single model call. 60s is enough for warm
    requests; cold-load on a LAN host can easily exceed that, so the
    field is exposed in config.toml for users on slower setups.
    """

    provider: str = 'ollama'
    model: str = 'qwen3:8b'
    host: str = 'http://localhost:11434'
    timeout_seconds: float = 60.0
    # `think` controls Ollama's chain-of-thought emission for models that
    # support it (qwen3, deepseek-r1, etc). Defaults to false because
    # butter's two-call planning loop (intent + synthesis) pays the CoT
    # cost twice per turn; live-REPL testing on 2026-05-14 saw 2-3x
    # latency on qwen3:8b with thinking enabled. Set true if a particular
    # model produces worse plans without CoT.
    think: bool = False


@dataclass(frozen=True, slots=True)
class StorageConfig:
    """Storage provider selection — local SQLite is the documented default."""

    provider: str = 'sqlite'
    path: str = '~/.butter-agent/butter.db'


@dataclass(frozen=True, slots=True)
class PluginSource:
    """One declared external plugin source, e.g. `github.com/user/repo@v0.2.1`.

    `repo` is the source identifier as written (host/path with no `@ref`).
    `ref` is the explicit pinned reference (tag or commit SHA). Branch
    refs like `main`, `master`, `HEAD` are rejected at parse time — the
    scope's "pinned refs" rule is enforced here, not at fetch time.

    `config` is the operator-supplied, plugin-scoped settings table from
    the `[[plugin]]` entry. Core treats the values as opaque — only the
    receiving plugin knows its own keys (mirrors how manifests keep
    input/output schemas plugin-side). Closed over by core and surfaced to
    exactly one plugin via `PluginContext.config` (invariant #6).
    """

    repo: str
    ref: str
    config: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class PluginPath:
    """A locally-resolved plugin directory — for development workflows.

    Bypasses the fetcher entirely; the loader uses the directory as-is.
    Mutually exclusive with `PluginSource` within a single `[[plugin]]`
    entry. `path` may use `~` and is left unexpanded here — the loader
    resolves it just before reading the manifest.

    `config` carries the same plugin-scoped settings table as
    `PluginSource.config` — see that docstring.
    """

    path: str
    config: Mapping[str, object] = field(default_factory=dict)


PluginDeclaration = PluginSource | PluginPath


@dataclass(frozen=True, slots=True)
class Config:
    """Top-level parsed config."""

    core: CoreConfig = field(default_factory=CoreConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    plugins: tuple[PluginDeclaration, ...] = ()


# --- Pin enforcement ---------------------------------------------------------

_BRANCH_REFS_REJECTED: Final[frozenset[str]] = frozenset({'main', 'master', 'head'})


# --- Loader ------------------------------------------------------------------


def load_config(toml_text: str) -> Config:
    """Parse `toml_text` into a validated `Config`.

    An empty document is valid and yields the documented defaults — that
    is the first-run experience promised by the scope.

    Raises:
        ConfigError: If TOML is malformed, a section has the wrong shape,
            an enum value is unknown, or a plugin source omits / uses a
            non-pinned ref.
    """
    try:
        data = tomllib.loads(toml_text)
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f'invalid TOML: {exc}') from exc

    if not isinstance(data, dict):
        raise ConfigError('top-level document must be a table')

    return Config(
        core=_parse_core(_section(data, 'core')),
        model=_parse_model(_section(data, 'model')),
        storage=_parse_storage(_section(data, 'storage')),
        plugins=_parse_plugins(data.get('plugin', [])),
    )


# --- Writer ------------------------------------------------------------------


def dump_config(config: Config) -> str:
    """Serialize `config` back to TOML text.

    The output round-trips through `load_config` — `load_config(dump_config(c)) == c`.
    Used by the `/configure` slash command to persist edits without losing
    sections (notably `[[plugin]]`) that the interactive walkthrough does
    not touch.

    Only the fields modelled by `Config` are emitted; commentary in the
    shipped `config.toml` is intentionally not preserved.
    """
    lines: list[str] = []

    lines.append('[core]')
    lines.append(f'max_blast_radius = {_quote(config.core.max_blast_radius.value)}')
    lines.append(f'network_allowlist = {_quote_list(config.core.network_allowlist)}')
    lines.append(f'capability_discovery = {"true" if config.core.capability_discovery else "false"}')
    lines.append(f'discovery_capability_threshold = {config.core.discovery_capability_threshold}')
    lines.append('')

    lines.append('[model]')
    lines.append(f'provider = {_quote(config.model.provider)}')
    lines.append(f'model = {_quote(config.model.model)}')
    lines.append(f'host = {_quote(config.model.host)}')
    lines.append(f'timeout_seconds = {_format_number(config.model.timeout_seconds)}')
    lines.append(f'think = {"true" if config.model.think else "false"}')
    lines.append('')

    lines.append('[storage]')
    lines.append(f'provider = {_quote(config.storage.provider)}')
    lines.append(f'path = {_quote(config.storage.path)}')

    for plugin in config.plugins:
        lines.append('')
        lines.append('[[plugin]]')
        if isinstance(plugin, PluginSource):
            lines.append(f'source = {_quote(f"{plugin.repo}@{plugin.ref}")}')
        else:
            lines.append(f'path = {_quote(plugin.path)}')
        if plugin.config:
            lines.append(f'config = {_toml_inline_table(plugin.config)}')

    return '\n'.join(lines) + '\n'


def _toml_inline_table(table: Mapping[str, object]) -> str:
    """Render a mapping as a TOML inline table that round-trips losslessly."""
    if not table:
        return '{}'
    return '{ ' + ', '.join(f'{_toml_key(k)} = {_toml_value(v)}' for k, v in table.items()) + ' }'


# Bare TOML keys (no quoting needed). Anything else is emitted as a basic
# string key so arbitrary operator-chosen config keys still round-trip.
_BARE_KEY_RE: Final = re.compile(r'^[A-Za-z0-9_-]+$')


def _toml_key(key: str) -> str:
    return key if _BARE_KEY_RE.match(key) else _quote(key)


def _toml_value(value: object) -> str:
    """Serialise a parsed-TOML value back to TOML text.

    Only the types `_parse_plugin_config` admits reach here, so the final
    branch is unreachable in a correctly validated `Config` — it is a
    defensive guard, not a user-facing path. Floats use `repr` (not
    `_format_number`) so an integer-valued float like `2.0` keeps its
    decimal point and parses back as a float rather than an int.
    """
    if isinstance(value, bool):
        return 'true' if value else 'false'
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return repr(value)
    if isinstance(value, str):
        return _quote(value)
    if isinstance(value, list):
        return '[' + ', '.join(_toml_value(v) for v in value) + ']'
    if isinstance(value, dict):
        return _toml_inline_table(value)
    raise ConfigError(f'cannot serialise config value of type {type(value).__name__}')


def _format_number(value: float) -> str:
    """Render a number for TOML output without a redundant trailing '.0'."""
    if value.is_integer():
        return str(int(value))
    return repr(value)


def _quote(value: str) -> str:
    """Render a string as a TOML basic string.

    Escapes the subset that can appear in our config values (backslash,
    double quote, control whitespace). Sufficient for paths, URLs, model
    names, and blast-radius enum values — the only string fields we emit.
    """
    escaped = value.replace('\\', '\\\\').replace('"', '\\"').replace('\n', '\\n').replace('\r', '\\r').replace('\t', '\\t')
    return f'"{escaped}"'


def _quote_list(values: tuple[str, ...]) -> str:
    if not values:
        return '[]'
    return '[' + ', '.join(_quote(v) for v in values) + ']'


# --- Section parsers ---------------------------------------------------------


def _section(data: dict[str, object], name: str) -> dict[str, object]:
    section = data.get(name, {})
    if not isinstance(section, dict):
        raise ConfigError(f'[{name}] must be a table')
    return section


def _parse_core(section: dict[str, object]) -> CoreConfig:
    radius_raw = _optional_str(section, 'core.max_blast_radius', 'max_blast_radius')
    if radius_raw is None:
        max_radius = CoreConfig().max_blast_radius
    else:
        try:
            max_radius = BlastRadius(radius_raw)
        except ValueError as exc:
            valid = ', '.join(r.value for r in BlastRadius)
            raise ConfigError(
                f'core.max_blast_radius: invalid value {radius_raw!r} (expected one of: {valid})',
            ) from exc

    allowlist = _optional_string_list(section, 'core.network_allowlist', 'network_allowlist')
    defaults = CoreConfig()
    discovery = _optional_bool(section, 'core.capability_discovery', 'capability_discovery')
    threshold = _optional_non_negative_int(
        section,
        'core.discovery_capability_threshold',
        'discovery_capability_threshold',
    )
    return CoreConfig(
        max_blast_radius=max_radius,
        network_allowlist=allowlist,
        capability_discovery=discovery if discovery is not None else defaults.capability_discovery,
        discovery_capability_threshold=threshold if threshold is not None else defaults.discovery_capability_threshold,
    )


def _parse_model(section: dict[str, object]) -> ModelConfig:
    defaults = ModelConfig()
    timeout = _optional_positive_float(section, 'model.timeout_seconds', 'timeout_seconds')
    think = _optional_bool(section, 'model.think', 'think')
    return ModelConfig(
        provider=_optional_str(section, 'model.provider', 'provider') or defaults.provider,
        model=_optional_str(section, 'model.model', 'model') or defaults.model,
        host=_optional_str(section, 'model.host', 'host') or defaults.host,
        timeout_seconds=timeout if timeout is not None else defaults.timeout_seconds,
        think=think if think is not None else defaults.think,
    )


def _parse_storage(section: dict[str, object]) -> StorageConfig:
    defaults = StorageConfig()
    return StorageConfig(
        provider=_optional_str(section, 'storage.provider', 'provider') or defaults.provider,
        path=_optional_str(section, 'storage.path', 'path') or defaults.path,
    )


def _parse_plugins(raw: object) -> tuple[PluginDeclaration, ...]:
    if not isinstance(raw, list):
        raise ConfigError('[[plugin]] must be an array of tables')
    return tuple(_parse_plugin(idx, item) for idx, item in enumerate(raw))


def _parse_plugin(index: int, raw: object) -> PluginDeclaration:
    """Parse one [[plugin]] entry as either a remote source or a local path.

    Exactly one of `source` / `path` must be set. `source` is the
    production form (pinned ref required); `path` is the local-dev form
    (no ref — the caller iterates whatever is on disk).
    """
    if not isinstance(raw, dict):
        raise ConfigError(f'[[plugin]] entry #{index + 1} must be a table')

    has_source = 'source' in raw
    has_path = 'path' in raw
    if has_source and has_path:
        raise ConfigError(f'[[plugin]] entry #{index + 1}: set either `source` or `path`, not both')
    if not has_source and not has_path:
        raise ConfigError(f'[[plugin]] entry #{index + 1}: missing `source` (production) or `path` (local-dev)')

    plugin_config = _parse_plugin_config(index, raw.get('config', {}))

    if has_path:
        path = raw['path']
        if not isinstance(path, str) or not path:
            raise ConfigError(f'[[plugin]] entry #{index + 1}: `path` must be a non-empty string')
        return PluginPath(path=path, config=plugin_config)

    source = raw['source']
    if not isinstance(source, str) or not source:
        raise ConfigError(f'[[plugin]] entry #{index + 1}: `source` must be a non-empty string')
    if '@' not in source:
        raise ConfigError(
            f'[[plugin]] entry #{index + 1}: source {source!r} is missing a pinned ref (expected e.g. github.com/user/repo@v0.1.0)',
        )
    repo, ref = source.rsplit('@', 1)
    if not repo:
        raise ConfigError(f'[[plugin]] entry #{index + 1}: source {source!r} has empty repo before @')
    if not ref:
        raise ConfigError(f'[[plugin]] entry #{index + 1}: source {source!r} has empty ref after @')
    if ref.lower() in _BRANCH_REFS_REJECTED:
        raise ConfigError(
            f'[[plugin]] entry #{index + 1}: source {source!r} uses unpinned ref {ref!r} (use a tag or commit SHA, not a branch)',
        )
    return PluginSource(repo=repo, ref=ref, config=plugin_config)


# TOML scalar/container types core can both store and re-emit. Anything
# outside this set (notably `datetime`, which tomllib produces for bare
# date/time literals) is rejected at parse time so the round-trip
# invariant `load_config(dump_config(c)) == c` can never be silently
# violated by an un-serialisable plugin-config value.
_TOML_SCALARS: Final = (bool, int, float, str)


def _parse_plugin_config(index: int, raw: object) -> dict[str, object]:
    """Validate one `[[plugin]].config` table.

    Values are plugin-private — core does not interpret keys. It only
    enforces that every value is a type it can faithfully round-trip
    through `dump_config`, so the receiving plugin sees exactly what the
    operator wrote and the config writer never loses data.
    """
    if not isinstance(raw, dict):
        raise ConfigError(f'[[plugin]] entry #{index + 1}: `config` must be a table')

    def _check(value: object, path: str) -> None:
        # `bool` is a subclass of `int`; the explicit tuple membership is
        # fine here since we accept both — no bool/int confusion to guard.
        if isinstance(value, _TOML_SCALARS):
            return
        if isinstance(value, list):
            for i, item in enumerate(value):
                _check(item, f'{path}[{i}]')
            return
        if isinstance(value, dict):
            for k, v in value.items():
                _check(v, f'{path}.{k}')
            return
        raise ConfigError(
            f'[[plugin]] entry #{index + 1}: config {path}: unsupported value type {type(value).__name__} (allowed: string, integer, float, bool, array, table)',
        )

    for key, value in raw.items():
        _check(value, key)
    return dict(raw)


# --- Field helpers -----------------------------------------------------------


def _optional_positive_float(section: dict[str, object], path: str, key: str) -> float | None:
    """Parse an optional positive finite float (e.g. a timeout). Returns None when absent.

    Rejects zero, negative, and non-finite values. TOML 1.0 explicitly
    permits `nan`/`inf` as float literals, so without the `isfinite`
    guard a value of `nan` would slip past the positive check (every
    comparison with NaN is False) and surface later as a transport
    error. Booleans are rejected because Python treats them as ints.
    """
    if key not in section:
        return None
    value = section[key]
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ConfigError(f'{path}: expected number, got {type(value).__name__}')
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ConfigError(f'{path}: expected finite number, got {numeric}')
    if numeric <= 0:
        raise ConfigError(f'{path}: expected positive number, got {numeric}')
    return numeric


def _optional_non_negative_int(section: dict[str, object], path: str, key: str) -> int | None:
    """Parse an optional non-negative integer (e.g. a count threshold).

    Returns None when absent. Booleans are rejected up front because
    Python treats `bool` as an `int` subclass — without the guard a
    stray `true` would silently parse as `1`. Zero is permitted: a
    threshold of 0 means "never skip discovery for size" (only the
    ≤1-plugin skip remains).
    """
    if key not in section:
        return None
    value = section[key]
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f'{path}: expected integer, got {type(value).__name__}')
    if value < 0:
        raise ConfigError(f'{path}: expected non-negative integer, got {value}')
    return value


def _optional_str(section: dict[str, object], path: str, key: str) -> str | None:
    if key not in section:
        return None
    value = section[key]
    if not isinstance(value, str):
        raise ConfigError(f'{path}: expected string, got {type(value).__name__}')
    return value


def _optional_bool(section: dict[str, object], path: str, key: str) -> bool | None:
    if key not in section:
        return None
    value = section[key]
    # `isinstance(True, int)` is True in Python — guard explicitly so a
    # stray integer doesn't silently masquerade as a bool.
    if not isinstance(value, bool):
        raise ConfigError(f'{path}: expected bool, got {type(value).__name__}')
    return value


def _optional_string_list(section: dict[str, object], path: str, key: str) -> tuple[str, ...]:
    if key not in section:
        return ()
    value = section[key]
    if not isinstance(value, list):
        raise ConfigError(f'{path}: expected array of strings, got {type(value).__name__}')
    result: list[str] = []
    for idx, item in enumerate(value):
        if not isinstance(item, str):
            raise ConfigError(f'{path}[{idx}]: expected string, got {type(item).__name__}')
        result.append(item)
    return tuple(result)
