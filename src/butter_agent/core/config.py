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
import tomllib
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
    """Core-side policy: the blast-radius ceiling and the network allowlist.

    `max_blast_radius` is the invariant-#7 ceiling — `RegistryBuilder`
    rejects any plugin whose declared radius exceeds it. `network_allowlist`
    is a future-facing hook for restricting which hosts a `NETWORK` plugin
    may reach; it is parsed and surfaced here but not yet enforced (the
    plugin source-fetch module will read it).
    """

    max_blast_radius: BlastRadius = BlastRadius.NETWORK
    network_allowlist: tuple[str, ...] = ()


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
    """

    repo: str
    ref: str


@dataclass(frozen=True, slots=True)
class Config:
    """Top-level parsed config."""

    core: CoreConfig = field(default_factory=CoreConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    plugins: tuple[PluginSource, ...] = ()


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
    lines.append('')

    lines.append('[model]')
    lines.append(f'provider = {_quote(config.model.provider)}')
    lines.append(f'model = {_quote(config.model.model)}')
    lines.append(f'host = {_quote(config.model.host)}')
    lines.append(f'timeout_seconds = {_format_number(config.model.timeout_seconds)}')
    lines.append('')

    lines.append('[storage]')
    lines.append(f'provider = {_quote(config.storage.provider)}')
    lines.append(f'path = {_quote(config.storage.path)}')

    for source in config.plugins:
        lines.append('')
        lines.append('[[plugin]]')
        lines.append(f'source = {_quote(f"{source.repo}@{source.ref}")}')

    return '\n'.join(lines) + '\n'


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
    return CoreConfig(max_blast_radius=max_radius, network_allowlist=allowlist)


def _parse_model(section: dict[str, object]) -> ModelConfig:
    defaults = ModelConfig()
    timeout = _optional_positive_float(section, 'model.timeout_seconds', 'timeout_seconds')
    return ModelConfig(
        provider=_optional_str(section, 'model.provider', 'provider') or defaults.provider,
        model=_optional_str(section, 'model.model', 'model') or defaults.model,
        host=_optional_str(section, 'model.host', 'host') or defaults.host,
        timeout_seconds=timeout if timeout is not None else defaults.timeout_seconds,
    )


def _parse_storage(section: dict[str, object]) -> StorageConfig:
    defaults = StorageConfig()
    return StorageConfig(
        provider=_optional_str(section, 'storage.provider', 'provider') or defaults.provider,
        path=_optional_str(section, 'storage.path', 'path') or defaults.path,
    )


def _parse_plugins(raw: object) -> tuple[PluginSource, ...]:
    if not isinstance(raw, list):
        raise ConfigError('[[plugin]] must be an array of tables')
    return tuple(_parse_plugin(idx, item) for idx, item in enumerate(raw))


def _parse_plugin(index: int, raw: object) -> PluginSource:
    if not isinstance(raw, dict):
        raise ConfigError(f'[[plugin]] entry #{index + 1} must be a table')
    source = raw.get('source')
    if not isinstance(source, str) or not source:
        raise ConfigError(f'[[plugin]] entry #{index + 1}: missing or empty string `source`')

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
    return PluginSource(repo=repo, ref=ref)


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


def _optional_str(section: dict[str, object], path: str, key: str) -> str | None:
    if key not in section:
        return None
    value = section[key]
    if not isinstance(value, str):
        raise ConfigError(f'{path}: expected string, got {type(value).__name__}')
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
