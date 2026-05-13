"""Plugin registry — manifest validation, frozen lookup, blast-radius gating.

Three invariants live here:

- #2 The plugin registry is **immutable after startup**. Plugins are added to a
  `RegistryBuilder`, which atomically produces a frozen `PluginRegistry`. The
  registry exposes lookup only; there is no `register` method on it.
- #6 Plugins cannot read each other's state. The registry hands out plugin
  references on demand (by name) and never wires them together; cross-plugin
  data only flows through the task executor's variable pool.
- #7 Blast radius can only be **restricted** by core config, never **expanded**
  by a plugin's manifest. The builder rejects any plugin whose declared
  blast-radius exceeds the core config's allowed maximum.

What this module does NOT do:

- Fetch plugin source from git URLs / pinned refs — that is a separate I/O
  concern handled by a future `plugin_source` module. The registry accepts
  already-instantiated plugins paired with their parsed manifests.
- Execute capabilities — the executor module owns that.
- Inject gates — the executor owns gate enforcement (invariant #5).
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from enum import StrEnum
from typing import Final, Protocol

# --- Blast radius ------------------------------------------------------------


class BlastRadius(StrEnum):
    """Declared scope of side effects a plugin may produce.

    Ordering matters: each tier is strictly broader than the previous one, so a
    config that permits `LOCAL_WRITE` also permits `READ_ONLY`. The integer rank
    is used for the "can-restrict-not-expand" comparison.
    """

    READ_ONLY = 'read-only'
    LOCAL_WRITE = 'local-write'
    NETWORK = 'network'


_RADIUS_RANK: Final[dict[BlastRadius, int]] = {
    BlastRadius.READ_ONLY: 0,
    BlastRadius.LOCAL_WRITE: 1,
    BlastRadius.NETWORK: 2,
}


def radius_permits(allowed: BlastRadius, requested: BlastRadius) -> bool:
    """Return True if a plugin requesting `requested` is permitted under config `allowed`."""
    return _RADIUS_RANK[requested] <= _RADIUS_RANK[allowed]


# --- Value types -------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Capability:
    """One capability declared by a plugin manifest.

    `input_schema` / `output_schema` are kept as plain dicts; the task executor
    validates step inputs against them. The registry does not interpret them.
    """

    name: str
    description: str
    input_schema: dict[str, object]
    output_schema: dict[str, object]


@dataclass(frozen=True, slots=True)
class PluginManifest:
    """Parsed and validated manifest.toml contents."""

    name: str
    version: str
    blast_radius: BlastRadius
    entrypoint: str
    capabilities: tuple[Capability, ...]

    def capability(self, name: str) -> Capability:
        """Look up a capability by name, raising if it does not exist."""
        for cap in self.capabilities:
            if cap.name == name:
                return cap
        raise CapabilityNotFoundError(f'plugin {self.name!r}: no capability {name!r}')


# --- Plugin contract ---------------------------------------------------------


class Plugin(Protocol):
    """The single-entrypoint contract every plugin satisfies.

    Implementations may be bundled built-ins (e.g. `notes`, `reminders`,
    `search`) or third-party repos cloned at startup. Either way the registry
    only ever calls `execute`.
    """

    async def execute(self, capability: str, inputs: dict[str, object]) -> dict[str, object]: ...


# --- Errors ------------------------------------------------------------------


class RegistryError(Exception):
    """Base class for registry-related errors."""


class ManifestError(RegistryError):
    """Raised when a manifest.toml fails to parse or validate."""


class BlastRadiusViolation(RegistryError):
    """Raised when a plugin's declared blast radius exceeds the configured ceiling."""


class DuplicatePluginError(RegistryError):
    """Raised when two plugins claim the same name."""


class RegistryFrozenError(RegistryError):
    """Raised on any attempt to mutate the registry after it is built."""


class PluginNotFoundError(RegistryError):
    """Raised when a lookup names a plugin not in the registry."""


class CapabilityNotFoundError(RegistryError):
    """Raised when a lookup names a capability not declared by the plugin."""


# --- Manifest parsing --------------------------------------------------------


def parse_manifest(toml_text: str) -> PluginManifest:
    """Parse a manifest.toml document and validate its shape.

    Args:
        toml_text: Raw contents of a manifest.toml file.

    Returns:
        A validated `PluginManifest`.

    Raises:
        ManifestError: If required fields are missing, types are wrong, or
            blast_radius is not one of the known tiers.
    """
    try:
        data = tomllib.loads(toml_text)
    except tomllib.TOMLDecodeError as exc:
        raise ManifestError(f'invalid TOML: {exc}') from exc

    plugin_section = _require_section(data, 'plugin')
    name = _require_str(plugin_section, 'plugin.name')
    version = _require_str(plugin_section, 'plugin.version', key='version')
    entrypoint = _require_str(plugin_section, 'plugin.entrypoint', key='entrypoint')
    radius_raw = _require_str(plugin_section, 'plugin.blast_radius', key='blast_radius')

    try:
        blast_radius = BlastRadius(radius_raw)
    except ValueError as exc:
        valid = ', '.join(r.value for r in BlastRadius)
        raise ManifestError(f'plugin {name!r}: invalid blast_radius {radius_raw!r} (expected one of: {valid})') from exc

    raw_caps = data.get('capability', [])
    if not isinstance(raw_caps, list) or not raw_caps:
        raise ManifestError(f'plugin {name!r}: must declare at least one [[capability]]')

    capabilities = tuple(_parse_capability(name, raw_cap) for raw_cap in raw_caps)
    _ensure_unique_capability_names(name, capabilities)

    return PluginManifest(
        name=name,
        version=version,
        blast_radius=blast_radius,
        entrypoint=entrypoint,
        capabilities=capabilities,
    )


def _require_section(data: dict[str, object], path: str) -> dict[str, object]:
    section = data.get(path)
    if not isinstance(section, dict):
        raise ManifestError(f'missing [{path}] section')
    return section


def _require_str(section: dict[str, object], path: str, key: str = 'name') -> str:
    value = section.get(key)
    if not isinstance(value, str) or not value:
        raise ManifestError(f'missing or empty string: {path}')
    return value


def _ensure_unique_capability_names(plugin_name: str, capabilities: tuple[Capability, ...]) -> None:
    seen: set[str] = set()
    for cap in capabilities:
        if cap.name in seen:
            raise ManifestError(f'plugin {plugin_name!r}: duplicate capability name {cap.name!r}')
        seen.add(cap.name)


def _parse_capability(plugin_name: str, raw: object) -> Capability:
    if not isinstance(raw, dict):
        raise ManifestError(f'plugin {plugin_name!r}: each [[capability]] must be a table')
    name = raw.get('name')
    description = raw.get('description')
    input_schema = raw.get('input_schema', {})
    output_schema = raw.get('output_schema', {})
    if not isinstance(name, str) or not name:
        raise ManifestError(f'plugin {plugin_name!r}: capability missing string name')
    if not isinstance(description, str) or not description:
        raise ManifestError(f'plugin {plugin_name!r}: capability {name!r} missing description')
    if not isinstance(input_schema, dict):
        raise ManifestError(f'plugin {plugin_name!r}: capability {name!r} input_schema must be a table')
    if not isinstance(output_schema, dict):
        raise ManifestError(f'plugin {plugin_name!r}: capability {name!r} output_schema must be a table')
    return Capability(
        name=name,
        description=description,
        input_schema=dict(input_schema),
        output_schema=dict(output_schema),
    )


# --- Registry ----------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RegisteredPlugin:
    """A plugin instance paired with its validated manifest."""

    manifest: PluginManifest
    plugin: Plugin


class PluginRegistry:
    """The frozen, immutable runtime view of all loaded plugins.

    Built once at startup by `RegistryBuilder.build()` and never modified
    after — invariant #2. Lookups are pure functions on the snapshot.
    """

    def __init__(self, entries: dict[str, RegisteredPlugin]) -> None:
        # Store as a plain dict — mutation is prevented by not exposing it.
        self._entries: Final[dict[str, RegisteredPlugin]] = dict(entries)

    def names(self) -> tuple[str, ...]:
        """Return all registered plugin names in registration order."""
        return tuple(self._entries.keys())

    def get(self, plugin_name: str) -> RegisteredPlugin:
        """Look up a plugin by name."""
        try:
            return self._entries[plugin_name]
        except KeyError as exc:
            raise PluginNotFoundError(f'no plugin registered as {plugin_name!r}') from exc

    def capability(self, plugin_name: str, capability_name: str) -> Capability:
        """Look up a capability on a plugin."""
        return self.get(plugin_name).manifest.capability(capability_name)

    def __contains__(self, plugin_name: object) -> bool:
        return isinstance(plugin_name, str) and plugin_name in self._entries

    def __len__(self) -> int:
        return len(self._entries)


class RegistryBuilder:
    """One-shot builder that produces a frozen `PluginRegistry`.

    Construction phase enforces:
    - blast-radius ceiling from core config (invariant #7)
    - unique plugin names (per-name handoff prevents cross-plugin state leaks,
      invariant #6)

    After `build()` is called, the builder rejects further registrations —
    that guards against accidental late-binding that would break invariant #2.
    """

    def __init__(self, *, max_blast_radius: BlastRadius) -> None:
        self._max_blast_radius = max_blast_radius
        self._entries: dict[str, RegisteredPlugin] = {}
        self._built = False

    def register(self, manifest: PluginManifest, plugin: Plugin) -> None:
        """Add a plugin to the in-progress registry."""
        if self._built:
            raise RegistryFrozenError('registry has already been built')
        if not radius_permits(self._max_blast_radius, manifest.blast_radius):
            raise BlastRadiusViolation(
                f'plugin {manifest.name!r} declares blast_radius={manifest.blast_radius.value} but core config permits at most {self._max_blast_radius.value}',
            )
        if manifest.name in self._entries:
            raise DuplicatePluginError(f'plugin {manifest.name!r} already registered')
        self._entries[manifest.name] = RegisteredPlugin(manifest=manifest, plugin=plugin)

    def build(self) -> PluginRegistry:
        """Freeze the builder and return the immutable registry."""
        if self._built:
            raise RegistryFrozenError('build() may only be called once')
        self._built = True
        return PluginRegistry(self._entries)
