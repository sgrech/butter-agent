"""Plugin registry — manifest validation, frozen lookup, blast-radius gating.

Four invariants live here:

- #2 The plugin registry is **immutable after startup**. Plugins are added to a
  `RegistryBuilder`, which atomically produces a frozen `PluginRegistry`. The
  registry exposes lookup only; there is no `register` method on it.
- #6 Plugins cannot read each other's state. The registry hands out plugin
  references on demand (by name) and never wires them together; cross-plugin
  data only flows through the task executor's variable pool. Plugin-to-plugin
  calls are restricted to capabilities flagged `internal=True`, declared
  ahead-of-time in the caller's manifest `requires` list, and audited here.
- #7 Blast radius can only be **restricted** by core config, never **expanded**
  by a plugin's manifest. The builder rejects any plugin whose declared
  blast-radius exceeds the core config's allowed maximum, and also rejects any
  plugin whose declared radius does not cover the transitive radius of the
  internal capabilities it can reach via `requires`.
- #1 The plugin-call graph is a DAG. Cycles in `requires` are rejected at
  build, so plugin initialization order is deterministic.

What this module does NOT do:

- Fetch plugin source from git URLs / pinned refs — that is a separate I/O
  concern handled by a future `plugin_source` module. The registry accepts
  already-instantiated plugins paired with their parsed manifests.
- Execute capabilities — the executor module owns that.
- Inject gates — the executor owns gate enforcement (invariant #5).
"""

from __future__ import annotations

import re
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

    `internal` capabilities are invisible to the planner and may only be invoked
    plugin-to-plugin via `PluginContext.call`. They never appear in a `TaskPlan`
    step; the executor rejects plans that name them.
    """

    name: str
    description: str
    input_schema: dict[str, object]
    output_schema: dict[str, object]
    internal: bool = False


@dataclass(frozen=True, slots=True)
class PluginManifest:
    """Parsed and validated manifest.toml contents.

    `requires` is a tuple of fully-qualified `plugin.capability` references —
    the internal capabilities this plugin needs at runtime. The builder
    validates every entry resolves to an `internal=True` capability and that
    the resulting dependency graph is acyclic.
    """

    name: str
    version: str
    blast_radius: BlastRadius
    entrypoint: str
    capabilities: tuple[Capability, ...]
    requires: tuple[str, ...] = ()

    def capability(self, name: str) -> Capability:
        """Look up a capability by name, raising if it does not exist."""
        for cap in self.capabilities:
            if cap.name == name:
                return cap
        raise CapabilityNotFoundError(f'plugin {self.name!r}: no capability {name!r}')


# --- Plugin contract ---------------------------------------------------------


class PluginContext(Protocol):
    """Per-invocation handle the executor injects into a plugin's `execute`.

    Carries the caller's identity (closed over by the executor) and exposes a
    restricted `call` method for plugin-to-plugin invocation against `internal`
    capabilities the caller declared in its manifest `requires`.

    A plugin never constructs a `PluginContext` directly — the executor builds
    one per `execute` call and passes it in. The plugin cannot forge identity
    by passing a different context to another plugin.
    """

    async def call(self, capability: str, inputs: dict[str, object]) -> dict[str, object]:
        """Invoke an internal capability declared in the caller's `requires`.

        The `capability` argument is the fully-qualified `plugin.capability`
        reference, identical to how it appears in `requires`. Raises if the
        target is not in the caller's permitted set — that's a plugin-author
        bug, not a runtime failure.
        """
        ...


class Plugin(Protocol):
    """The single-entrypoint contract every plugin satisfies.

    Implementations may be bundled built-ins (e.g. `notes`, `reminders`,
    `search`) or third-party repos cloned at startup. Either way the registry
    only ever calls `execute`, and the executor always supplies a fresh
    `context` carrying the caller's identity.
    """

    async def execute(
        self,
        capability: str,
        inputs: dict[str, object],
        context: PluginContext,
    ) -> dict[str, object]: ...


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


class RequiresValidationError(RegistryError):
    """Raised when a manifest's `requires` references an unknown or non-internal capability."""


class RequiresCycleError(RegistryError):
    """Raised when `requires` declarations form a cycle across plugins."""


class TransitiveBlastRadiusViolation(RegistryError):
    """Raised when a plugin's declared blast radius does not cover the radius reachable via `requires`."""


# --- Manifest parsing --------------------------------------------------------


# Plugin and capability names are surfaced verbatim into the model prompt
# (e.g. `plugin "clock" capability "now"`) and into `failure_reason`
# strings. Restricting them to a safe identifier charset at parse time
# means downstream renderers can interpolate without escaping —
# fix-at-the-trust-boundary, per PR #17 review.
_IDENTIFIER_RE = re.compile(r'^[a-z][a-z0-9_]*$')
_IDENTIFIER_HINT = 'must match [a-z][a-z0-9_]* (lowercase letter, then lowercase letters/digits/underscores)'

# `requires` entries are written as `plugin_name.capability_name` — two
# identifiers separated by a single dot. Anything else is a manifest bug
# caught at parse time.
_REQUIRES_REF_RE = re.compile(r'^([a-z][a-z0-9_]*)\.([a-z][a-z0-9_]*)$')


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
    if not _IDENTIFIER_RE.match(name):
        raise ManifestError(f'plugin name {name!r}: {_IDENTIFIER_HINT}')
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

    requires = _parse_requires(name, plugin_section.get('requires', []))

    return PluginManifest(
        name=name,
        version=version,
        blast_radius=blast_radius,
        entrypoint=entrypoint,
        capabilities=capabilities,
        requires=requires,
    )


def _parse_requires(plugin_name: str, raw: object) -> tuple[str, ...]:
    if not isinstance(raw, list):
        raise ManifestError(f'plugin {plugin_name!r}: requires must be a list of strings')
    seen: set[str] = set()
    out: list[str] = []
    for entry in raw:
        if not isinstance(entry, str) or not _REQUIRES_REF_RE.match(entry):
            raise ManifestError(
                f'plugin {plugin_name!r}: requires entry {entry!r} must be of the form plugin.capability (both identifiers)',
            )
        if entry in seen:
            raise ManifestError(f'plugin {plugin_name!r}: duplicate requires entry {entry!r}')
        seen.add(entry)
        out.append(entry)
    return tuple(out)


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
    internal = raw.get('internal', False)
    if not isinstance(name, str) or not name:
        raise ManifestError(f'plugin {plugin_name!r}: capability missing string name')
    if not _IDENTIFIER_RE.match(name):
        raise ManifestError(f'plugin {plugin_name!r}: capability name {name!r}: {_IDENTIFIER_HINT}')
    if not isinstance(description, str) or not description:
        raise ManifestError(f'plugin {plugin_name!r}: capability {name!r} missing description')
    if not isinstance(input_schema, dict):
        raise ManifestError(f'plugin {plugin_name!r}: capability {name!r} input_schema must be a table')
    if not isinstance(output_schema, dict):
        raise ManifestError(f'plugin {plugin_name!r}: capability {name!r} output_schema must be a table')
    if not isinstance(internal, bool):
        raise ManifestError(f'plugin {plugin_name!r}: capability {name!r} internal must be a boolean')
    return Capability(
        name=name,
        description=description,
        input_schema=dict(input_schema),
        output_schema=dict(output_schema),
        internal=internal,
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
        """Freeze the builder and return the immutable registry.

        Three cross-plugin checks run here, after every plugin has been
        registered, because each needs the full graph:

        - every `requires` entry resolves to an `internal=True` capability;
        - the dependency graph induced by `requires` is acyclic;
        - each plugin's declared blast radius covers the transitive maximum
          radius reachable through `requires`.

        Any failure aborts the build — invariant #2 means a partially valid
        registry can never be exposed at runtime.
        """
        if self._built:
            raise RegistryFrozenError('build() may only be called once')
        self._validate_requires_targets()
        self._validate_no_cycles()
        self._validate_transitive_blast_radius()
        self._built = True
        return PluginRegistry(self._entries)

    def _validate_requires_targets(self) -> None:
        """Every `requires` entry must point at an existing `internal=True` capability."""
        for entry in self._entries.values():
            for ref in entry.manifest.requires:
                target_plugin, target_capability = ref.split('.', 1)
                if target_plugin == entry.manifest.name:
                    raise RequiresValidationError(
                        f'plugin {entry.manifest.name!r}: requires entry {ref!r} points at itself',
                    )
                target_entry = self._entries.get(target_plugin)
                if target_entry is None:
                    raise RequiresValidationError(
                        f'plugin {entry.manifest.name!r}: requires entry {ref!r} names unknown plugin {target_plugin!r}',
                    )
                try:
                    cap = target_entry.manifest.capability(target_capability)
                except CapabilityNotFoundError as exc:
                    raise RequiresValidationError(
                        f'plugin {entry.manifest.name!r}: requires entry {ref!r}: {exc}',
                    ) from exc
                if not cap.internal:
                    raise RequiresValidationError(
                        f'plugin {entry.manifest.name!r}: requires entry {ref!r} targets a non-internal capability (only internal=true capabilities are callable plugin-to-plugin)',
                    )

    def _validate_no_cycles(self) -> None:
        """Topo-sort plugins by `requires`; reject if cycles remain."""
        # in_degree counts how many *unresolved* requires each plugin still has.
        # An edge from caller→target means caller depends on target. A plugin
        # with zero outgoing requires has in_degree 0 in the reverse graph and
        # is the natural "leaf to process first".
        dependents: dict[str, set[str]] = {name: set() for name in self._entries}
        unresolved: dict[str, int] = {name: 0 for name in self._entries}
        for entry in self._entries.values():
            caller = entry.manifest.name
            targets = {ref.split('.', 1)[0] for ref in entry.manifest.requires}
            unresolved[caller] = len(targets)
            for target in targets:
                dependents[target].add(caller)

        ready: list[str] = [name for name, count in unresolved.items() if count == 0]
        resolved = 0
        while ready:
            current = ready.pop()
            resolved += 1
            for dependent in dependents[current]:
                unresolved[dependent] -= 1
                if unresolved[dependent] == 0:
                    ready.append(dependent)

        if resolved != len(self._entries):
            stuck = sorted(name for name, count in unresolved.items() if count > 0)
            raise RequiresCycleError(
                f'requires graph contains a cycle involving: {", ".join(stuck)}',
            )

    def _validate_transitive_blast_radius(self) -> None:
        """A plugin's declared radius must cover the radius reachable via `requires`."""
        for entry in self._entries.values():
            transitive = self._transitive_radius(entry.manifest.name)
            if _RADIUS_RANK[transitive] > _RADIUS_RANK[entry.manifest.blast_radius]:
                raise TransitiveBlastRadiusViolation(
                    f'plugin {entry.manifest.name!r} declares blast_radius={entry.manifest.blast_radius.value} but transitively reaches {transitive.value} through requires',
                )

    def _transitive_radius(self, plugin_name: str) -> BlastRadius:
        """Walk `requires` from `plugin_name` and return the max radius reachable.

        The plugin's own declared radius is included so the comparison in
        `_validate_transitive_blast_radius` is `transitive ≤ declared` rather
        than `max(declared, transitive) ≤ declared`. Cycles are already
        rejected by `_validate_no_cycles`, so a plain DFS terminates.
        """
        visited: set[str] = set()
        max_radius = self._entries[plugin_name].manifest.blast_radius

        def walk(current: str) -> None:
            nonlocal max_radius
            if current in visited:
                return
            visited.add(current)
            manifest = self._entries[current].manifest
            if _RADIUS_RANK[manifest.blast_radius] > _RADIUS_RANK[max_radius]:
                max_radius = manifest.blast_radius
            for ref in manifest.requires:
                walk(ref.split('.', 1)[0])

        walk(plugin_name)
        return max_radius
