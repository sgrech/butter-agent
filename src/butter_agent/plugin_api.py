"""Public import surface for plugin authors.

Third-party plugins should import from this module — never from
`butter_agent.core.*` directly. The core layout is free to evolve;
this module is the stability contract.

What's re-exported:

- `Plugin` — the single-entrypoint Protocol every plugin satisfies.
- `BlastRadius` — the radius tiers a plugin may declare in its manifest.
- `Capability`, `PluginManifest` — value types for plugins that want to
  parse or inspect their own manifest at test time.
- `ManifestError`, `CapabilityNotFoundError` — the errors a plugin
  author can meaningfully react to.
- `parse_manifest` — convenience for tests that want to validate the
  plugin's shipped `manifest.toml` round-trips through butter's
  validation.

What's deliberately NOT re-exported:

- Anything from `core.loop` / `core.task_executor` / `core.context_manager`.
  Plugins don't drive the loop, don't see task plans, and don't assemble
  model context. The Protocol is the boundary.
- `RegistryBuilder` / `PluginRegistry`. Building the registry is butter's
  job, not the plugin's.

Stability promise: symbols re-exported here will not be removed or have
their meaning changed without a major version bump of butter-agent. New
symbols may be added; existing ones are frozen.
"""

from __future__ import annotations

from butter_agent.core.registry import (
    BlastRadius,
    Capability,
    CapabilityNotFoundError,
    ManifestError,
    Plugin,
    PluginManifest,
    parse_manifest,
)

__all__ = [
    'BlastRadius',
    'Capability',
    'CapabilityNotFoundError',
    'ManifestError',
    'Plugin',
    'PluginManifest',
    'parse_manifest',
]
