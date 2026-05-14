"""Tests for the `butter_agent.plugin_api` public surface.

These tests are intentionally narrow: they pin which names the module
re-exports, that the re-exports are the same objects as the originals
in `core.registry` (so isinstance checks across the boundary work), and
that the surface is sufficient to declare a minimal plugin without
touching `core.*` directly. Anything beyond that belongs in the
underlying module's tests.

The whole point of this module is stability — these tests fail loudly
if a future refactor accidentally drops or renames a re-export.
"""

from __future__ import annotations

from butter_agent import plugin_api
from butter_agent.core import registry as _registry


def test_reexports_identity() -> None:
    """Re-exports must be the same objects, not copies or wrappers."""
    assert plugin_api.Plugin is _registry.Plugin
    assert plugin_api.BlastRadius is _registry.BlastRadius
    assert plugin_api.Capability is _registry.Capability
    assert plugin_api.PluginManifest is _registry.PluginManifest
    assert plugin_api.ManifestError is _registry.ManifestError
    assert plugin_api.CapabilityNotFoundError is _registry.CapabilityNotFoundError
    assert plugin_api.parse_manifest is _registry.parse_manifest


def test_canonical_manifest_filename_is_pinned() -> None:
    """The loader and plugins must agree on one filename.

    Re-exporting a constant — and locking it down here — prevents the
    convention from drifting between butter-agent docs, the loader, and
    plugin author guides. If the canonical name ever needs to change,
    flip it deliberately in one place.
    """
    assert plugin_api.MANIFEST_FILENAME == 'manifest.toml'


def test_dunder_all_entries_are_resolvable() -> None:
    """Every name in `__all__` must actually be present on the module.

    Catches the drift where a symbol is removed from the module but its
    `__all__` entry is left behind — `from butter_agent.plugin_api import *`
    would then fail at runtime in plugin authors' code.
    """
    for name in plugin_api.__all__:
        assert hasattr(plugin_api, name), f'__all__ promises {name!r} but it is missing'


def test_minimal_plugin_can_be_declared_via_public_api_only() -> None:
    """A plugin can implement the contract using only `plugin_api` imports.

    This is the smoke test that the surface is *sufficient* — if a plugin
    author would have to reach into `core.*` to get something the loader
    will eventually want, that gap surfaces here.
    """

    class MinimalPlugin:
        async def execute(
            self,
            capability: str,
            inputs: dict[str, object],
            context: plugin_api.PluginContext,
        ) -> dict[str, object]:
            del inputs, context
            return {'echo': capability}

    # The Protocol is structural — implementing `execute` is enough.
    plugin: plugin_api.Plugin = MinimalPlugin()
    assert plugin is not None  # Type-check is the assertion; the runtime call would await.


def test_parse_manifest_round_trips_minimal_manifest() -> None:
    """Plugin authors need to validate their shipped `manifest.toml` in tests.

    Re-exporting `parse_manifest` lets them do that without importing
    butter's internals.
    """
    manifest_text = """
[plugin]
name = "example"
version = "0.1.0"
entrypoint = "example:Plugin"
blast_radius = "read-only"

[[capability]]
name = "ping"
description = "Returns pong."
input_schema = {}
output_schema = { reply = "string" }
"""
    manifest = plugin_api.parse_manifest(manifest_text)
    assert manifest.name == 'example'
    assert manifest.blast_radius is plugin_api.BlastRadius.READ_ONLY
    assert manifest.capability('ping').description == 'Returns pong.'
