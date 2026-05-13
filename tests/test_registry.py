"""Tests for the plugin registry.

Covers manifest parsing, blast-radius gating (invariant #7), freeze semantics
(invariant #2), lookup, and the most likely manifest-rejection cases.
"""

from __future__ import annotations

import pytest

from butter_agent.core.registry import (
    BlastRadius,
    BlastRadiusViolation,
    CapabilityNotFoundError,
    DuplicatePluginError,
    ManifestError,
    PluginNotFoundError,
    PluginRegistry,
    RegistryBuilder,
    RegistryFrozenError,
    parse_manifest,
    radius_permits,
)


# A minimal stub plugin used to populate the registry in tests.
class _StubPlugin:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, object]]] = []

    async def execute(self, capability: str, inputs: dict[str, object]) -> dict[str, object]:
        self.calls.append((capability, inputs))
        return {'ok': True}


def _valid_toml(name: str = 'notes', radius: str = 'local-write') -> str:
    return f"""
[plugin]
name = "{name}"
version = "0.1.0"
blast_radius = "{radius}"
entrypoint = "main:Plugin"

[[capability]]
name = "create"
description = "Create a new note"
input_schema = {{ title = "string", body = "string" }}
output_schema = {{ id = "integer" }}

[[capability]]
name = "list"
description = "List all notes"
input_schema = {{}}
output_schema = {{ notes = "array" }}
"""


# --- Manifest parsing -------------------------------------------------------


def test_parse_manifest_happy_path() -> None:
    manifest = parse_manifest(_valid_toml())
    assert manifest.name == 'notes'
    assert manifest.version == '0.1.0'
    assert manifest.blast_radius is BlastRadius.LOCAL_WRITE
    assert manifest.entrypoint == 'main:Plugin'
    assert len(manifest.capabilities) == 2
    assert manifest.capabilities[0].name == 'create'
    assert manifest.capability('list').description == 'List all notes'


def test_parse_manifest_rejects_unknown_blast_radius() -> None:
    bad = _valid_toml().replace('local-write', 'cosmic')
    with pytest.raises(ManifestError, match='invalid blast_radius'):
        parse_manifest(bad)


def test_parse_manifest_requires_at_least_one_capability() -> None:
    toml_text = """
[plugin]
name = "empty"
version = "0.1.0"
blast_radius = "read-only"
entrypoint = "main:Plugin"
"""
    with pytest.raises(ManifestError, match='at least one'):
        parse_manifest(toml_text)


def test_parse_manifest_rejects_capability_without_description() -> None:
    toml_text = """
[plugin]
name = "bad"
version = "0.1.0"
blast_radius = "read-only"
entrypoint = "main:Plugin"

[[capability]]
name = "do_thing"
input_schema = {}
output_schema = {}
"""
    with pytest.raises(ManifestError, match='missing description'):
        parse_manifest(toml_text)


def test_parse_manifest_rejects_invalid_toml() -> None:
    with pytest.raises(ManifestError, match='invalid TOML'):
        parse_manifest('not = valid = toml')


def test_parse_manifest_rejects_missing_plugin_section() -> None:
    with pytest.raises(ManifestError, match=r'missing \[plugin\]'):
        parse_manifest('[other]\nname = "x"\n')


def test_capability_not_found_raises() -> None:
    manifest = parse_manifest(_valid_toml())
    with pytest.raises(CapabilityNotFoundError, match="no capability 'missing'"):
        manifest.capability('missing')


# --- Blast radius gating ----------------------------------------------------


def test_radius_permits_orders_tiers_correctly() -> None:
    assert radius_permits(BlastRadius.NETWORK, BlastRadius.READ_ONLY)
    assert radius_permits(BlastRadius.LOCAL_WRITE, BlastRadius.READ_ONLY)
    assert radius_permits(BlastRadius.READ_ONLY, BlastRadius.READ_ONLY)
    assert not radius_permits(BlastRadius.READ_ONLY, BlastRadius.LOCAL_WRITE)
    assert not radius_permits(BlastRadius.LOCAL_WRITE, BlastRadius.NETWORK)


def test_builder_rejects_blast_radius_above_ceiling() -> None:
    builder = RegistryBuilder(max_blast_radius=BlastRadius.LOCAL_WRITE)
    manifest = parse_manifest(_valid_toml(name='searx', radius='network'))

    with pytest.raises(BlastRadiusViolation, match='searx'):
        builder.register(manifest, _StubPlugin())


# --- Builder + registry -----------------------------------------------------


def test_builder_builds_a_frozen_registry() -> None:
    builder = RegistryBuilder(max_blast_radius=BlastRadius.NETWORK)
    manifest = parse_manifest(_valid_toml())
    plugin = _StubPlugin()
    builder.register(manifest, plugin)

    registry = builder.build()
    assert isinstance(registry, PluginRegistry)
    assert len(registry) == 1
    assert 'notes' in registry
    assert registry.names() == ('notes',)
    assert registry.get('notes').plugin is plugin
    assert registry.capability('notes', 'create').name == 'create'


def test_builder_rejects_duplicate_names() -> None:
    builder = RegistryBuilder(max_blast_radius=BlastRadius.NETWORK)
    builder.register(parse_manifest(_valid_toml()), _StubPlugin())

    with pytest.raises(DuplicatePluginError, match="'notes'"):
        builder.register(parse_manifest(_valid_toml()), _StubPlugin())


def test_register_after_build_raises() -> None:
    builder = RegistryBuilder(max_blast_radius=BlastRadius.NETWORK)
    builder.register(parse_manifest(_valid_toml()), _StubPlugin())
    builder.build()

    with pytest.raises(RegistryFrozenError):
        builder.register(parse_manifest(_valid_toml(name='other')), _StubPlugin())


def test_double_build_raises() -> None:
    builder = RegistryBuilder(max_blast_radius=BlastRadius.NETWORK)
    builder.register(parse_manifest(_valid_toml()), _StubPlugin())
    builder.build()

    with pytest.raises(RegistryFrozenError):
        builder.build()


def test_registry_lookup_misses() -> None:
    builder = RegistryBuilder(max_blast_radius=BlastRadius.NETWORK)
    registry = builder.build()

    with pytest.raises(PluginNotFoundError):
        registry.get('nope')
    assert 'nope' not in registry
    assert len(registry) == 0
