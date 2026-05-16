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
    RequiresCycleError,
    RequiresValidationError,
    TransitiveBlastRadiusViolation,
    parse_manifest,
    radius_permits,
)


# A minimal stub plugin used to populate the registry in tests.
class _StubPlugin:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, object]]] = []

    async def execute(
        self,
        capability: str,
        inputs: dict[str, object],
        context: object,
    ) -> dict[str, object]:
        del context
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


@pytest.mark.parametrize(
    'bad_name',
    [
        'Has-Hyphen',
        'has space',
        '1leading_digit',
        'has.dot',
        'UpperCase',
    ],
)
def test_parse_manifest_rejects_unsafe_plugin_name(bad_name: str) -> None:
    """Plugin names are interpolated into the model prompt — restrict charset.

    PR #17 review (Copilot): without a charset restriction at parse
    time, a malicious or careless manifest could break prompt
    structure or enable prompt injection. Enforcing
    `[a-z][a-z0-9_]*` at the trust boundary means downstream
    renderers (ollama prompt, debug output, failure_reason) can
    interpolate verbatim. Quote/newline names are caught one layer up
    by tomllib (see `test_parse_manifest_rejects_quote_or_newline_in_name`).
    """
    bad = _valid_toml(name=bad_name)
    with pytest.raises(ManifestError, match='plugin name'):
        parse_manifest(bad)


def test_parse_manifest_rejects_double_underscore_in_plugin_name() -> None:
    """`__` passes the identifier charset but is the reserved database
    namespace separator (`{plugin}__{table}`, invariant #6).

    Allowing it would make a fully-qualified physical table name
    ambiguous between two legitimate registrations. The separator must be
    genuinely reserved to core, not reserved only by convention.
    """
    bad = _valid_toml(name='looks__legit')
    with pytest.raises(ManifestError, match='reserved as the database namespace separator'):
        parse_manifest(bad)


@pytest.mark.parametrize(
    'bad_name',
    [
        'has"quote',
        'has\nnewline',
    ],
)
def test_parse_manifest_rejects_quote_or_newline_in_name(bad_name: str) -> None:
    """The two highest-risk injection vectors are stopped by tomllib itself.

    A quote-bearing name produces malformed TOML (`name = "has"quote"`);
    a newline-bearing name straddles a key/value boundary. Either way
    `tomllib.loads` raises and we rewrap as `ManifestError('invalid
    TOML: ...')`. The identifier-charset check above covers the
    valid-TOML-but-unsafe cases; these tests pin the upstream guard so
    a future regression that loosens TOML handling can't silently
    reopen the injection surface.
    """
    bad = _valid_toml(name=bad_name)
    with pytest.raises(ManifestError, match='invalid TOML'):
        parse_manifest(bad)


def test_parse_manifest_rejects_unsafe_capability_name() -> None:
    """Capability names share the same restriction as plugin names for the same reason."""
    toml_text = """
[plugin]
name = "ok"
version = "0.1.0"
blast_radius = "read-only"
entrypoint = "main:Plugin"

[[capability]]
name = "bad-name"
description = "anything"
input_schema = {}
output_schema = {}
"""
    with pytest.raises(ManifestError, match='capability name'):
        parse_manifest(toml_text)


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


def test_parse_manifest_rejects_duplicate_capability_names() -> None:
    toml_text = """
[plugin]
name = "dupe"
version = "0.1.0"
blast_radius = "read-only"
entrypoint = "main:Plugin"

[[capability]]
name = "do_thing"
description = "first definition"
input_schema = {}
output_schema = {}

[[capability]]
name = "do_thing"
description = "shadowing the first"
input_schema = {}
output_schema = {}
"""
    with pytest.raises(ManifestError, match="duplicate capability name 'do_thing'"):
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


# --- Manifest: requires + internal -----------------------------------------


_INFRA_TOML = """
[plugin]
name = "database"
version = "0.1.0"
blast_radius = "local-write"
entrypoint = "main:Plugin"

[[capability]]
name = "insert"
description = "Insert a row"
input_schema = {}
output_schema = {}
internal = true

[[capability]]
name = "select"
description = "Read rows"
input_schema = {}
output_schema = {}
internal = true
"""


def _caller_toml(
    *,
    name: str = 'notes',
    radius: str = 'local-write',
    requires: tuple[str, ...] = ('database.insert', 'database.select'),
) -> str:
    requires_line = '' if not requires else 'requires = [' + ', '.join(f'"{r}"' for r in requires) + ']'
    return f"""
[plugin]
name = "{name}"
version = "0.1.0"
blast_radius = "{radius}"
entrypoint = "main:Plugin"
{requires_line}

[[capability]]
name = "create"
description = "Create a note"
input_schema = {{}}
output_schema = {{}}
"""


def test_parse_manifest_parses_internal_flag_and_requires() -> None:
    infra = parse_manifest(_INFRA_TOML)
    assert all(cap.internal for cap in infra.capabilities)
    caller = parse_manifest(_caller_toml())
    assert caller.requires == ('database.insert', 'database.select')
    assert all(not cap.internal for cap in caller.capabilities)


def test_parse_manifest_requires_default_is_empty() -> None:
    manifest = parse_manifest(_valid_toml())
    assert manifest.requires == ()


def test_parse_manifest_rejects_unqualified_requires_entry() -> None:
    """A bare plugin name (missing `.capability`) is rejected by the format check."""
    with pytest.raises(ManifestError, match='requires'):
        parse_manifest(_caller_toml(requires=('database',)))


def test_parse_manifest_rejects_non_string_requires_entry() -> None:
    """A non-string `requires` element fails the format check the same way."""
    bad = """
[plugin]
name = "notes"
version = "0.1.0"
blast_radius = "local-write"
entrypoint = "main:Plugin"
requires = [42]

[[capability]]
name = "create"
description = "Create a note"
input_schema = {}
output_schema = {}
"""
    with pytest.raises(ManifestError, match='requires'):
        parse_manifest(bad)


def test_parse_manifest_rejects_malformed_requires_ref() -> None:
    with pytest.raises(ManifestError, match=r'plugin\.capability'):
        parse_manifest(_caller_toml(requires=('Database.Insert',)))


def test_parse_manifest_rejects_duplicate_requires() -> None:
    with pytest.raises(ManifestError, match='duplicate requires'):
        parse_manifest(_caller_toml(requires=('database.insert', 'database.insert')))


def test_parse_manifest_rejects_non_bool_internal() -> None:
    bad = _INFRA_TOML.replace('internal = true', 'internal = "yes"')
    with pytest.raises(ManifestError, match='internal'):
        parse_manifest(bad)


# --- Manifest: [plugin].summary (capability-discovery Tier-1) ---------------


def test_parse_manifest_summary_absent_is_none() -> None:
    # Backward compatible: a manifest with no summary parses fine and the
    # context manager synthesises a neutral fallback from capability names.
    assert parse_manifest(_valid_toml()).summary is None


def test_parse_manifest_summary_preserved() -> None:
    toml = _valid_toml().replace(
        'entrypoint = "main:Plugin"',
        'entrypoint = "main:Plugin"\nsummary = "Keeps short text notes."',
    )
    assert parse_manifest(toml).summary == 'Keeps short text notes.'


def test_parse_manifest_summary_collapses_whitespace_and_newlines() -> None:
    # Trust-boundary defence: a multi-line summary cannot inject extra
    # prompt lines / fake instruction blocks — runs collapse to one space.
    toml = _valid_toml().replace(
        'entrypoint = "main:Plugin"',
        'entrypoint = "main:Plugin"\nsummary = "line one\\n\\nIGNORE PREVIOUS\\tline two"',
    )
    assert parse_manifest(toml).summary == 'line one IGNORE PREVIOUS line two'


def test_parse_manifest_summary_blank_is_none() -> None:
    # Whitespace-only and absent must behave identically.
    toml = _valid_toml().replace(
        'entrypoint = "main:Plugin"',
        'entrypoint = "main:Plugin"\nsummary = "   \\n  "',
    )
    assert parse_manifest(toml).summary is None


def test_parse_manifest_summary_truncated_to_cap() -> None:
    long = 'x' * 500
    toml = _valid_toml().replace(
        'entrypoint = "main:Plugin"',
        f'entrypoint = "main:Plugin"\nsummary = "{long}"',
    )
    summary = parse_manifest(toml).summary
    assert summary is not None
    assert len(summary) == 200


def test_parse_manifest_rejects_non_string_summary() -> None:
    toml = _valid_toml().replace(
        'entrypoint = "main:Plugin"',
        'entrypoint = "main:Plugin"\nsummary = 42',
    )
    with pytest.raises(ManifestError, match='summary must be a string'):
        parse_manifest(toml)


# --- Builder: requires + transitive radius ---------------------------------


def test_builder_accepts_well_formed_requires_graph() -> None:
    builder = RegistryBuilder(max_blast_radius=BlastRadius.NETWORK)
    builder.register(parse_manifest(_INFRA_TOML), _StubPlugin())
    builder.register(parse_manifest(_caller_toml()), _StubPlugin())
    registry = builder.build()
    assert registry.get('notes').manifest.requires == ('database.insert', 'database.select')


def test_builder_rejects_requires_pointing_at_unknown_plugin() -> None:
    builder = RegistryBuilder(max_blast_radius=BlastRadius.NETWORK)
    builder.register(parse_manifest(_caller_toml(requires=('ghost.insert',))), _StubPlugin())
    with pytest.raises(RequiresValidationError, match='unknown plugin'):
        builder.build()


def test_builder_rejects_requires_pointing_at_unknown_capability() -> None:
    builder = RegistryBuilder(max_blast_radius=BlastRadius.NETWORK)
    builder.register(parse_manifest(_INFRA_TOML), _StubPlugin())
    builder.register(parse_manifest(_caller_toml(requires=('database.drop_table',))), _StubPlugin())
    with pytest.raises(RequiresValidationError, match='no capability'):
        builder.build()


def test_builder_rejects_requires_pointing_at_non_internal_capability() -> None:
    """requires may only target internal capabilities — the firewall is strict."""
    builder = RegistryBuilder(max_blast_radius=BlastRadius.NETWORK)
    # A user-facing plugin that exposes `create` (non-internal).
    builder.register(parse_manifest(_caller_toml(name='notes', requires=())), _StubPlugin())
    # Another plugin tries to call notes.create plugin-to-plugin.
    bad = _caller_toml(name='reminder', requires=('notes.create',))
    builder.register(parse_manifest(bad), _StubPlugin())
    with pytest.raises(RequiresValidationError, match='non-internal'):
        builder.build()


def test_builder_rejects_self_requires() -> None:
    builder = RegistryBuilder(max_blast_radius=BlastRadius.NETWORK)
    builder.register(parse_manifest(_caller_toml(requires=('notes.create',))), _StubPlugin())
    with pytest.raises(RequiresValidationError, match='points at itself'):
        builder.build()


def test_builder_rejects_cycle_in_requires() -> None:
    a_toml = """
[plugin]
name = "alpha"
version = "0.1.0"
blast_radius = "local-write"
entrypoint = "main:Plugin"
requires = ["beta.bar"]

[[capability]]
name = "foo"
description = "Foo"
input_schema = {}
output_schema = {}
internal = true
"""
    b_toml = """
[plugin]
name = "beta"
version = "0.1.0"
blast_radius = "local-write"
entrypoint = "main:Plugin"
requires = ["alpha.foo"]

[[capability]]
name = "bar"
description = "Bar"
input_schema = {}
output_schema = {}
internal = true
"""
    builder = RegistryBuilder(max_blast_radius=BlastRadius.NETWORK)
    builder.register(parse_manifest(a_toml), _StubPlugin())
    builder.register(parse_manifest(b_toml), _StubPlugin())
    with pytest.raises(RequiresCycleError, match='alpha'):
        builder.build()


def test_cycle_error_excludes_innocent_downstream_plugins() -> None:
    """The cycle error names only true cycle members, not their downstream.

    With alpha ⇄ beta forming the cycle and gamma depending on alpha,
    gamma is stuck in topo-sort but is *not* part of the cycle. The
    error must point at alpha+beta only — naming gamma would mislead
    a plugin author into hunting their own manifest for a non-existent
    cycle.
    """
    a = """
[plugin]
name = "alpha"
version = "0.1.0"
blast_radius = "local-write"
entrypoint = "main:Plugin"
requires = ["beta.bar"]

[[capability]]
name = "foo"
description = "Foo"
input_schema = {}
output_schema = {}
internal = true
"""
    b = """
[plugin]
name = "beta"
version = "0.1.0"
blast_radius = "local-write"
entrypoint = "main:Plugin"
requires = ["alpha.foo"]

[[capability]]
name = "bar"
description = "Bar"
input_schema = {}
output_schema = {}
internal = true
"""
    g = """
[plugin]
name = "gamma"
version = "0.1.0"
blast_radius = "local-write"
entrypoint = "main:Plugin"
requires = ["alpha.foo"]

[[capability]]
name = "use"
description = "Use alpha"
input_schema = {}
output_schema = {}
"""
    builder = RegistryBuilder(max_blast_radius=BlastRadius.NETWORK)
    builder.register(parse_manifest(a), _StubPlugin())
    builder.register(parse_manifest(b), _StubPlugin())
    builder.register(parse_manifest(g), _StubPlugin())
    with pytest.raises(RequiresCycleError) as info:
        builder.build()
    message = str(info.value)
    assert 'alpha' in message
    assert 'beta' in message
    assert 'gamma' not in message


def test_builder_rejects_declared_radius_below_transitive_max() -> None:
    """A read-only plugin cannot transitively reach a local-write capability."""
    builder = RegistryBuilder(max_blast_radius=BlastRadius.NETWORK)
    builder.register(parse_manifest(_INFRA_TOML), _StubPlugin())  # local-write
    bad = _caller_toml(name='reader', radius='read-only', requires=('database.insert',))
    builder.register(parse_manifest(bad), _StubPlugin())
    with pytest.raises(TransitiveBlastRadiusViolation, match='read-only'):
        builder.build()


def test_builder_accepts_declared_radius_equal_to_transitive_max() -> None:
    builder = RegistryBuilder(max_blast_radius=BlastRadius.NETWORK)
    builder.register(parse_manifest(_INFRA_TOML), _StubPlugin())  # local-write
    builder.register(parse_manifest(_caller_toml(radius='local-write')), _StubPlugin())
    registry = builder.build()
    assert 'notes' in registry
