"""Tests for the context manager — the small-context-footprint enforcer.

Covers the three footprint policies (capability filtering, history
windowing, memory retrieval), the seam protocols, and the default
in-process implementations.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from butter_agent.core.context_manager import (
    CapabilityDescriptor,
    CapabilityFilter,
    ConversationEntry,
    ConversationHistory,
    DefaultContextManager,
    InMemoryConversationHistory,
    KeywordCapabilityFilter,
    MemoryRetriever,
    MemorySnippet,
    NullMemoryRetriever,
    PluginIndexEntry,
)
from butter_agent.core.loop import DiscoverySelection, ExecutionResult, PlanStep, TaskPlan, Turn
from butter_agent.core.registry import (
    BlastRadius,
    PluginRegistry,
    RegistryBuilder,
    parse_manifest,
)

# --- Test plumbing -----------------------------------------------------------


class _StubPlugin:
    async def execute(
        self,
        capability: str,
        inputs: dict[str, object],
        context: object,
    ) -> dict[str, object]:
        del capability, inputs, context
        return {}


def _manifest_toml(name: str, *capabilities: tuple[str, str]) -> str:
    caps_block = '\n'.join(f'[[capability]]\nname = "{cname}"\ndescription = "{cdesc}"\ninput_schema = {{}}\noutput_schema = {{}}' for cname, cdesc in capabilities)
    return f"""
[plugin]
name = "{name}"
version = "0.1.0"
blast_radius = "read-only"
entrypoint = "main:Plugin"

{caps_block}
"""


def _manifest_toml_with_schema(name: str, capability: str, description: str, input_schema: dict[str, str]) -> str:
    """Manifest variant that exposes a non-empty input_schema.

    The flat `_manifest_toml` helper hard-codes empty schemas; this
    variant exists so tests can exercise the `required_inputs` surface
    on `CapabilityDescriptor` without rewriting the existing helper.
    """
    fields = ', '.join(f'{key} = "{type_name}"' for key, type_name in input_schema.items())
    return f"""
[plugin]
name = "{name}"
version = "0.1.0"
blast_radius = "read-only"
entrypoint = "main:Plugin"

[[capability]]
name = "{capability}"
description = "{description}"
input_schema = {{ {fields} }}
output_schema = {{}}
"""


def _registry(*plugins: tuple[str, tuple[tuple[str, str], ...]]) -> PluginRegistry:
    builder = RegistryBuilder(max_blast_radius=BlastRadius.NETWORK)
    for plugin_name, caps in plugins:
        manifest = parse_manifest(_manifest_toml(plugin_name, *caps))
        builder.register(manifest, _StubPlugin())
    return builder.build()


def _turn(text: str) -> Turn:
    return Turn(turn_id='t1', user_input=text, timestamp=0.0)


@dataclass
class _RecordingFilter:
    return_value: tuple[CapabilityDescriptor, ...]
    seen: list[tuple[Turn, tuple[CapabilityDescriptor, ...]]] = field(default_factory=list)

    def select(
        self,
        turn: Turn,
        available: tuple[CapabilityDescriptor, ...],
    ) -> tuple[CapabilityDescriptor, ...]:
        self.seen.append((turn, available))
        return self.return_value


@dataclass
class _RecordingMemory:
    snippets: tuple[MemorySnippet, ...]
    calls: list[tuple[str, int]] = field(default_factory=list)

    async def retrieve(self, query: str, limit: int) -> tuple[MemorySnippet, ...]:
        self.calls.append((query, limit))
        return self.snippets


# --- DefaultContextManager.assemble -----------------------------------------


async def test_assemble_skips_internal_capabilities_in_descriptor_list() -> None:
    """Internal capabilities are infrastructure, not planner-visible.

    Surfacing them to the model would spend tokens advertising plans the
    executor must then reject — and would confuse the model about which
    capabilities are actually available.
    """
    builder = RegistryBuilder(max_blast_radius=BlastRadius.NETWORK)
    infra_toml = """
[plugin]
name = "infra"
version = "0.1.0"
blast_radius = "local-write"
entrypoint = "main:Plugin"

[[capability]]
name = "store"
description = "Internal store"
input_schema = {}
output_schema = {}
internal = true

[[capability]]
name = "stats"
description = "Public stats"
input_schema = {}
output_schema = {}
"""
    builder.register(parse_manifest(infra_toml), _StubPlugin())
    registry = builder.build()

    seen: list[tuple[CapabilityDescriptor, ...]] = []

    class _Recorder:
        def select(
            self,
            turn: Turn,
            available: tuple[CapabilityDescriptor, ...],
        ) -> tuple[CapabilityDescriptor, ...]:
            del turn
            seen.append(available)
            return available

    cm = DefaultContextManager(registry, InMemoryConversationHistory(), capability_filter=_Recorder())
    await cm.assemble(_turn('hi'))

    (available,) = seen
    names = {desc.capability for desc in available}
    assert names == {'stats'}, 'internal capabilities must not be surfaced to the planner'


async def test_assemble_surfaces_all_capability_descriptors_when_filter_returns_all() -> None:
    registry = _registry(
        ('notes', (('create', 'Create a new note'), ('list', 'List all notes'))),
        ('search', (('web', 'Search the web'),)),
    )
    history = InMemoryConversationHistory()
    descriptors_seen: list[tuple[CapabilityDescriptor, ...]] = []

    class _PassThroughFilter:
        def select(
            self,
            turn: Turn,
            available: tuple[CapabilityDescriptor, ...],
        ) -> tuple[CapabilityDescriptor, ...]:
            descriptors_seen.append(available)
            return available

    cm = DefaultContextManager(registry, history, capability_filter=_PassThroughFilter())

    context = await cm.assemble(_turn('hi'))

    assert descriptors_seen == [
        (
            CapabilityDescriptor(plugin='notes', capability='create', description='Create a new note'),
            CapabilityDescriptor(plugin='notes', capability='list', description='List all notes'),
            CapabilityDescriptor(plugin='search', capability='web', description='Search the web'),
        ),
    ]
    assert context.turn.user_input == 'hi'
    assert context.payload['capabilities'] == descriptors_seen[0]
    assert context.payload['history'] == ()
    assert context.payload['memory'] == ()


async def test_assemble_uses_capability_filter_output_in_payload() -> None:
    registry = _registry(
        ('notes', (('create', 'Create a new note'),)),
        ('search', (('web', 'Search the web'),)),
    )
    only_search = (CapabilityDescriptor(plugin='search', capability='web', description='Search the web'),)
    recording = _RecordingFilter(return_value=only_search)

    cm = DefaultContextManager(registry, InMemoryConversationHistory(), capability_filter=recording)

    context = await cm.assemble(_turn('find something'))

    assert context.payload['capabilities'] == only_search
    assert recording.seen[0][0].user_input == 'find something'


async def test_assemble_windows_history_to_configured_limit() -> None:
    registry = _registry(('notes', (('create', 'Create a note'),)))
    history = InMemoryConversationHistory()
    for idx in range(5):
        await history.append(
            ConversationEntry(turn_id=f't{idx}', user_input=f'q{idx}', assistant_reply=f'r{idx}', timestamp=float(idx)),
        )

    cm = DefaultContextManager(registry, history, history_window=2)
    context = await cm.assemble(_turn('next'))

    surfaced = context.payload['history']
    assert isinstance(surfaced, tuple)
    assert [entry.turn_id for entry in surfaced] == ['t3', 't4']


async def test_assemble_skips_history_when_window_is_zero() -> None:
    registry = _registry(('notes', (('create', 'Create a note'),)))
    history = InMemoryConversationHistory()
    await history.append(ConversationEntry(turn_id='t0', user_input='hi', assistant_reply='hello', timestamp=0.0))

    cm = DefaultContextManager(registry, history, history_window=0)
    context = await cm.assemble(_turn('next'))

    assert context.payload['history'] == ()


async def test_assemble_retrieves_memory_using_user_input_as_query() -> None:
    registry = _registry(('notes', (('create', 'Create a note'),)))
    snippets = (MemorySnippet(source='memory-mcp', content='user prefers terse replies'),)
    memory = _RecordingMemory(snippets=snippets)

    cm = DefaultContextManager(registry, InMemoryConversationHistory(), memory=memory, memory_top_k=3)
    context = await cm.assemble(_turn('how should I phrase this?'))

    assert memory.calls == [('how should I phrase this?', 3)]
    assert context.payload['memory'] == snippets


async def test_assemble_skips_memory_when_top_k_is_zero() -> None:
    registry = _registry(('notes', (('create', 'Create a note'),)))
    memory = _RecordingMemory(snippets=(MemorySnippet(source='m', content='c'),))

    cm = DefaultContextManager(registry, InMemoryConversationHistory(), memory=memory, memory_top_k=0)
    context = await cm.assemble(_turn('q'))

    assert memory.calls == []
    assert context.payload['memory'] == ()


async def test_assemble_synthesis_attaches_execution_and_omits_capabilities() -> None:
    # Synthesis-mode assembly: model is about to summarise a plan that just
    # ran. Capabilities are deliberately omitted so the prompt does not
    # nudge the model toward proposing more actions; the execution result
    # is attached so the adapter can render it into the user prompt.
    registry = _registry(('notes', (('create', 'Create a note'),)))
    cm = DefaultContextManager(registry, InMemoryConversationHistory())
    plan = TaskPlan(
        steps=(PlanStep(step=1, plugin='notes', capability='create', inputs={}, gate='none', outputs_as='n'),),
    )
    execution = ExecutionResult(plan=plan, outputs={'n': {'id': 7}})

    context = await cm.assemble(_turn('save a note'), execution=execution)

    assert context.payload['execution'] is execution
    assert 'capabilities' not in context.payload


async def test_assemble_synthesis_still_surfaces_history_and_memory() -> None:
    # History and memory remain relevant during synthesis — the model still
    # needs prior conversation context to phrase its reply naturally.
    registry = _registry(('notes', (('create', 'Create a note'),)))
    history = InMemoryConversationHistory()
    await history.append(ConversationEntry(turn_id='t0', user_input='hi', assistant_reply='hello', timestamp=0.0))
    snippets = (MemorySnippet(source='memory-mcp', content='user prefers terse replies'),)
    memory = _RecordingMemory(snippets=snippets)
    cm = DefaultContextManager(registry, history, memory=memory)
    execution = ExecutionResult(
        plan=TaskPlan(steps=(PlanStep(step=1, plugin='notes', capability='create', inputs={}, gate='none'),)),
        outputs={},
    )

    context = await cm.assemble(_turn('save it'), execution=execution)

    assert context.payload['memory'] == snippets
    surfaced = context.payload['history']
    assert isinstance(surfaced, tuple)
    assert [entry.turn_id for entry in surfaced] == ['t0']


def test_negative_history_window_rejected() -> None:
    registry = _registry(('notes', (('create', 'Create a note'),)))
    with pytest.raises(ValueError, match='history_window'):
        DefaultContextManager(registry, InMemoryConversationHistory(), history_window=-1)


def test_negative_memory_top_k_rejected() -> None:
    registry = _registry(('notes', (('create', 'Create a note'),)))
    with pytest.raises(ValueError, match='memory_top_k'):
        DefaultContextManager(registry, InMemoryConversationHistory(), memory_top_k=-1)


async def test_assemble_surfaces_required_inputs_on_descriptor() -> None:
    """Descriptors expose the manifest's input_schema keys as `required_inputs`.

    Without this, the model has no way to know step inputs like `tz` are
    required and the executor atomically rejects the plan — see the
    `model-baseline-and-input-schema-prompt` spec for the failure mode.
    """
    builder = RegistryBuilder(max_blast_radius=BlastRadius.NETWORK)
    manifest = parse_manifest(
        _manifest_toml_with_schema('clock', 'now', 'Return the current wall-clock time.', {'tz': 'string'}),
    )
    builder.register(manifest, _StubPlugin())
    registry = builder.build()
    surfaced: list[tuple[CapabilityDescriptor, ...]] = []

    class _Capture:
        def select(
            self,
            turn: Turn,
            available: tuple[CapabilityDescriptor, ...],
        ) -> tuple[CapabilityDescriptor, ...]:
            surfaced.append(available)
            return available

    cm = DefaultContextManager(registry, InMemoryConversationHistory(), capability_filter=_Capture())
    await cm.assemble(_turn('what time is it'))

    assert surfaced == [
        (
            CapabilityDescriptor(
                plugin='clock',
                capability='now',
                description='Return the current wall-clock time.',
                required_inputs=('tz',),
            ),
        ),
    ]


# --- KeywordCapabilityFilter ------------------------------------------------


def test_keyword_filter_returns_all_when_available_fits_within_top_k() -> None:
    available = (
        CapabilityDescriptor(plugin='notes', capability='create', description='Create a note'),
        CapabilityDescriptor(plugin='search', capability='web', description='Search the web'),
    )
    out = KeywordCapabilityFilter(top_k=5).select(_turn('save a note'), available)
    assert out == available


def test_keyword_filter_ranks_by_token_overlap() -> None:
    available = (
        CapabilityDescriptor(plugin='notes', capability='create', description='Create a new note'),
        CapabilityDescriptor(plugin='search', capability='web', description='Search the web'),
        CapabilityDescriptor(plugin='reminders', capability='add', description='Add a reminder'),
    )
    out = KeywordCapabilityFilter(top_k=2).select(_turn('search the web for something'), available)
    assert out[0].plugin == 'search'
    assert len(out) == 2


def test_keyword_filter_falls_back_to_registration_order_when_no_overlap() -> None:
    available = (
        CapabilityDescriptor(plugin='notes', capability='create', description='Create a note'),
        CapabilityDescriptor(plugin='search', capability='web', description='Search the web'),
        CapabilityDescriptor(plugin='reminders', capability='add', description='Add a reminder'),
    )
    out = KeywordCapabilityFilter(top_k=2).select(_turn('zzz qqq'), available)
    assert out == available[:2]


def test_keyword_filter_falls_back_when_input_is_empty() -> None:
    available = (
        CapabilityDescriptor(plugin='notes', capability='create', description='Create a note'),
        CapabilityDescriptor(plugin='search', capability='web', description='Search the web'),
    )
    out = KeywordCapabilityFilter(top_k=1).select(_turn(''), available)
    assert out == available[:1]


def test_keyword_filter_ranks_required_input_names_as_haystack_tokens() -> None:
    """Required-input names participate in token-overlap scoring.

    Without this, "what time is it in china" would not pick `clock.now`
    over an unrelated capability whose description happens to mention
    time, because `tz` / `timezone` only appears in the input schema.
    """
    available = (
        CapabilityDescriptor(plugin='notes', capability='create', description='Save a note'),
        CapabilityDescriptor(plugin='clock', capability='now', description='Return the current wall-clock value', required_inputs=('timezone',)),
    )
    out = KeywordCapabilityFilter(top_k=1).select(_turn('timezone for shanghai'), available)
    assert out[0].plugin == 'clock'


def test_keyword_filter_rejects_non_positive_top_k() -> None:
    with pytest.raises(ValueError, match='top_k'):
        KeywordCapabilityFilter(top_k=0)


# --- InMemoryConversationHistory --------------------------------------------


async def test_in_memory_history_round_trips_entries_in_order() -> None:
    history = InMemoryConversationHistory()
    e1 = ConversationEntry(turn_id='a', user_input='hi', assistant_reply='hello', timestamp=1.0)
    e2 = ConversationEntry(turn_id='b', user_input='bye', assistant_reply='see you', timestamp=2.0)

    await history.append(e1)
    await history.append(e2)

    assert await history.recent(10) == (e1, e2)
    assert await history.recent(1) == (e2,)
    assert await history.recent(0) == ()


async def test_in_memory_history_evicts_oldest_past_max_entries() -> None:
    history = InMemoryConversationHistory(max_entries=2)
    for idx in range(4):
        await history.append(
            ConversationEntry(turn_id=f't{idx}', user_input=f'q{idx}', assistant_reply=None, timestamp=float(idx)),
        )
    surfaced = await history.recent(10)
    assert [entry.turn_id for entry in surfaced] == ['t2', 't3']


def test_in_memory_history_rejects_non_positive_max() -> None:
    with pytest.raises(ValueError, match='max_entries'):
        InMemoryConversationHistory(max_entries=0)


# --- NullMemoryRetriever -----------------------------------------------------


async def test_null_memory_retriever_returns_empty() -> None:
    result = await NullMemoryRetriever().retrieve('anything', 5)
    assert result == ()


# --- Capability discovery ---------------------------------------------------


def _caps(n: int, prefix: str) -> tuple[tuple[str, str], ...]:
    return tuple((f'{prefix}{i}', f'{prefix} capability {i}') for i in range(n))


def test_discovery_inactive_by_default() -> None:
    cm = DefaultContextManager(_registry(('notes', _caps(3, 'n'))), InMemoryConversationHistory())
    assert cm.discovery_active is False


async def test_discovery_active_surfaces_plugin_index_not_capabilities() -> None:
    # 3 plugins, 9 user-facing caps > default threshold 8 → discovery on.
    registry = _registry(
        ('notes', _caps(3, 'n')),
        ('search', _caps(3, 's')),
        ('files', _caps(3, 'f')),
    )
    cm = DefaultContextManager(registry, InMemoryConversationHistory(), capability_discovery=True)

    assert cm.discovery_active is True
    context = await cm.assemble(_turn('what dependencies does pyproject have'))

    assert 'capabilities' not in context.payload
    index = context.payload['plugin_index']
    assert index == (
        PluginIndexEntry(name='notes', summary='3 capabilities: n0, n1, n2'),
        PluginIndexEntry(name='search', summary='3 capabilities: s0, s1, s2'),
        PluginIndexEntry(name='files', summary='3 capabilities: f0, f1, f2'),
    )


async def test_discovery_index_prefers_manifest_summary_over_generated() -> None:
    summarised = """
[plugin]
name = "weather"
version = "0.1.0"
blast_radius = "network"
entrypoint = "main:Plugin"
summary = "Live forecasts and current conditions."

[[capability]]
name = "forecast"
description = "N-day forecast"
input_schema = {}
output_schema = {}
"""
    builder = RegistryBuilder(max_blast_radius=BlastRadius.NETWORK)
    builder.register(parse_manifest(summarised), _StubPlugin())
    builder.register(parse_manifest(_manifest_toml('notes', ('create', 'Create a note'))), _StubPlugin())
    registry = builder.build()

    cm = DefaultContextManager(
        registry,
        InMemoryConversationHistory(),
        capability_discovery=True,
        discovery_capability_threshold=0,
    )
    context = await cm.assemble(_turn('hi'))

    assert context.payload['plugin_index'] == (
        PluginIndexEntry(name='weather', summary='Live forecasts and current conditions.'),
        PluginIndexEntry(name='notes', summary='1 capability: create'),
    )


async def test_discovery_index_omits_internal_only_plugins() -> None:
    # An all-internal infra plugin (e.g. database) is unplannable, so it
    # must not appear in the Tier-1 index — selecting it could only fail.
    infra = """
[plugin]
name = "database"
version = "0.1.0"
blast_radius = "local-write"
entrypoint = "main:Plugin"

[[capability]]
name = "insert"
description = "internal"
input_schema = {}
output_schema = {}
internal = true
"""
    builder = RegistryBuilder(max_blast_radius=BlastRadius.NETWORK)
    builder.register(parse_manifest(infra), _StubPlugin())
    builder.register(parse_manifest(_manifest_toml('notes', ('create', 'Create a note'), ('list', 'List notes'))), _StubPlugin())
    # A second user-facing plugin so the index has >1 row and discovery
    # actually activates (otherwise the single-plugin skip would fire and
    # we'd never see the index — the omission is what's under test).
    builder.register(parse_manifest(_manifest_toml('search', ('web', 'Search the web'))), _StubPlugin())
    registry = builder.build()

    cm = DefaultContextManager(
        registry,
        InMemoryConversationHistory(),
        capability_discovery=True,
        discovery_capability_threshold=0,
    )
    context = await cm.assemble(_turn('hi'))

    assert context.payload['plugin_index'] == (
        PluginIndexEntry(name='notes', summary='2 capabilities: create, list'),
        PluginIndexEntry(name='search', summary='1 capability: web'),
    )


def test_discovery_skipped_when_single_user_facing_plugin() -> None:
    # One plugin → the index has nothing to choose between; the extra
    # round-trip is pure latency, so discovery stays off even if enabled.
    registry = _registry(('notes', _caps(20, 'n')))
    cm = DefaultContextManager(registry, InMemoryConversationHistory(), capability_discovery=True)
    assert cm.discovery_active is False


def test_discovery_skipped_when_menu_within_threshold() -> None:
    # 2 plugins but only 6 caps ≤ threshold 8 — the keyword filter would
    # surface the whole menu untruncated anyway, so discovery is skipped.
    registry = _registry(('notes', _caps(3, 'n')), ('search', _caps(3, 's')))
    cm = DefaultContextManager(registry, InMemoryConversationHistory(), capability_discovery=True)
    assert cm.discovery_active is False


async def test_tier2_selection_filters_to_named_plugins() -> None:
    registry = _registry(
        ('notes', (('create', 'Create a note'),)),
        ('search', (('web', 'Search the web'),)),
        ('files', (('read', 'Read a file'),)),
    )
    cm = DefaultContextManager(registry, InMemoryConversationHistory(), capability_discovery=True)

    context = await cm.assemble(_turn('find it'), selection=DiscoverySelection(plugins=('search', 'files')))

    caps = context.payload['capabilities']
    assert isinstance(caps, tuple)
    assert {c.plugin for c in caps} == {'search', 'files'}
    assert 'plugin_index' not in context.payload


async def test_tier2_empty_selection_falls_back_to_keyword_filter() -> None:
    # Spec migration step 3: an empty/unresolved selection falls back to
    # the keyword filter over the whole set, never an empty menu.
    registry = _registry(
        ('notes', (('create', 'Create a note'),)),
        ('search', (('web', 'Search the web'),)),
    )
    cm = DefaultContextManager(registry, InMemoryConversationHistory(), capability_discovery=True)

    context = await cm.assemble(_turn('search the web'), selection=DiscoverySelection(plugins=()))

    caps = context.payload['capabilities']
    assert isinstance(caps, tuple)
    assert {c.plugin for c in caps} == {'notes', 'search'}


async def test_tier2_unknown_plugin_selection_falls_back_to_keyword_filter() -> None:
    registry = _registry(
        ('notes', (('create', 'Create a note'),)),
        ('search', (('web', 'Search the web'),)),
    )
    cm = DefaultContextManager(registry, InMemoryConversationHistory(), capability_discovery=True)

    context = await cm.assemble(_turn('hi'), selection=DiscoverySelection(plugins=('ghost',)))

    caps = context.payload['capabilities']
    assert isinstance(caps, tuple)
    assert caps != ()
    assert {c.plugin for c in caps} == {'notes', 'search'}


async def test_assemble_rejects_execution_and_selection_together() -> None:
    registry = _registry(('notes', (('create', 'Create a note'),)))
    cm = DefaultContextManager(registry, InMemoryConversationHistory())
    execution = ExecutionResult(
        plan=TaskPlan(steps=(PlanStep(step=1, plugin='notes', capability='create', inputs={}, gate='none'),)),
        outputs={},
    )
    with pytest.raises(ValueError, match='mutually exclusive'):
        await cm.assemble(_turn('q'), execution=execution, selection=DiscoverySelection(plugins=('notes',)))


def test_negative_discovery_threshold_rejected() -> None:
    with pytest.raises(ValueError, match='discovery_capability_threshold'):
        DefaultContextManager(
            _registry(('notes', (('create', 'Create a note'),))),
            InMemoryConversationHistory(),
            discovery_capability_threshold=-1,
        )


# --- Protocol introspection --------------------------------------------------


def test_protocols_are_importable_and_named() -> None:
    assert ConversationHistory.__name__ == 'ConversationHistory'
    assert MemoryRetriever.__name__ == 'MemoryRetriever'
    assert CapabilityFilter.__name__ == 'CapabilityFilter'
