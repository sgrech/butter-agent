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
)
from butter_agent.core.loop import ExecutionResult, PlanStep, TaskPlan, Turn
from butter_agent.core.registry import (
    BlastRadius,
    PluginRegistry,
    RegistryBuilder,
    parse_manifest,
)

# --- Test plumbing -----------------------------------------------------------


class _StubPlugin:
    async def execute(self, capability: str, inputs: dict[str, object]) -> dict[str, object]:
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


# --- Protocol introspection --------------------------------------------------


def test_protocols_are_importable_and_named() -> None:
    assert ConversationHistory.__name__ == 'ConversationHistory'
    assert MemoryRetriever.__name__ == 'MemoryRetriever'
    assert CapabilityFilter.__name__ == 'CapabilityFilter'
