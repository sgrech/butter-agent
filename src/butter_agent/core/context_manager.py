"""Context manager — small-context-footprint enforcer.

Assembles the per-turn `ModelContext` for the agent loop. The constraint
this module owns: the model sees only relevant info per turn — capability
descriptions are filtered (not dumped), conversation history is windowed,
memory snippets are retrieved (not dumped).

Three seams expressed as Protocols make the footprint policy pluggable
without changing the loop's shape (invariant #1):

- `ConversationHistory` — append-and-window store. Day-2 default is an
  in-process list; SQLite-backed storage is the eventual home (see the
  Memory/State Layer in the scope).
- `MemoryRetriever` — pulls a handful of memory snippets relevant to a
  turn. Default is `NullMemoryRetriever` for installations without a
  memory backend wired up.
- `CapabilityFilter` — decides which capability descriptions enter the
  prompt. This is the load-bearing enforcer of the "filtered not dumped"
  half of the footprint constraint.

This module does NOT execute capabilities, mutate the registry, or
interpret model output — those belong to the executor and the loop.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol

from butter_agent.core.loop import ModelContext, Turn
from butter_agent.core.registry import PluginRegistry

# --- Value types -------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ConversationEntry:
    """One completed turn's surface — what the model needs to recall it."""

    turn_id: str
    user_input: str
    assistant_reply: str | None
    timestamp: float


@dataclass(frozen=True, slots=True)
class MemorySnippet:
    """A single retrieved memory record surfaced into the model context."""

    source: str
    content: str


@dataclass(frozen=True, slots=True)
class CapabilityDescriptor:
    """A capability advertised to the model for plan construction.

    Only the plugin name, capability name, and description are surfaced.
    Input/output schemas are the executor's concern (validation happens
    after the model returns a plan), not the model's.
    """

    plugin: str
    capability: str
    description: str


# --- Seam protocols ----------------------------------------------------------


class ConversationHistory(Protocol):
    """Append-and-window store for prior turn surfaces."""

    async def append(self, entry: ConversationEntry) -> None: ...

    async def recent(self, limit: int) -> tuple[ConversationEntry, ...]: ...


class MemoryRetriever(Protocol):
    """Pulls a handful of memory snippets relevant to a turn — never the full store."""

    async def retrieve(self, query: str, limit: int) -> tuple[MemorySnippet, ...]: ...


class CapabilityFilter(Protocol):
    """Decides which capability descriptions enter the model context."""

    def select(
        self,
        turn: Turn,
        available: tuple[CapabilityDescriptor, ...],
    ) -> tuple[CapabilityDescriptor, ...]: ...


# --- Default implementations -------------------------------------------------


class InMemoryConversationHistory:
    """Process-local conversation log, bounded to the most recent entries.

    Acceptable as a day-2 stand-in for the SQLite-backed store. The
    replacement only needs to honour the `ConversationHistory` protocol;
    callers of `DefaultContextManager` are unaffected.
    """

    def __init__(self, max_entries: int = 1000) -> None:
        if max_entries <= 0:
            raise ValueError('max_entries must be positive')
        self._max = max_entries
        self._entries: list[ConversationEntry] = []

    async def append(self, entry: ConversationEntry) -> None:
        self._entries.append(entry)
        if len(self._entries) > self._max:
            # Drop the oldest entries so the working set stays bounded.
            self._entries = self._entries[-self._max :]

    async def recent(self, limit: int) -> tuple[ConversationEntry, ...]:
        if limit <= 0:
            return ()
        return tuple(self._entries[-limit:])


class NullMemoryRetriever:
    """Memory store that returns nothing — for installations without a memory backend."""

    async def retrieve(self, query: str, limit: int) -> tuple[MemorySnippet, ...]:
        return ()


class KeywordCapabilityFilter:
    """Token-overlap ranker: return up to `top_k` capabilities by overlap with the turn text.

    Falls back to the first `top_k` capabilities (in registration order) when
    no capability shares any token with the input — the model still needs
    *something* to plan with, but never the entire registry.
    """

    def __init__(self, top_k: int = 8) -> None:
        if top_k <= 0:
            raise ValueError('top_k must be positive')
        self._top_k = top_k

    def select(
        self,
        turn: Turn,
        available: tuple[CapabilityDescriptor, ...],
    ) -> tuple[CapabilityDescriptor, ...]:
        if len(available) <= self._top_k:
            return available
        tokens = _tokenise(turn.user_input)
        if not tokens:
            return available[: self._top_k]
        scored = [(_score(desc, tokens), idx, desc) for idx, desc in enumerate(available)]
        # Sort by descending score, then by registration order to break ties deterministically.
        scored.sort(key=lambda triple: (-triple[0], triple[1]))
        if scored[0][0] == 0:
            return available[: self._top_k]
        return tuple(desc for _, _, desc in scored[: self._top_k])


_TOKEN_SPLIT = re.compile(r'\W+')


def _tokenise(text: str) -> frozenset[str]:
    return frozenset(token for token in _TOKEN_SPLIT.split(text.lower()) if token)


def _score(desc: CapabilityDescriptor, tokens: frozenset[str]) -> int:
    haystack = _tokenise(f'{desc.plugin} {desc.capability} {desc.description}')
    return len(tokens & haystack)


# --- The context manager -----------------------------------------------------


class DefaultContextManager:
    """Composes per-turn context from registry, history, and memory.

    Footprint policy:
    - `capabilities`: a filtered subset from the registry, never the full set.
    - `history`: the last `history_window` conversation entries, oldest first.
    - `memory`: up to `memory_top_k` retrieved snippets, never the full store.

    The registry is the same frozen `PluginRegistry` (invariant #2) the
    executor uses, so the model only ever sees descriptions for capabilities
    the executor can actually invoke.
    """

    def __init__(
        self,
        registry: PluginRegistry,
        history: ConversationHistory,
        *,
        memory: MemoryRetriever | None = None,
        capability_filter: CapabilityFilter | None = None,
        history_window: int = 10,
        memory_top_k: int = 5,
    ) -> None:
        if history_window < 0:
            raise ValueError('history_window must be non-negative')
        if memory_top_k < 0:
            raise ValueError('memory_top_k must be non-negative')
        self._registry = registry
        self._history = history
        self._memory: MemoryRetriever = memory if memory is not None else NullMemoryRetriever()
        self._capability_filter: CapabilityFilter = capability_filter if capability_filter is not None else KeywordCapabilityFilter()
        self._history_window = history_window
        self._memory_top_k = memory_top_k

    async def assemble(self, turn: Turn) -> ModelContext:
        descriptors = _all_descriptors(self._registry)
        capabilities = self._capability_filter.select(turn, descriptors)
        history = await self._history.recent(self._history_window) if self._history_window else ()
        memory = await self._memory.retrieve(turn.user_input, self._memory_top_k) if self._memory_top_k else ()
        payload: dict[str, object] = {
            'capabilities': capabilities,
            'history': history,
            'memory': memory,
        }
        return ModelContext(turn=turn, payload=payload)


def _all_descriptors(registry: PluginRegistry) -> tuple[CapabilityDescriptor, ...]:
    descriptors: list[CapabilityDescriptor] = []
    for name in registry.names():
        manifest = registry.get(name).manifest
        for cap in manifest.capabilities:
            descriptors.append(
                CapabilityDescriptor(plugin=name, capability=cap.name, description=cap.description),
            )
    return tuple(descriptors)
