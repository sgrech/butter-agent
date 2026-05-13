"""Agent core loop — the shape-fixed orchestrator of a single turn.

Invariant #1: the core loop never changes shape at runtime. Every turn is the
same five steps:

    user input → Turn → ContextManager → ModelClient → discriminate → reply or plan

If the model output is a direct reply, return it. If it is a task plan, hand it
off to the TaskExecutor (which owns gate enforcement and `$variable` resolution
per invariants #3, #4, #5). The loop itself never validates plan content,
resolves variables, or makes blast-radius decisions — those belong to the
executor.

The loop depends on three seams expressed as Protocols. Implementations live in
sibling modules (`context_manager.py`, `task_executor.py`) and in adapters for
the model provider. This file owns the orchestration only.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Protocol

# --- Value types -------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Turn:
    """A single user-initiated turn entering the loop.

    Attributes:
        turn_id: Stable identifier for correlation across logs and history.
        user_input: Raw text from the active interface adapter (REPL, Telegram).
        timestamp: Unix epoch seconds when the turn was received.
    """

    turn_id: str
    user_input: str
    timestamp: float


@dataclass(frozen=True, slots=True)
class ModelContext:
    """Assembled context passed to the model on a single turn.

    Built by ContextManager. The loop is opaque to its contents; it forwards
    the object to the model and never inspects it. This is what enforces
    invariant #1 — the loop's shape is independent of what context-assembly
    decides to include.
    """

    turn: Turn
    payload: dict[str, object]


@dataclass(frozen=True, slots=True)
class PlanStep:
    """One step in a task plan, as produced by the model.

    Validation of these fields is the task executor's responsibility (invariant
    #3: plans are validated atomically before any step executes). The loop
    only carries the values through.
    """

    step: int
    plugin: str
    capability: str
    inputs: dict[str, object]
    gate: str
    outputs_as: str | None = None


@dataclass(frozen=True, slots=True)
class TaskPlan:
    """Ordered task plan to be handed to the executor."""

    steps: tuple[PlanStep, ...]


@dataclass(frozen=True, slots=True)
class ModelReply:
    """A direct, conversational reply with no plugin execution required."""

    text: str


# Discriminated union: the model produces exactly one of these.
ModelOutput = ModelReply | TaskPlan


@dataclass(frozen=True, slots=True)
class TurnResult:
    """The loop's return value for a completed turn.

    Either `reply` is set (direct conversational reply) or `executed_plan` is
    set (the plan that the executor walked, along with whatever it returned).
    Exactly one of the two is populated.
    """

    turn: Turn
    reply: ModelReply | None = None
    executed_plan: ExecutionResult | None = None


@dataclass(frozen=True, slots=True)
class ExecutionResult:
    """Opaque envelope returned by the executor.

    The loop does not interpret its contents; it just hands it back to the
    caller (REPL/adapter) for display. Concrete shape is defined by the
    executor module.
    """

    plan: TaskPlan
    outputs: dict[str, dict[str, object]] = field(default_factory=dict)
    halted_at_step: int | None = None
    halt_reason: str | None = None


# --- Seam protocols ----------------------------------------------------------


class ContextManager(Protocol):
    """Assembles the per-turn model context (history, memory, capabilities).

    The loop calls `assemble` once per turn. Implementations enforce the
    small-context-footprint constraint — windowing history, filtering plugin
    capability descriptions, retrieving relevant memory snippets.
    """

    async def assemble(self, turn: Turn) -> ModelContext: ...


class ModelClient(Protocol):
    """Adapter to the underlying LLM (Ollama / Qwen3 8B by default).

    Implementations are responsible for prompting the model to return a
    structured `ModelOutput`. Parse failures should raise `ModelProtocolError`
    so the loop can surface a clean diagnostic rather than guess.
    """

    async def generate(self, context: ModelContext) -> ModelOutput: ...


class TaskExecutor(Protocol):
    """Walks a task plan, resolves variables, enforces gates.

    The loop hands off plans verbatim. The executor owns validation (invariant
    #3), variable resolution (invariant #4), and gate enforcement (invariant
    #5).
    """

    async def execute(self, plan: TaskPlan) -> ExecutionResult: ...


# --- Errors ------------------------------------------------------------------


class LoopError(Exception):
    """Base class for loop-level errors."""


class ModelProtocolError(LoopError):
    """Raised by a ModelClient when it cannot produce a well-formed ModelOutput."""


# --- The loop ---------------------------------------------------------------


class AgentLoop:
    """The shape-fixed agent core.

    A single `AgentLoop` instance holds the wiring (context manager, model
    client, task executor) and is invoked once per turn. The wiring is set at
    construction and never reassigned — invariant #1.
    """

    def __init__(
        self,
        context_manager: ContextManager,
        model: ModelClient,
        executor: TaskExecutor,
    ) -> None:
        self._context_manager = context_manager
        self._model = model
        self._executor = executor

    async def run_turn(self, user_input: str) -> TurnResult:
        """Run one turn through the fixed five-step pipeline.

        Args:
            user_input: Raw text from the active interface adapter.

        Returns:
            A TurnResult with exactly one of `reply` or `executed_plan` set.

        Raises:
            ModelProtocolError: If the model adapter cannot produce a valid
                ModelOutput. The caller decides how to surface this; the loop
                does not retry or guess.
        """
        turn = _build_turn(user_input)
        context = await self._context_manager.assemble(turn)
        output = await self._model.generate(context)

        if isinstance(output, ModelReply):
            return TurnResult(turn=turn, reply=output)

        # output is a TaskPlan
        result = await self._executor.execute(output)
        return TurnResult(turn=turn, executed_plan=result)


# --- Helpers -----------------------------------------------------------------


def _build_turn(user_input: str) -> Turn:
    """Construct a Turn from raw user input."""
    return Turn(
        turn_id=uuid.uuid4().hex,
        user_input=user_input,
        timestamp=time.time(),
    )
