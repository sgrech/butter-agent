"""Agent core loop — the shape-fixed orchestrator of a single turn.

Invariant #1: the core loop never changes shape at runtime. Every turn walks
the same fixed pipeline:

    user input → Turn → ContextManager → ModelClient → discriminate
        → ModelReply: return as the assistant surface
        → TaskPlan: TaskExecutor → if not halted, second ModelClient call in
          synthesis mode to produce a natural-language reply grounded in the
          tool outputs

The synthesis pass is what makes the agent multi-turn aware: without it, tool
outputs never enter conversation history, so the next turn can't reason about
"that flight", "the time you just told me", etc. Synthesis is run only on
successful executions — a halted plan stops at the halt boundary and is
rendered verbatim, since the operator already chose to abort.

The loop depends on four seams expressed as Protocols. Implementations live in
sibling modules (`context_manager.py`, `task_executor.py`) and in adapters for
the model provider. This file owns the orchestration only.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from butter_agent.core.context_manager import ConversationEntry

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
    """Envelope returned by the executor, optionally augmented with a synthesis reply.

    The executor itself only populates `plan`, `outputs`, and the halt fields.
    The loop fills in `synthesis_reply` after a successful (non-halted) run by
    calling the model a second time with the tool outputs in context. This is
    what carries plugin results into the conversation surface — without it
    the agent is a one-shot tool dispatcher with no awareness of what the
    tools observed.
    """

    plan: TaskPlan
    outputs: dict[str, dict[str, object]] = field(default_factory=dict)
    halted_at_step: int | None = None
    halt_reason: str | None = None
    synthesis_reply: ModelReply | None = None


# --- Seam protocols ----------------------------------------------------------


class ContextManager(Protocol):
    """Assembles the per-turn model context (history, memory, capabilities).

    `assemble` is called once for the initial intent-recognition pass and a
    second time (with `execution` populated) when the loop wants the model to
    synthesize a natural-language reply from a freshly executed plan. The
    `execution` argument is optional so the seam stays one method — the
    presence of an `ExecutionResult` is what flips the implementation into
    synthesis-context mode.
    """

    async def assemble(self, turn: Turn, execution: ExecutionResult | None = None) -> ModelContext: ...


class ConversationLog(Protocol):
    """Append-only sink for completed turns.

    Narrower than `core.context_manager.ConversationHistory` — the loop only
    needs to record turns, not read them back. The history protocol used by
    `ContextManager` satisfies this writer-side view structurally.
    """

    async def append(self, entry: ConversationEntry) -> None: ...


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
    client, task executor, conversation log) and is invoked once per turn.
    The wiring is set at construction and never reassigned — invariant #1.

    `history` is optional purely so tests can wire a loop without caring
    about persistence; production composition always provides one.
    `_NullConversationLog` is the no-op default so the loop's history-append
    code path is uniform.
    """

    def __init__(
        self,
        context_manager: ContextManager,
        model: ModelClient,
        executor: TaskExecutor,
        history: ConversationLog | None = None,
    ) -> None:
        self._context_manager = context_manager
        self._model = model
        self._executor = executor
        self._history: ConversationLog = history if history is not None else _NullConversationLog()

    async def run_turn(self, user_input: str) -> TurnResult:
        """Run one turn through the fixed pipeline.

        Pipeline:
        1. Build turn, assemble intent-recognition context, call model.
        2. ModelReply → record to history, return.
        3. TaskPlan → executor.execute(plan).
        4. Halted execution → record halt reason to history, return.
        5. Successful execution → assemble synthesis context (includes the
           plan + outputs), call model again, expect a ModelReply, record to
           history, attach to ExecutionResult.synthesis_reply, return.

        Raises:
            ModelProtocolError: On invalid model output. Synthesis also raises
                this if the model returns a TaskPlan instead of a ModelReply
                — synthesis must not recursively plan.
        """
        turn = _build_turn(user_input)
        context = await self._context_manager.assemble(turn)
        output = await self._model.generate(context)

        if isinstance(output, ModelReply):
            await self._record(turn, output.text)
            return TurnResult(turn=turn, reply=output)

        execution = await self._executor.execute(output)

        if execution.halted_at_step is not None:
            await self._record(turn, execution.halt_reason)
            return TurnResult(turn=turn, executed_plan=execution)

        synthesis = await self._synthesize(turn, execution)
        execution_with_reply = ExecutionResult(
            plan=execution.plan,
            outputs=execution.outputs,
            halted_at_step=execution.halted_at_step,
            halt_reason=execution.halt_reason,
            synthesis_reply=synthesis,
        )
        await self._record(turn, synthesis.text)
        return TurnResult(turn=turn, executed_plan=execution_with_reply)

    async def _synthesize(self, turn: Turn, execution: ExecutionResult) -> ModelReply:
        """Second model call: synthesize a natural-language reply from tool outputs.

        Raises:
            ModelProtocolError: If the model returns a TaskPlan instead of a
                ModelReply. Recursive planning from synthesis is rejected —
                the model has just observed tool results and must reply.
        """
        context = await self._context_manager.assemble(turn, execution=execution)
        output = await self._model.generate(context)
        if not isinstance(output, ModelReply):
            raise ModelProtocolError(
                'synthesis pass must return a reply, not a plan — the model was asked to summarise tool outputs, not propose new actions',
            )
        return output

    async def _record(self, turn: Turn, assistant_reply: str | None) -> None:
        # Imported lazily so the loop module does not depend on context_manager
        # at import time — context_manager already depends on loop for value types.
        from butter_agent.core.context_manager import ConversationEntry

        await self._history.append(
            ConversationEntry(
                turn_id=turn.turn_id,
                user_input=turn.user_input,
                assistant_reply=assistant_reply,
                timestamp=turn.timestamp,
            ),
        )


class _NullConversationLog:
    """No-op `ConversationLog` used when `AgentLoop` is constructed without history."""

    async def append(self, entry: ConversationEntry) -> None:
        del entry


# --- Helpers -----------------------------------------------------------------


def _build_turn(user_input: str) -> Turn:
    """Construct a Turn from raw user input."""
    return Turn(
        turn_id=uuid.uuid4().hex,
        user_input=user_input,
        timestamp=time.time(),
    )
