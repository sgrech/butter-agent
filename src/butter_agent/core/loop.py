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


@dataclass(frozen=True, slots=True)
class DiscoverySelection:
    """The model's Tier-1 discovery response: the plugin(s) it intends to use.

    Emitted only on the discovery pass, and only when the context manager
    reports `discovery_active`. The loop turns this into a Tier-2 context —
    the named plugins' full capability schemas — and re-prompts the model
    for the actual plan. A discovery pass may instead return a `ModelReply`
    (purely conversational turn, no tools needed); a `TaskPlan` from the
    discovery pass is a protocol violation (the plan is built against
    Tier-2 detail the model has not been shown yet).

    `plugins` is the raw set of names the model asked for. The context
    manager is responsible for resolving them against the frozen registry
    and deciding the fallback when none resolve — the loop carries the
    value through without interpreting it (mirrors how it treats plans).
    """

    plugins: tuple[str, ...]


# Discriminated union: the model produces exactly one of these. `TaskPlan`
# and `ModelReply` are the terminal outputs of the intent/planning and
# synthesis passes; `DiscoverySelection` is the intermediate Tier-1 output
# that only the discovery pass may produce.
ModelOutput = ModelReply | TaskPlan | DiscoverySelection


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

    Three terminal states are encoded by which optional fields are set:

    - Success: only `plan` and `outputs` (all steps ran).
    - Halted: `halted_at_step` + `halt_reason` (a `confirm`/`human` gate
      aborted before the step ran). No synthesis runs — the user
      explicitly stopped the plan.
    - Failed: `failed_at_step` + `failure_reason` (a plugin raised mid-
      step). `outputs` contains whatever the prior steps produced.
      Synthesis still runs so the model can acknowledge the failure to
      the user. See `plugin-failure-recovery.md`.

    `synthesis_reply` is filled by the loop after the second model
    call. The executor never sets it.
    """

    plan: TaskPlan
    outputs: dict[str, dict[str, object]] = field(default_factory=dict)
    halted_at_step: int | None = None
    halt_reason: str | None = None
    failed_at_step: int | None = None
    failure_reason: str | None = None
    synthesis_reply: ModelReply | None = None


# --- Seam protocols ----------------------------------------------------------


class ContextManager(Protocol):
    """Assembles the per-turn model context (history, memory, capabilities).

    `assemble` is called for up to three passes, distinguished by which
    optional argument is set so the seam stays one method:

    - neither set → the first pass. When `discovery_active` is false this is
      the keyword-filtered intent pass (capabilities in context). When true
      it is the Tier-1 discovery pass (a compact plugin index, no
      capabilities) and the loop expects a `DiscoverySelection` back.
    - `selection` set → the Tier-2 planning pass: full capability schemas
      for exactly the plugins the model named in discovery.
    - `execution` set → the synthesis pass: the executed plan + outputs, no
      capabilities (the model must reply, not plan).

    `discovery_active` is fixed for the process — it depends only on the
    frozen registry (invariant #2) and startup config, never on the turn —
    so the loop reads it once-per-turn but the value never changes. It is
    what tells the loop whether the first pass is a discovery round-trip or
    the legacy single intent pass; the loop's two code paths are both
    static, selected by a startup-resolved boolean (invariant #1: shape
    fixed per process, not runtime-reconfigured).
    """

    @property
    def discovery_active(self) -> bool: ...

    async def assemble(
        self,
        turn: Turn,
        execution: ExecutionResult | None = None,
        selection: DiscoverySelection | None = None,
    ) -> ModelContext: ...


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
        1. Build turn, resolve the planning output (`_plan`): either the
           legacy single intent pass, or — when discovery is active — the
           Tier-1 → Tier-2 discovery round-trip. Both shapes are static and
           selected by the process-fixed `discovery_active` (invariant #1).
        2. ModelReply → record to history, return.
        3. TaskPlan → executor.execute(plan).
        4. Halted execution → record halt reason to history, return.
        5. Successful/failed execution → assemble synthesis context (the
           plan + outputs), call model again, expect a ModelReply, record to
           history, attach to ExecutionResult.synthesis_reply, return.

        Raises:
            ModelProtocolError: On invalid model output (a plan from the
                discovery pass, a discovery selection from the intent/
                planning pass, or a plan from synthesis — synthesis must not
                recursively plan).
        """
        turn = _build_turn(user_input)
        output = await self._plan(turn)

        if isinstance(output, ModelReply):
            await self._record(turn, output.text)
            return TurnResult(turn=turn, reply=output)

        execution = await self._executor.execute(output)

        if execution.halted_at_step is not None:
            # Halted by a gate (user aborted). No synthesis — the user
            # explicitly stopped the plan and does not need a recap.
            await self._record(turn, execution.halt_reason)
            return TurnResult(turn=turn, executed_plan=execution)

        # Successful or failed executions both run synthesis. On failure
        # the synthesis prompt surfaces the failed step so the model can
        # acknowledge it; see plugin-failure-recovery.md.
        synthesis = await self._synthesize(turn, execution)
        execution_with_reply = ExecutionResult(
            plan=execution.plan,
            outputs=execution.outputs,
            halted_at_step=execution.halted_at_step,
            halt_reason=execution.halt_reason,
            failed_at_step=execution.failed_at_step,
            failure_reason=execution.failure_reason,
            synthesis_reply=synthesis,
        )
        await self._record(turn, synthesis.text)
        return TurnResult(turn=turn, executed_plan=execution_with_reply)

    async def _plan(self, turn: Turn) -> ModelReply | TaskPlan:
        """Resolve the turn to a reply or a plan, inserting discovery if active.

        Discovery off (legacy shape, byte-for-byte the prior behaviour):
        one `assemble` → one `generate`.

        Discovery on: Tier-1 `assemble` → `generate`. A `ModelReply` ends
        the turn (no tools needed). A `DiscoverySelection` triggers a
        Tier-2 `assemble(selection=...)` → `generate` for the actual plan.
        The skip-when-trivial mitigation lives in the context manager, not
        here: when the install is too small to benefit it reports
        `discovery_active == False`, so this method never pays the extra
        round-trip for trivial registries.

        Raises:
            ModelProtocolError: A `DiscoverySelection` from a non-discovery
                pass, or a `TaskPlan` from the discovery pass — both mean
                the model planned against detail it was not shown.
        """
        if self._context_manager.discovery_active:
            discovery_ctx = await self._context_manager.assemble(turn)
            discovery_out = await self._model.generate(discovery_ctx)
            if isinstance(discovery_out, ModelReply):
                return discovery_out
            if not isinstance(discovery_out, DiscoverySelection):
                raise ModelProtocolError(
                    'discovery pass must return a plugin selection or a reply, not a plan — the model has not been shown capability schemas yet',
                )
            plan_ctx = await self._context_manager.assemble(turn, selection=discovery_out)
            return _expect_reply_or_plan(await self._model.generate(plan_ctx), pass_name='planning')

        context = await self._context_manager.assemble(turn)
        return _expect_reply_or_plan(await self._model.generate(context), pass_name='intent')

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


def _expect_reply_or_plan(output: ModelOutput, *, pass_name: str) -> ModelReply | TaskPlan:
    """Narrow an intent/planning-pass output, rejecting a stray DiscoverySelection.

    Only the discovery pass may emit a `DiscoverySelection`. Seeing one
    here means the model was prompted for a plan but answered with a
    plugin pick — surface it as a protocol error rather than letting it
    fall through to the executor as a non-plan.
    """
    if isinstance(output, ModelReply | TaskPlan):
        return output
    raise ModelProtocolError(
        f'{pass_name} pass must return a reply or a plan, not a discovery selection',
    )
