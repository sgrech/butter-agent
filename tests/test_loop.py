"""Tests for the agent core loop.

The loop's contract: same five steps every turn, no shape change, clean
discrimination between direct reply and task plan. Tests use stub
implementations of the three seam protocols.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import pytest

from butter_agent.core.context_manager import ConversationEntry
from butter_agent.core.loop import (
    AgentLoop,
    ContextManager,
    ConversationLog,
    DiscoverySelection,
    ExecutionResult,
    ModelClient,
    ModelContext,
    ModelOutput,
    ModelProtocolError,
    ModelReply,
    PlanStep,
    TaskExecutor,
    TaskPlan,
    Turn,
)


@dataclass
class _StubContextManager:
    """Records every `assemble` call's (turn, execution, selection).

    `discovery_active` defaults False so the legacy single-pass tests are
    unchanged. Set it True to exercise the Tier-1 → Tier-2 path; the
    payload mirrors `DefaultContextManager` shape (a `plugin_index` key on
    the Tier-1 pass, `capabilities` on the Tier-2 pass) so assertions can
    distinguish the passes.
    """

    calls: list[Turn]
    discovery_active: bool = False
    assembled: list[tuple[ExecutionResult | None, DiscoverySelection | None]] = field(default_factory=list)

    async def assemble(
        self,
        turn: Turn,
        execution: ExecutionResult | None = None,
        selection: DiscoverySelection | None = None,
    ) -> ModelContext:
        self.calls.append(turn)
        self.assembled.append((execution, selection))
        # Mirror DefaultContextManager: history is a tuple, not a list.
        payload: dict[str, object] = {'history': ()}
        if execution is not None:
            payload['execution'] = execution
        elif selection is not None:
            payload['capabilities'] = ()
        elif self.discovery_active:
            payload['plugin_index'] = ()
        return ModelContext(turn=turn, payload=payload)


@dataclass
class _StubModel:
    """Queue of outputs returned in order across `generate()` calls.

    Holds a deque rather than a single value so the same stub can serve
    the intent-recognition pass and the synthesis pass with distinct
    outputs. Surfaces unexpected extra calls as `IndexError` from the
    underlying `deque.popleft()` — a noisy failure that points at the
    test's wiring rather than at the loop.
    """

    outputs: deque[ModelOutput]
    contexts_seen: list[ModelContext] = field(default_factory=list)

    async def generate(self, context: ModelContext) -> ModelOutput:
        self.contexts_seen.append(context)
        return self.outputs.popleft()


@dataclass
class _StubExecutor:
    result: ExecutionResult
    plans_seen: list[TaskPlan]

    async def execute(self, plan: TaskPlan) -> ExecutionResult:
        self.plans_seen.append(plan)
        return self.result


@dataclass
class _RecordingLog:
    """`ConversationLog` that captures every appended entry."""

    entries: list[ConversationEntry] = field(default_factory=list)

    async def append(self, entry: ConversationEntry) -> None:
        self.entries.append(entry)


def _wire(
    *model_outputs: ModelOutput,
    executor_result: ExecutionResult | None = None,
    history: ConversationLog | None = None,
    discovery_active: bool = False,
) -> tuple[AgentLoop, _StubContextManager, _StubModel, _StubExecutor]:
    cm = _StubContextManager(calls=[], discovery_active=discovery_active)
    model = _StubModel(outputs=deque(model_outputs))
    placeholder_plan = TaskPlan(steps=())
    executor = _StubExecutor(
        result=executor_result or ExecutionResult(plan=placeholder_plan),
        plans_seen=[],
    )
    loop = AgentLoop(context_manager=cm, model=model, executor=executor, history=history)
    return loop, cm, model, executor


async def test_direct_reply_returns_reply_and_does_not_invoke_executor() -> None:
    loop, cm, model, executor = _wire(ModelReply(text='hi there'))

    result = await loop.run_turn('hello')

    assert result.reply == ModelReply(text='hi there')
    assert result.executed_plan is None
    assert len(cm.calls) == 1
    assert cm.calls[0].user_input == 'hello'
    assert len(model.contexts_seen) == 1
    assert executor.plans_seen == []


async def test_task_plan_routes_to_executor_then_synthesizes() -> None:
    plan = TaskPlan(
        steps=(PlanStep(step=1, plugin='notes', capability='create', inputs={'body': 'x'}, gate='none'),),
    )
    raw_execution = ExecutionResult(plan=plan, outputs={'note': {'id': 42}})
    synthesis = ModelReply(text='Saved note 42.')
    loop, _cm, model, executor = _wire(plan, synthesis, executor_result=raw_execution)

    result = await loop.run_turn('save a note')

    assert result.reply is None
    assert executor.plans_seen == [plan]
    assert result.executed_plan is not None
    # Loop reconstructs ExecutionResult to attach the synthesis reply.
    assert result.executed_plan.outputs == {'note': {'id': 42}}
    assert result.executed_plan.synthesis_reply == synthesis
    # Two model calls: intent recognition + synthesis. The second context
    # carries the execution payload key.
    assert len(model.contexts_seen) == 2
    assert 'execution' not in model.contexts_seen[0].payload
    assert model.contexts_seen[1].payload['execution'] is raw_execution


async def test_halted_plan_skips_synthesis() -> None:
    # A halted plan must not trigger a second model call — the operator
    # already aborted, asking the model to "synthesize a reply from no
    # outputs" would be wasted latency at best and confusing at worst.
    plan = TaskPlan(
        steps=(PlanStep(step=1, plugin='notes', capability='create', inputs={}, gate='confirm'),),
    )
    halted = ExecutionResult(
        plan=plan,
        outputs={},
        halted_at_step=1,
        halt_reason="gate 'confirm' aborted at step 1",
    )
    loop, _, model, _ = _wire(plan, executor_result=halted)

    result = await loop.run_turn('do it')

    assert result.executed_plan is halted
    assert result.executed_plan.synthesis_reply is None
    # Exactly one model call — the intent-recognition pass — should have run.
    assert len(model.contexts_seen) == 1


async def test_failed_plan_runs_synthesis_with_failure_in_context() -> None:
    """A plugin-failed execution still runs synthesis so the model can apologise.

    Spec: `plugin-failure-recovery.md`. Contrast with `halted` (operator
    aborted — no synthesis) and `success` (synthesis to summarise
    outputs). Failure synthesises so the user gets a reply explaining
    what broke instead of a bare `[error] plan rejected: ...`.
    """
    plan = TaskPlan(
        steps=(PlanStep(step=1, plugin='clock', capability='diff', inputs={}, gate='none'),),
    )
    failed = ExecutionResult(
        plan=plan,
        outputs={},
        failed_at_step=1,
        failure_reason="plugin 'clock' capability 'diff' raised: bad input",
    )
    synthesis = ModelReply(text='Sorry, the diff step failed because the timestamps were not ISO strings.')
    loop, _cm, model, _ = _wire(plan, synthesis, executor_result=failed)

    result = await loop.run_turn('diff two times')

    assert result.executed_plan is not None
    assert result.executed_plan.failed_at_step == 1
    assert result.executed_plan.synthesis_reply == synthesis
    # Two model calls: intent + synthesis. Synthesis context carries the
    # failed execution so the prompt can render the FAILED marker.
    assert len(model.contexts_seen) == 2
    assert model.contexts_seen[1].payload['execution'] is failed


async def test_synthesis_rejects_recursive_plan() -> None:
    # If the model returns another plan during synthesis, that's a protocol
    # violation — synthesis must reply, not propose new actions.
    plan = TaskPlan(
        steps=(PlanStep(step=1, plugin='notes', capability='create', inputs={}, gate='none'),),
    )
    raw_execution = ExecutionResult(plan=plan, outputs={})
    loop, _, _, _ = _wire(plan, plan, executor_result=raw_execution)

    with pytest.raises(ModelProtocolError, match='synthesis pass must return a reply'):
        await loop.run_turn('go')


async def test_history_records_direct_reply() -> None:
    log = _RecordingLog()
    loop, _, _, _ = _wire(ModelReply(text='hello back'), history=log)

    await loop.run_turn('hi')

    assert len(log.entries) == 1
    assert log.entries[0].user_input == 'hi'
    assert log.entries[0].assistant_reply == 'hello back'


async def test_history_records_synthesized_reply_not_raw_outputs() -> None:
    # The synthesized reply is the assistant's natural-language surface and
    # is what subsequent turns should be able to recall. Raw plugin outputs
    # belong in the conversation only via the model's paraphrase of them.
    plan = TaskPlan(
        steps=(PlanStep(step=1, plugin='clock', capability='now', inputs={}, gate='none'),),
    )
    raw_execution = ExecutionResult(plan=plan, outputs={'t': {'time': '17:00'}})
    log = _RecordingLog()
    loop, _, _, _ = _wire(plan, ModelReply(text='It is 17:00.'), executor_result=raw_execution, history=log)

    await loop.run_turn('what time is it')

    assert len(log.entries) == 1
    assert log.entries[0].assistant_reply == 'It is 17:00.'


async def test_history_records_halt_reason() -> None:
    plan = TaskPlan(
        steps=(PlanStep(step=1, plugin='notes', capability='create', inputs={}, gate='confirm'),),
    )
    halted = ExecutionResult(
        plan=plan,
        outputs={},
        halted_at_step=1,
        halt_reason="gate 'confirm' aborted at step 1",
    )
    log = _RecordingLog()
    loop, _, _, _ = _wire(plan, executor_result=halted, history=log)

    await loop.run_turn('go')

    assert len(log.entries) == 1
    assert log.entries[0].assistant_reply == "gate 'confirm' aborted at step 1"


async def test_turn_id_is_unique_per_turn() -> None:
    loop, cm, _, _ = _wire(ModelReply(text='ok'), ModelReply(text='ok'))

    await loop.run_turn('first')
    await loop.run_turn('second')

    assert len(cm.calls) == 2
    assert cm.calls[0].turn_id != cm.calls[1].turn_id


async def test_model_protocol_error_propagates() -> None:
    class _BadModel:
        async def generate(self, context: ModelContext) -> ModelOutput:
            raise ModelProtocolError('malformed output')

    cm = _StubContextManager(calls=[])
    executor = _StubExecutor(result=ExecutionResult(plan=TaskPlan(steps=())), plans_seen=[])
    loop = AgentLoop(context_manager=cm, model=_BadModel(), executor=executor)

    with pytest.raises(ModelProtocolError, match='malformed output'):
        await loop.run_turn('hi')

    assert executor.plans_seen == []


async def test_discovery_selection_drives_tier2_plan_then_synthesis() -> None:
    # Discovery active: model picks plugins (Tier-1), the loop re-assembles
    # with that selection (Tier-2), the model plans, executor runs, synthesis
    # replies. Three model calls; the second assemble carries the selection.
    selection = DiscoverySelection(plugins=('filesystem',))
    plan = TaskPlan(
        steps=(PlanStep(step=1, plugin='filesystem', capability='read_file', inputs={'path': 'pyproject.toml'}, gate='none'),),
    )
    raw_execution = ExecutionResult(plan=plan, outputs={'f': {'content': 'x'}})
    synthesis = ModelReply(text='pyproject lists pytest.')
    loop, cm, model, executor = _wire(
        selection,
        plan,
        synthesis,
        executor_result=raw_execution,
        discovery_active=True,
    )

    result = await loop.run_turn('what dependencies does pyproject have')

    assert executor.plans_seen == [plan]
    assert result.executed_plan is not None
    assert result.executed_plan.synthesis_reply == synthesis
    # discovery → planning → synthesis.
    assert len(model.contexts_seen) == 3
    # assemble: Tier-1 (no execution/selection), Tier-2 (selection set),
    # synthesis (execution set).
    assert cm.assembled[0] == (None, None)
    assert cm.assembled[1] == (None, selection)
    assert cm.assembled[2][0] is raw_execution
    assert 'plugin_index' in model.contexts_seen[0].payload
    assert 'capabilities' in model.contexts_seen[1].payload


async def test_discovery_reply_short_circuits_without_planning() -> None:
    # A purely conversational turn: the model replies at the discovery pass
    # and the loop must not run a planning pass or the executor.
    loop, cm, model, executor = _wire(
        ModelReply(text='Hello!'),
        discovery_active=True,
    )

    result = await loop.run_turn('hi')

    assert result.reply == ModelReply(text='Hello!')
    assert executor.plans_seen == []
    assert len(model.contexts_seen) == 1
    assert cm.assembled == [(None, None)]


async def test_plan_from_discovery_pass_is_protocol_error() -> None:
    # The model cannot plan before being shown Tier-2 schemas.
    plan = TaskPlan(steps=(PlanStep(step=1, plugin='x', capability='y', inputs={}, gate='none'),))
    loop, _, _, executor = _wire(plan, discovery_active=True)

    with pytest.raises(ModelProtocolError, match='discovery pass must return a plugin selection'):
        await loop.run_turn('go')

    assert executor.plans_seen == []


async def test_discovery_selection_from_planning_pass_is_protocol_error() -> None:
    # Tier-2 must yield a reply or a plan — a second discovery selection
    # means the model planned against detail it was not shown.
    loop, _, _, _ = _wire(
        DiscoverySelection(plugins=('a',)),
        DiscoverySelection(plugins=('b',)),
        discovery_active=True,
    )

    with pytest.raises(ModelProtocolError, match='planning pass must return a reply or a plan'):
        await loop.run_turn('go')


async def test_discovery_selection_from_intent_pass_when_discovery_off_is_protocol_error() -> None:
    # Discovery off: the single intent pass must yield reply or plan. A
    # stray DiscoverySelection (model ignoring the prompt) is rejected
    # rather than silently falling through to the executor as a non-plan.
    loop, _, _, executor = _wire(DiscoverySelection(plugins=('a',)))

    with pytest.raises(ModelProtocolError, match='intent pass must return a reply or a plan'):
        await loop.run_turn('go')

    assert executor.plans_seen == []


def test_protocols_are_runtime_introspectable() -> None:
    # Confirm the protocols are importable and named as expected — they are the
    # loop's stable seams.
    assert ContextManager.__name__ == 'ContextManager'
    assert ModelClient.__name__ == 'ModelClient'
    assert TaskExecutor.__name__ == 'TaskExecutor'
    assert ConversationLog.__name__ == 'ConversationLog'
