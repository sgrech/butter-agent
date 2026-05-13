"""Tests for the agent core loop.

The loop's contract: same five steps every turn, no shape change, clean
discrimination between direct reply and task plan. Tests use stub
implementations of the three seam protocols.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from butter_agent.core.loop import (
    AgentLoop,
    ContextManager,
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
    calls: list[Turn]

    async def assemble(self, turn: Turn) -> ModelContext:
        self.calls.append(turn)
        return ModelContext(turn=turn, payload={'history': []})


@dataclass
class _StubModel:
    output: ModelOutput
    contexts_seen: list[ModelContext]

    async def generate(self, context: ModelContext) -> ModelOutput:
        self.contexts_seen.append(context)
        return self.output


@dataclass
class _StubExecutor:
    result: ExecutionResult
    plans_seen: list[TaskPlan]

    async def execute(self, plan: TaskPlan) -> ExecutionResult:
        self.plans_seen.append(plan)
        return self.result


def _wire(model_output: ModelOutput, executor_result: ExecutionResult | None = None) -> tuple[AgentLoop, _StubContextManager, _StubModel, _StubExecutor]:
    cm = _StubContextManager(calls=[])
    model = _StubModel(output=model_output, contexts_seen=[])
    placeholder_plan = TaskPlan(steps=())
    executor = _StubExecutor(
        result=executor_result or ExecutionResult(plan=placeholder_plan),
        plans_seen=[],
    )
    loop = AgentLoop(context_manager=cm, model=model, executor=executor)
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


async def test_task_plan_routes_to_executor() -> None:
    plan = TaskPlan(
        steps=(PlanStep(step=1, plugin='notes', capability='create', inputs={'body': 'x'}, gate='none'),),
    )
    expected_result = ExecutionResult(plan=plan, outputs={'note': {'id': 42}})
    loop, _, _, executor = _wire(plan, executor_result=expected_result)

    result = await loop.run_turn('save a note')

    assert result.reply is None
    assert result.executed_plan is expected_result
    assert executor.plans_seen == [plan]


async def test_turn_id_is_unique_per_turn() -> None:
    loop, cm, _, _ = _wire(ModelReply(text='ok'))

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


def test_protocols_are_runtime_introspectable() -> None:
    # Confirm the protocols are importable and named as expected — they are the
    # loop's stable seams.
    assert ContextManager.__name__ == 'ContextManager'
    assert ModelClient.__name__ == 'ModelClient'
    assert TaskExecutor.__name__ == 'TaskExecutor'
