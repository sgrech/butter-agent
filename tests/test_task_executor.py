"""Tests for the task executor.

Covers the three invariants this module owns: atomic plan validation (#3),
dict-lookup variable resolution (#4), and gate enforcement including network
blast-radius minimum-gate injection (#5/#7).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

import pytest

from butter_agent.core.loop import PlanStep, TaskPlan
from butter_agent.core.registry import (
    BlastRadius,
    Capability,
    PluginManifest,
    RegistryBuilder,
)
from butter_agent.core.task_executor import (
    AlwaysContinueGateHandler,
    DefaultTaskExecutor,
    Gate,
    GateDecision,
    GateHandler,
    PlanValidationError,
    VariableResolutionError,
)

# --- Plugin / registry test helpers -----------------------------------------


@dataclass
class _RecordingPlugin:
    """Plugin stub that records every (capability, inputs) it was asked to run."""

    responses: dict[str, dict[str, object]]
    calls: list[tuple[str, dict[str, object]]] = field(default_factory=list)

    async def execute(self, capability: str, inputs: dict[str, object]) -> dict[str, object]:
        self.calls.append((capability, dict(inputs)))
        return dict(self.responses.get(capability, {}))


def _manifest(
    name: str,
    *,
    radius: BlastRadius = BlastRadius.READ_ONLY,
    capabilities: tuple[Capability, ...],
) -> PluginManifest:
    return PluginManifest(
        name=name,
        version='0.1.0',
        blast_radius=radius,
        entrypoint='stub:Plugin',
        capabilities=capabilities,
    )


def _cap(name: str, input_schema: dict[str, object] | None = None, output_schema: dict[str, object] | None = None) -> Capability:
    return Capability(
        name=name,
        description=f'{name} capability',
        input_schema=input_schema or {},
        output_schema=output_schema or {},
    )


def _make_executor(
    *pairs: tuple[PluginManifest, _RecordingPlugin],
    gate_handler: GateHandler | None = None,
) -> DefaultTaskExecutor:
    builder = RegistryBuilder(max_blast_radius=BlastRadius.NETWORK)
    for manifest, plugin in pairs:
        builder.register(manifest, plugin)
    registry = builder.build()
    return DefaultTaskExecutor(registry=registry, gate_handler=gate_handler or AlwaysContinueGateHandler())


# --- Happy path -------------------------------------------------------------


async def test_single_step_plan_executes_and_records_output() -> None:
    notes = _RecordingPlugin(responses={'create': {'id': 7}})
    manifest = _manifest('notes', capabilities=(_cap('create', input_schema={'body': 'string'}),))
    executor = _make_executor((manifest, notes))

    plan = TaskPlan(steps=(PlanStep(step=1, plugin='notes', capability='create', inputs={'body': 'hi'}, gate='none', outputs_as='note'),))

    result = await executor.execute(plan)

    assert notes.calls == [('create', {'body': 'hi'})]
    assert result.outputs == {'note': {'id': 7}}
    assert result.halted_at_step is None
    assert result.halt_reason is None


async def test_multi_step_plan_walks_all_steps_in_order() -> None:
    a = _RecordingPlugin(responses={'do': {'x': 1}})
    b = _RecordingPlugin(responses={'do': {'y': 2}})
    executor = _make_executor(
        (_manifest('a', capabilities=(_cap('do'),)), a),
        (_manifest('b', capabilities=(_cap('do'),)), b),
    )

    plan = TaskPlan(
        steps=(
            PlanStep(step=1, plugin='a', capability='do', inputs={}, gate='none', outputs_as='first'),
            PlanStep(step=2, plugin='b', capability='do', inputs={}, gate='none', outputs_as='second'),
        )
    )

    result = await executor.execute(plan)

    assert a.calls == [('do', {})]
    assert b.calls == [('do', {})]
    assert result.outputs == {'first': {'x': 1}, 'second': {'y': 2}}


# --- Atomic validation (invariant #3) ---------------------------------------


async def test_validation_rejects_unknown_plugin_before_executing_anything() -> None:
    plugin = _RecordingPlugin(responses={'do': {}})
    executor = _make_executor(
        (_manifest('real', capabilities=(_cap('do'),)), plugin),
    )

    plan = TaskPlan(
        steps=(
            PlanStep(step=1, plugin='real', capability='do', inputs={}, gate='none'),
            PlanStep(step=2, plugin='ghost', capability='do', inputs={}, gate='none'),
        )
    )

    with pytest.raises(PlanValidationError, match='ghost'):
        await executor.execute(plan)

    # Atomic: first step's plugin must not have run.
    assert plugin.calls == []


async def test_validation_rejects_unknown_capability() -> None:
    plugin = _RecordingPlugin(responses={'do': {}})
    executor = _make_executor(
        (_manifest('real', capabilities=(_cap('do'),)), plugin),
    )

    plan = TaskPlan(steps=(PlanStep(step=1, plugin='real', capability='nope', inputs={}, gate='none'),))

    with pytest.raises(PlanValidationError, match='nope'):
        await executor.execute(plan)
    assert plugin.calls == []


async def test_validation_rejects_invalid_gate() -> None:
    plugin = _RecordingPlugin(responses={'do': {}})
    executor = _make_executor(
        (_manifest('real', capabilities=(_cap('do'),)), plugin),
    )

    plan = TaskPlan(steps=(PlanStep(step=1, plugin='real', capability='do', inputs={}, gate='whatever'),))

    with pytest.raises(PlanValidationError, match='invalid gate'):
        await executor.execute(plan)
    assert plugin.calls == []


async def test_validation_rejects_missing_required_input() -> None:
    plugin = _RecordingPlugin(responses={'create': {}})
    executor = _make_executor(
        (_manifest('notes', capabilities=(_cap('create', input_schema={'body': 'string'}),)), plugin),
    )

    plan = TaskPlan(steps=(PlanStep(step=1, plugin='notes', capability='create', inputs={}, gate='none'),))

    with pytest.raises(PlanValidationError, match="missing required input 'body'"):
        await executor.execute(plan)
    assert plugin.calls == []


async def test_validation_rejects_reference_to_unknown_alias() -> None:
    plugin = _RecordingPlugin(responses={'do': {}})
    executor = _make_executor(
        (_manifest('notes', capabilities=(_cap('do', input_schema={'body': 'string'}),)), plugin),
    )

    plan = TaskPlan(steps=(PlanStep(step=1, plugin='notes', capability='do', inputs={'body': '$booking.summary'}, gate='none'),))

    with pytest.raises(PlanValidationError, match='unknown alias'):
        await executor.execute(plan)
    assert plugin.calls == []


async def test_validation_rejects_non_sequential_step_numbers() -> None:
    plugin = _RecordingPlugin(responses={'do': {}})
    executor = _make_executor(
        (_manifest('p', capabilities=(_cap('do'),)), plugin),
    )

    plan = TaskPlan(
        steps=(
            PlanStep(step=1, plugin='p', capability='do', inputs={}, gate='none'),
            PlanStep(step=3, plugin='p', capability='do', inputs={}, gate='none'),
        )
    )

    with pytest.raises(PlanValidationError, match='sequential'):
        await executor.execute(plan)
    assert plugin.calls == []


async def test_validation_rejects_duplicate_outputs_as_alias() -> None:
    plugin = _RecordingPlugin(responses={'do': {}})
    executor = _make_executor(
        (_manifest('p', capabilities=(_cap('do'),)), plugin),
    )

    plan = TaskPlan(
        steps=(
            PlanStep(step=1, plugin='p', capability='do', inputs={}, gate='none', outputs_as='r'),
            PlanStep(step=2, plugin='p', capability='do', inputs={}, gate='none', outputs_as='r'),
        )
    )

    with pytest.raises(PlanValidationError, match='already declared'):
        await executor.execute(plan)
    assert plugin.calls == []


async def test_validation_rejects_empty_plan() -> None:
    executor = _make_executor()
    with pytest.raises(PlanValidationError, match='no steps'):
        await executor.execute(TaskPlan(steps=()))


# --- Variable resolution (invariant #4) -------------------------------------


async def test_variable_reference_is_resolved_by_dict_lookup() -> None:
    book = _RecordingPlugin(responses={'book': {'departure_time': '08:00', 'summary': 'LHR->JFK'}})
    cal = _RecordingPlugin(responses={'create_event': {'event_id': 'e1'}})
    executor = _make_executor(
        (_manifest('flight', capabilities=(_cap('book'),)), book),
        (_manifest('calendar', capabilities=(_cap('create_event', input_schema={'start_time': 'string'}),)), cal),
    )

    plan = TaskPlan(
        steps=(
            PlanStep(step=1, plugin='flight', capability='book', inputs={}, gate='none', outputs_as='booking'),
            PlanStep(
                step=2,
                plugin='calendar',
                capability='create_event',
                inputs={'start_time': '$booking.departure_time'},
                gate='none',
            ),
        )
    )

    await executor.execute(plan)

    assert cal.calls == [('create_event', {'start_time': '08:00'})]


async def test_missing_field_in_resolved_alias_raises_variable_resolution_error() -> None:
    book = _RecordingPlugin(responses={'book': {'summary': 'no time field'}})
    cal = _RecordingPlugin(responses={'create_event': {}})
    executor = _make_executor(
        (_manifest('flight', capabilities=(_cap('book'),)), book),
        (_manifest('calendar', capabilities=(_cap('create_event', input_schema={'start_time': 'string'}),)), cal),
    )

    plan = TaskPlan(
        steps=(
            PlanStep(step=1, plugin='flight', capability='book', inputs={}, gate='none', outputs_as='booking'),
            PlanStep(
                step=2,
                plugin='calendar',
                capability='create_event',
                inputs={'start_time': '$booking.departure_time'},
                gate='none',
            ),
        )
    )

    with pytest.raises(VariableResolutionError, match='departure_time'):
        await executor.execute(plan)

    # Step 1 ran (its output was recorded); step 2 raised before invoking the plugin.
    assert book.calls == [('book', {})]
    assert cal.calls == []


async def test_non_string_and_non_matching_string_inputs_pass_through_unchanged() -> None:
    plugin = _RecordingPlugin(responses={'do': {}})
    executor = _make_executor(
        (_manifest('p', capabilities=(_cap('do'),)), plugin),
    )

    plan = TaskPlan(
        steps=(
            PlanStep(
                step=1,
                plugin='p',
                capability='do',
                inputs={'count': 3, 'literal_dollar': '$not a ref', 'text': 'hello'},
                gate='none',
            ),
        )
    )

    await executor.execute(plan)

    assert plugin.calls == [('do', {'count': 3, 'literal_dollar': '$not a ref', 'text': 'hello'})]


# --- Gate enforcement (invariants #5, #7) -----------------------------------


@dataclass
class _RecordingGateHandler:
    decisions: list[GateDecision]
    seen: list[tuple[int, Gate]] = field(default_factory=list)

    async def on_gate(
        self,
        step: PlanStep,
        effective_gate: Gate,
        prior_outputs: Mapping[str, Mapping[str, object]],
    ) -> GateDecision:
        self.seen.append((step.step, effective_gate))
        return self.decisions[len(self.seen) - 1]


async def test_none_gate_does_not_call_handler() -> None:
    plugin = _RecordingPlugin(responses={'do': {}})
    handler = _RecordingGateHandler(decisions=[])
    executor = _make_executor(
        (_manifest('p', capabilities=(_cap('do'),)), plugin),
        gate_handler=handler,
    )

    plan = TaskPlan(steps=(PlanStep(step=1, plugin='p', capability='do', inputs={}, gate='none'),))

    await executor.execute(plan)

    assert handler.seen == []
    assert plugin.calls == [('do', {})]


async def test_confirm_gate_consults_handler_and_runs_on_continue() -> None:
    plugin = _RecordingPlugin(responses={'do': {}})
    handler = _RecordingGateHandler(decisions=[GateDecision.CONTINUE])
    executor = _make_executor(
        (_manifest('p', capabilities=(_cap('do'),)), plugin),
        gate_handler=handler,
    )

    plan = TaskPlan(steps=(PlanStep(step=1, plugin='p', capability='do', inputs={}, gate='confirm'),))

    await executor.execute(plan)

    assert handler.seen == [(1, Gate.CONFIRM)]
    assert plugin.calls == [('do', {})]


async def test_human_gate_abort_halts_plan_before_running_step() -> None:
    plugin_a = _RecordingPlugin(responses={'do': {'k': 1}})
    plugin_b = _RecordingPlugin(responses={'do': {}})
    handler = _RecordingGateHandler(decisions=[GateDecision.ABORT])
    executor = _make_executor(
        (_manifest('a', capabilities=(_cap('do'),)), plugin_a),
        (_manifest('b', capabilities=(_cap('do'),)), plugin_b),
        gate_handler=handler,
    )

    plan = TaskPlan(
        steps=(
            PlanStep(step=1, plugin='a', capability='do', inputs={}, gate='none', outputs_as='first'),
            PlanStep(step=2, plugin='b', capability='do', inputs={}, gate='human'),
        )
    )

    result = await executor.execute(plan)

    assert plugin_a.calls == [('do', {})]
    assert plugin_b.calls == []
    assert result.halted_at_step == 2
    assert result.halt_reason is not None and 'human' in result.halt_reason
    assert result.outputs == {'first': {'k': 1}}


async def test_network_blast_radius_upgrades_none_gate_to_confirm() -> None:
    plugin = _RecordingPlugin(responses={'search': {}})
    handler = _RecordingGateHandler(decisions=[GateDecision.CONTINUE])
    executor = _make_executor(
        (_manifest('flight', radius=BlastRadius.NETWORK, capabilities=(_cap('search'),)), plugin),
        gate_handler=handler,
    )

    plan = TaskPlan(steps=(PlanStep(step=1, plugin='flight', capability='search', inputs={}, gate='none'),))

    await executor.execute(plan)

    assert handler.seen == [(1, Gate.CONFIRM)]
    assert plugin.calls == [('search', {})]


async def test_network_plugin_with_human_gate_keeps_human_gate() -> None:
    plugin = _RecordingPlugin(responses={'search': {}})
    handler = _RecordingGateHandler(decisions=[GateDecision.CONTINUE])
    executor = _make_executor(
        (_manifest('flight', radius=BlastRadius.NETWORK, capabilities=(_cap('search'),)), plugin),
        gate_handler=handler,
    )

    plan = TaskPlan(steps=(PlanStep(step=1, plugin='flight', capability='search', inputs={}, gate='human'),))

    await executor.execute(plan)

    assert handler.seen == [(1, Gate.HUMAN)]


async def test_non_network_plugin_keeps_declared_none_gate() -> None:
    plugin = _RecordingPlugin(responses={'do': {}})
    handler = _RecordingGateHandler(decisions=[])
    executor = _make_executor(
        (_manifest('p', radius=BlastRadius.LOCAL_WRITE, capabilities=(_cap('do'),)), plugin),
        gate_handler=handler,
    )

    plan = TaskPlan(steps=(PlanStep(step=1, plugin='p', capability='do', inputs={}, gate='none'),))

    await executor.execute(plan)

    assert handler.seen == []
    assert plugin.calls == [('do', {})]
