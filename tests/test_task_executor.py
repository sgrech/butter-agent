"""Tests for the task executor.

Covers the three invariants this module owns: atomic plan validation (#3),
dict-lookup variable resolution (#4), and gate enforcement including network
blast-radius minimum-gate injection (#5/#7).
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass, field

import pytest

from butter_agent.core.loop import PlanStep, TaskPlan
from butter_agent.core.registry import (
    BlastRadius,
    Capability,
    Plugin,
    PluginContext,
    PluginManifest,
    PluginRegistry,
    RegisteredPlugin,
    RegistryBuilder,
)
from butter_agent.core.task_executor import (
    _DB_TABLE_KEY,
    _MAX_CTX_DEPTH,
    _NAMESPACED_DB_PLUGIN,
    AlwaysContinueGateHandler,
    DefaultTaskExecutor,
    Gate,
    GateDecision,
    GateHandler,
    PlanValidationError,
    PluginContextError,
    VariableResolutionError,
    _PluginContext,
)
from butter_agent.plugins.database import PLUGIN_NAME as _DB_PLUGIN_NAME
from tests.support import FakePluginContext

# --- Plugin / registry test helpers -----------------------------------------


@dataclass
class _RecordingPlugin:
    """Plugin stub that records every (capability, inputs) it was asked to run."""

    responses: dict[str, dict[str, object]]
    calls: list[tuple[str, dict[str, object]]] = field(default_factory=list)

    async def execute(
        self,
        capability: str,
        inputs: dict[str, object],
        context: object,
    ) -> dict[str, object]:
        del context
        self.calls.append((capability, dict(inputs)))
        return dict(self.responses.get(capability, {}))


def _manifest(
    name: str,
    *,
    radius: BlastRadius = BlastRadius.READ_ONLY,
    capabilities: tuple[Capability, ...],
    requires: tuple[str, ...] = (),
) -> PluginManifest:
    return PluginManifest(
        name=name,
        version='0.1.0',
        blast_radius=radius,
        entrypoint='stub:Plugin',
        capabilities=capabilities,
        requires=requires,
    )


def _cap(
    name: str,
    input_schema: dict[str, object] | None = None,
    output_schema: dict[str, object] | None = None,
    *,
    internal: bool = False,
) -> Capability:
    return Capability(
        name=name,
        description=f'{name} capability',
        input_schema=input_schema or {},
        output_schema=output_schema or {},
        internal=internal,
    )


def _make_executor(
    *pairs: tuple[PluginManifest, Plugin],
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


async def test_executor_passes_plugin_context_to_execute() -> None:
    """Every `execute` call gets a real, per-invocation `PluginContext`.

    Slice 2 replaced the raising stub with a context that dispatches for
    real. The Protocol is three-arg; the executor must supply a context
    that satisfies it.
    """

    captured: list[PluginContext] = []

    class _ContextCapturingPlugin:
        async def execute(
            self,
            capability: str,
            inputs: dict[str, object],
            context: PluginContext,
        ) -> dict[str, object]:
            del capability, inputs
            captured.append(context)
            return {}

    manifest = _manifest('notes', capabilities=(_cap('create'),))
    executor = _make_executor((manifest, _ContextCapturingPlugin()))
    plan = TaskPlan(
        steps=(PlanStep(step=1, plugin='notes', capability='create', inputs={}, gate='none'),),
    )
    await executor.execute(plan)
    assert len(captured) == 1
    assert callable(captured[0].call)


# --- PluginContext dispatch (invariant #6, task #381 slice 2) ----------------


@dataclass
class _CallingPlugin:
    """Plugin that, on its declared capability, makes one `ctx.call` and
    returns whatever the internal capability produced."""

    target: str
    inner_inputs: dict[str, object]
    seen: list[dict[str, object]] = field(default_factory=list)

    async def execute(
        self,
        capability: str,
        inputs: dict[str, object],
        context: PluginContext,
    ) -> dict[str, object]:
        del capability, inputs
        result = await context.call(self.target, self.inner_inputs)
        self.seen.append(result)
        return result


async def test_plugin_context_call_dispatches_to_declared_internal_capability() -> None:
    """A plugin that declared `requires` reaches the internal capability and
    gets its output back, while the internal cap stays out of the plan."""
    database = _RecordingPlugin(responses={'insert': {'id': 42}})
    notes = _CallingPlugin(target='database.insert', inner_inputs={'row': {'body': 'hi'}})
    executor = _make_executor(
        (
            _manifest(
                'database',
                radius=BlastRadius.LOCAL_WRITE,
                capabilities=(_cap('insert', input_schema={'row': 'object'}, internal=True),),
            ),
            database,
        ),
        (
            _manifest(
                'notes',
                radius=BlastRadius.LOCAL_WRITE,
                capabilities=(_cap('create', input_schema={'body': 'string'}),),
                requires=('database.insert',),
            ),
            notes,
        ),
    )

    plan = TaskPlan(
        steps=(PlanStep(step=1, plugin='notes', capability='create', inputs={'body': 'hi'}, gate='none', outputs_as='note'),),
    )
    result = await executor.execute(plan)

    assert database.calls == [('insert', {'row': {'body': 'hi'}})]
    assert notes.seen == [{'id': 42}]
    assert result.outputs == {'note': {'id': 42}}
    assert result.failed_at_step is None


async def test_plugin_context_call_to_undeclared_capability_is_recorded_as_failure() -> None:
    """Calling a capability not in the plugin's `requires` is a plugin-author
    bug — surfaced as a step failure (invariant #6), never an executor crash.

    `PluginContextError` is deliberately not an `ExecutorError`, so it flows
    through the same path as any other third-party plugin exception: recorded
    on the result so the loop can synthesise instead of tearing down.
    """
    database = _RecordingPlugin(responses={'insert': {'id': 1}})
    # `notes` declares NO requires, yet tries to call database.insert.
    notes = _CallingPlugin(target='database.insert', inner_inputs={'row': {}})
    executor = _make_executor(
        (
            _manifest(
                'database',
                radius=BlastRadius.LOCAL_WRITE,
                capabilities=(_cap('insert', internal=True),),
            ),
            database,
        ),
        (
            _manifest('notes', radius=BlastRadius.LOCAL_WRITE, capabilities=(_cap('create'),)),
            notes,
        ),
    )

    plan = TaskPlan(
        steps=(PlanStep(step=1, plugin='notes', capability='create', inputs={}, gate='none'),),
    )
    result = await executor.execute(plan)

    assert result.failed_at_step == 1
    assert result.failure_reason is not None
    assert 'PluginContextError' in result.failure_reason
    assert 'database.insert' in result.failure_reason
    # The internal plugin was never reached — enforcement happens before dispatch.
    assert database.calls == []


async def test_plugin_context_dispatch_is_recursive_with_per_hop_requires() -> None:
    """`a -> b -> c`: each hop gets a fresh context restricted to its own
    `requires`. The chain composes and the outermost output propagates."""
    c = _RecordingPlugin(responses={'leaf': {'value': 'deep'}})
    b = _CallingPlugin(target='c.leaf', inner_inputs={})
    a = _CallingPlugin(target='b.mid', inner_inputs={})
    executor = _make_executor(
        (
            _manifest('c', radius=BlastRadius.LOCAL_WRITE, capabilities=(_cap('leaf', internal=True),)),
            c,
        ),
        (
            _manifest(
                'b',
                radius=BlastRadius.LOCAL_WRITE,
                capabilities=(_cap('mid', internal=True),),
                requires=('c.leaf',),
            ),
            b,
        ),
        (
            _manifest(
                'a',
                radius=BlastRadius.LOCAL_WRITE,
                capabilities=(_cap('top'),),
                requires=('b.mid',),
            ),
            a,
        ),
    )

    plan = TaskPlan(
        steps=(PlanStep(step=1, plugin='a', capability='top', inputs={}, gate='none', outputs_as='r'),),
    )
    result = await executor.execute(plan)

    assert c.calls == [('leaf', {})]
    assert b.seen == [{'value': 'deep'}]
    assert a.seen == [{'value': 'deep'}]
    assert result.outputs == {'r': {'value': 'deep'}}


def _registry(*pairs: tuple[PluginManifest, Plugin]) -> PluginRegistry:
    builder = RegistryBuilder(max_blast_radius=BlastRadius.NETWORK)
    for manifest, plugin in pairs:
        builder.register(manifest, plugin)
    return builder.build()


def _unchecked_registry(*pairs: tuple[PluginManifest, Plugin]) -> PluginRegistry:
    """Build a `PluginRegistry` without `RegistryBuilder.build()` validation.

    Used only to exercise `_PluginContext`'s runtime defence-in-depth guard,
    which is otherwise unreachable because build-time validation rejects a
    `requires` entry that points at a non-internal capability.
    """
    return PluginRegistry({m.name: RegisteredPlugin(manifest=m, plugin=p) for m, p in pairs})


async def test_plugin_context_call_raises_plugin_context_error_for_undeclared_target() -> None:
    """White-box: enforcement raises `PluginContextError` directly, before
    any dispatch. (The executor wraps this into a step failure; here we pin
    the exception type so a rename can't silently weaken the contract.)"""
    database = _RecordingPlugin(responses={'insert': {}})
    registry = _registry(
        (
            _manifest('database', radius=BlastRadius.LOCAL_WRITE, capabilities=(_cap('insert', internal=True),)),
            database,
        ),
        (_manifest('notes', capabilities=(_cap('create'),)), _RecordingPlugin(responses={})),
    )
    ctx = _PluginContext(owner='notes', registry=registry, step=1)

    with pytest.raises(PluginContextError, match='did not declare it in manifest requires'):
        await ctx.call('database.insert', {})
    assert database.calls == []


async def test_plugin_context_depth_ceiling_raises_before_dispatch() -> None:
    """Defence-in-depth: at the nesting ceiling, `call` fails fast with a
    `PluginContextError` (named owner + capability) instead of recursing to
    Python's recursion limit and surfacing an opaque deep `RecursionError`.

    Real cycles are unreachable (rejected at registry build), so the guard
    is exercised by constructing a context already at the ceiling depth.
    """
    database = _RecordingPlugin(responses={'insert': {}})
    registry = _registry(
        (
            _manifest('database', radius=BlastRadius.LOCAL_WRITE, capabilities=(_cap('insert', internal=True),)),
            database,
        ),
        (
            _manifest(
                'notes',
                radius=BlastRadius.LOCAL_WRITE,
                capabilities=(_cap('create'),),
                requires=('database.insert',),
            ),
            _RecordingPlugin(responses={}),
        ),
    )
    ctx = _PluginContext(owner='notes', registry=registry, step=1, depth=_MAX_CTX_DEPTH)

    with pytest.raises(PluginContextError, match='nesting exceeded'):
        await ctx.call('database.insert', {})
    # Guard fires before any dispatch — the target is never reached.
    assert database.calls == []


async def test_plugin_context_call_raises_for_non_internal_target() -> None:
    """Defence in depth: even if `requires` somehow names a non-internal cap
    (registry build should reject this), `call` refuses at runtime."""
    other = _RecordingPlugin(responses={'public': {}})
    # `requires` points at a non-internal cap. Construct the manifest
    # directly so we bypass RegistryBuilder's build-time rejection and
    # exercise the runtime guard in isolation.
    caller_manifest = _manifest('caller', capabilities=(_cap('go'),), requires=('other.public',))
    registry = _unchecked_registry(
        (_manifest('other', capabilities=(_cap('public', internal=False),)), other),
        (caller_manifest, _RecordingPlugin(responses={})),
    )
    ctx = _PluginContext(owner='caller', registry=registry, step=2)

    with pytest.raises(PluginContextError, match='not internal'):
        await ctx.call('other.public', {})
    assert other.calls == []


async def test_plugin_context_logs_nested_call_under_parent_step(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A nested `ctx.call` logs against its parent plan step, indented one
    level below the step line so a reader sees the call hierarchy."""
    database = _RecordingPlugin(responses={'insert': {'id': 1}})
    notes = _CallingPlugin(target='database.insert', inner_inputs={'row': {}})
    executor = _make_executor(
        (
            _manifest(
                'database',
                radius=BlastRadius.LOCAL_WRITE,
                capabilities=(_cap('insert', internal=True),),
            ),
            database,
        ),
        (
            _manifest(
                'notes',
                radius=BlastRadius.LOCAL_WRITE,
                capabilities=(_cap('create'),),
                requires=('database.insert',),
            ),
            notes,
        ),
    )

    plan = TaskPlan(
        steps=(PlanStep(step=1, plugin='notes', capability='create', inputs={}, gate='none'),),
    )
    with caplog.at_level(logging.DEBUG, logger='butter_agent.core.task_executor'):
        await executor.execute(plan)

    messages = [r.getMessage() for r in caplog.records]
    assert 'step 1: executing notes.create' in messages
    nested = '  step 1: notes -> database.insert'
    assert nested in messages
    # Nested line is indented past the parent step line.
    assert messages.index('step 1: executing notes.create') < messages.index(nested)


async def test_validation_rejects_plans_naming_internal_capability() -> None:
    """The model must never place an internal capability in a plan.

    Gates and variable-pool semantics don't apply to internal calls, so the
    only legal way to reach them is `PluginContext.call` from within another
    plugin. A plan that names one is malformed regardless of how it got
    emitted.
    """
    db = _RecordingPlugin(responses={'insert': {'id': 1}})
    manifest = _manifest(
        'database',
        radius=BlastRadius.LOCAL_WRITE,
        capabilities=(_cap('insert', input_schema={'row': 'object'}, internal=True),),
    )
    executor = _make_executor((manifest, db))

    plan = TaskPlan(
        steps=(PlanStep(step=1, plugin='database', capability='insert', inputs={'row': {}}, gate='none'),),
    )
    with pytest.raises(PlanValidationError, match='internal'):
        await executor.execute(plan)
    # And the plugin was never invoked.
    assert db.calls == []


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


async def test_gate_handler_receives_read_only_snapshot_of_prior_outputs() -> None:
    @dataclass
    class _MutatingHandler:
        captured: list[Mapping[str, Mapping[str, object]]] = field(default_factory=list)
        mutation_errors: list[type[BaseException]] = field(default_factory=list)

        async def on_gate(
            self,
            step: PlanStep,
            effective_gate: Gate,
            prior_outputs: Mapping[str, Mapping[str, object]],
        ) -> GateDecision:
            self.captured.append(prior_outputs)
            try:
                prior_outputs['first']['k'] = 99  # type: ignore[index]
            except TypeError as exc:
                self.mutation_errors.append(type(exc))
            return GateDecision.CONTINUE

    plugin_a = _RecordingPlugin(responses={'do': {'k': 1}})
    plugin_b = _RecordingPlugin(responses={'do': {}})
    handler = _MutatingHandler()
    executor = _make_executor(
        (_manifest('a', capabilities=(_cap('do'),)), plugin_a),
        (_manifest('b', capabilities=(_cap('do'),)), plugin_b),
        gate_handler=handler,
    )

    plan = TaskPlan(
        steps=(
            PlanStep(step=1, plugin='a', capability='do', inputs={}, gate='none', outputs_as='first'),
            PlanStep(step=2, plugin='b', capability='do', inputs={}, gate='confirm'),
        )
    )

    result = await executor.execute(plan)

    assert handler.mutation_errors == [TypeError]
    # The real output pool is untouched — the handler only ever saw a proxy.
    assert result.outputs == {'first': {'k': 1}}


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


# --- Plugin execution errors ------------------------------------------------


class _RaisingPlugin:
    """Plugin stub that raises an arbitrary exception from execute()."""

    def __init__(self, exc: Exception) -> None:
        self._exc = exc

    async def execute(
        self,
        capability: str,
        inputs: dict[str, object],
        context: object,
    ) -> dict[str, object]:
        del capability, inputs, context
        raise self._exc


async def test_plugin_runtime_exception_recorded_as_failure_not_raised() -> None:
    """Plugins are third-party code (invariant #6) and may raise for any reason.

    User-test on 2026-05-14: `clock.diff` raised ValueError mid-turn
    and the traceback tore the REPL down. New contract: the executor
    catches the exception, returns an `ExecutionResult` with
    `failed_at_step` / `failure_reason` set, and the loop runs
    synthesis so the model acknowledges the failure to the user. See
    `specs/development/plugin-failure-recovery.md`.
    """
    plugin = _RaisingPlugin(ValueError('bad input'))
    manifest = _manifest('clock', capabilities=(_cap('diff'),))
    executor = _make_executor((manifest, plugin))

    plan = TaskPlan(steps=(PlanStep(step=1, plugin='clock', capability='diff', inputs={}, gate='none', outputs_as=None),))

    result = await executor.execute(plan)

    assert result.failed_at_step == 1
    assert result.failure_reason is not None
    assert "plugin 'clock' capability 'diff' raised ValueError: bad input" in result.failure_reason
    assert result.outputs == {}
    assert result.halted_at_step is None


async def test_plugin_failure_reason_collapses_newlines_and_truncates() -> None:
    """`failure_reason` is interpolated into the synthesis prompt and debug output.

    PR #17 review (Copilot): a stray newline in the exception message
    would break prompt structure; an enormous message would burn the
    context budget. The executor collapses whitespace to a single line
    and bounds the length before storing the reason on the result.
    """
    huge_multiline = 'line one\nline two with\ttabs\nand   spaces\n' + ('x' * 500)
    plugin = _RaisingPlugin(RuntimeError(huge_multiline))
    manifest = _manifest('clock', capabilities=(_cap('now'),))
    executor = _make_executor((manifest, plugin))

    plan = TaskPlan(steps=(PlanStep(step=1, plugin='clock', capability='now', inputs={}, gate='none', outputs_as=None),))
    result = await executor.execute(plan)

    assert result.failure_reason is not None
    assert '\n' not in result.failure_reason
    assert '\t' not in result.failure_reason
    # Exception type surfaces so the model can distinguish e.g. ValueError
    # vs TimeoutError without parsing the message body.
    assert 'RuntimeError' in result.failure_reason
    # Bounded length so a malicious / runaway exception can't blow up the prompt.
    assert len(result.failure_reason) < 500


async def test_plugin_failure_stops_subsequent_steps_with_partial_outputs() -> None:
    """A mid-plan failure stops execution; prior steps' outputs are preserved.

    Downstream steps typically reference the failed step's `$alias.field`
    and would cascade-fail anyway. Stopping at the failure point and
    surfacing partial outputs gives the synthesis turn enough context
    to acknowledge what ran versus what didn't.
    """
    good = _RecordingPlugin(responses={'do': {'value': 7}})
    bad = _RaisingPlugin(RuntimeError('boom'))
    executor = _make_executor(
        (_manifest('a', capabilities=(_cap('do'),)), good),
        (_manifest('b', capabilities=(_cap('do'),)), bad),
    )

    plan = TaskPlan(
        steps=(
            PlanStep(step=1, plugin='a', capability='do', inputs={}, gate='none', outputs_as='first'),
            PlanStep(step=2, plugin='b', capability='do', inputs={}, gate='none', outputs_as='second'),
            PlanStep(step=3, plugin='a', capability='do', inputs={}, gate='none', outputs_as='third'),
        )
    )

    result = await executor.execute(plan)

    assert result.failed_at_step == 2
    assert result.outputs == {'first': {'value': 7}}
    # Step 3 never ran — both because it was after the failure point and
    # because the executor halts rather than cascading. Verify via call log.
    assert good.calls == [('do', {})]


# --- FakePluginContext helper (task #381 slice 2, checklist #395) ------------


async def test_fake_plugin_context_records_calls_and_returns_canned_response() -> None:
    """A plugin can be exercised in isolation with the shared fake context —
    no registry or executor required."""

    class _Plugin:
        async def execute(
            self,
            capability: str,
            inputs: dict[str, object],
            context: PluginContext,
        ) -> dict[str, object]:
            del capability
            stored = await context.call('database.insert', {'row': inputs})
            return {'note_id': stored['id']}

    ctx = FakePluginContext(responses={'database.insert': {'id': 99}})
    result = await _Plugin().execute('create', {'body': 'hi'}, ctx)

    assert result == {'note_id': 99}
    assert ctx.calls == [('database.insert', {'row': {'body': 'hi'}})]


async def test_fake_plugin_context_raises_configured_error() -> None:
    """`errors` lets a test drive the plugin's failure path deterministically."""
    ctx = FakePluginContext(errors={'database.insert': RuntimeError('disk full')})
    with pytest.raises(RuntimeError, match='disk full'):
        await ctx.call('database.insert', {'row': {}})
    # The invocation is recorded before the configured error fires.
    assert ctx.calls == [('database.insert', {'row': {}})]


async def test_fake_plugin_context_unstubbed_capability_is_loud() -> None:
    """A missing canned response is a test-wiring bug, not an empty dict —
    it must fail loudly so an unstubbed dependency can't pass silently."""
    ctx = FakePluginContext()
    with pytest.raises(AssertionError, match='no canned response'):
        await ctx.call('database.select', {})


# --- Database namespace boundary (task #381 slice 3, invariant #6) -----------


def test_core_db_constants_agree_with_plugin() -> None:
    """Core's namespaced-plugin constant must equal the plugin's own name.

    They are intentionally NOT a shared import (core has no dependency
    edge on a plugin module); this test is the contract that stops the
    two literals drifting apart.
    """
    assert _NAMESPACED_DB_PLUGIN == _DB_PLUGIN_NAME
    assert _DB_TABLE_KEY == 'table'


def _db_registry(owner_requires: tuple[str, ...] = ('database.insert',)) -> tuple[_RecordingPlugin, PluginRegistry]:
    db = _RecordingPlugin(responses={'insert': {'id': 1}})
    registry = _registry(
        (
            _manifest(
                'database',
                radius=BlastRadius.LOCAL_WRITE,
                capabilities=(_cap('insert', internal=True),),
            ),
            db,
        ),
        (
            _manifest(
                'notes',
                radius=BlastRadius.LOCAL_WRITE,
                capabilities=(_cap('create'),),
                requires=owner_requires,
            ),
            _RecordingPlugin(responses={}),
        ),
    )
    return db, registry


async def test_db_table_is_prefixed_with_caller_namespace() -> None:
    """The database plugin receives `{caller}__{table}` — never the bare name."""
    db = _RecordingPlugin(responses={'insert': {'id': 1}})
    notes = _CallingPlugin(target='database.insert', inner_inputs={'table': 'entries', 'row': {'body': 'hi'}})
    executor = _make_executor(
        (
            _manifest('database', radius=BlastRadius.LOCAL_WRITE, capabilities=(_cap('insert', internal=True),)),
            db,
        ),
        (
            _manifest(
                'notes',
                radius=BlastRadius.LOCAL_WRITE,
                capabilities=(_cap('create'),),
                requires=('database.insert',),
            ),
            notes,
        ),
    )

    plan = TaskPlan(steps=(PlanStep(step=1, plugin='notes', capability='create', inputs={}, gate='none'),))
    result = await executor.execute(plan)

    assert result.failed_at_step is None
    assert db.calls == [('insert', {'table': 'notes__entries', 'row': {'body': 'hi'}})]


async def test_db_table_containing_separator_is_rejected() -> None:
    """A caller-supplied name with the reserved `__` is refused (it would
    let a plugin address another namespace). Surfaces as a step failure."""
    db = _RecordingPlugin(responses={'insert': {'id': 1}})
    notes = _CallingPlugin(target='database.insert', inner_inputs={'table': 'other__secret', 'row': {}})
    executor = _make_executor(
        (
            _manifest('database', radius=BlastRadius.LOCAL_WRITE, capabilities=(_cap('insert', internal=True),)),
            db,
        ),
        (
            _manifest(
                'notes',
                radius=BlastRadius.LOCAL_WRITE,
                capabilities=(_cap('create'),),
                requires=('database.insert',),
            ),
            notes,
        ),
    )

    plan = TaskPlan(steps=(PlanStep(step=1, plugin='notes', capability='create', inputs={}, gate='none'),))
    result = await executor.execute(plan)

    assert result.failed_at_step == 1
    assert result.failure_reason is not None
    assert 'PluginContextError' in result.failure_reason
    assert '__' in result.failure_reason
    # Rejected before the store was ever reached.
    assert db.calls == []


async def test_apply_db_namespace_rewrites_only_db_target() -> None:
    """White-box: the hook prefixes for the db target, no-ops otherwise."""
    _, registry = _db_registry()
    ctx = _PluginContext(owner='notes', registry=registry, step=1)

    db_inputs: dict[str, object] = {'table': 'entries', 'row': {'k': 1}}
    ctx._apply_db_namespace('database', db_inputs)
    assert db_inputs == {'table': 'notes__entries', 'row': {'k': 1}}

    # Non-db target: untouched even if it has a `table` key.
    other: dict[str, object] = {'table': 'entries'}
    ctx._apply_db_namespace('clock', other)
    assert other == {'table': 'entries'}

    # db target but no `table` key: pass through (plugin raises its own
    # missing-input error downstream).
    no_table: dict[str, object] = {'row': {}}
    ctx._apply_db_namespace('database', no_table)
    assert no_table == {'row': {}}


async def test_apply_db_namespace_rejects_non_string_table() -> None:
    _, registry = _db_registry()
    ctx = _PluginContext(owner='notes', registry=registry, step=1)
    with pytest.raises(PluginContextError, match='must be a non-empty string'):
        ctx._apply_db_namespace('database', {'table': 123})


async def test_apply_db_namespace_rejects_owner_with_separator() -> None:
    """Defence-in-depth: `parse_manifest` forbids `__` in plugin names, so
    this is unreachable in a correct registry. Build one directly (bypassing
    parse_manifest) to prove the core guard still refuses to mint an
    ambiguous fully-qualified name if that validation ever regressed."""
    db = _RecordingPlugin(responses={'insert': {}})
    registry = _unchecked_registry(
        (_manifest('database', radius=BlastRadius.LOCAL_WRITE, capabilities=(_cap('insert', internal=True),)), db),
        (_manifest('a__b', capabilities=(_cap('go'),), requires=('database.insert',)), _RecordingPlugin(responses={})),
    )
    ctx = _PluginContext(owner='a__b', registry=registry, step=1)
    with pytest.raises(PluginContextError, match='reserved to core'):
        ctx._apply_db_namespace('database', {'table': 'entries'})
