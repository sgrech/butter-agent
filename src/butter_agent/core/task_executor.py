"""Task executor — atomic plan validation, gate enforcement, variable resolution.

Three invariants live here:

- #3 A plan is validated **atomically** before any step executes. If any step
  references a missing plugin/capability, declares an unknown gate, omits a
  required input, or references a `$alias.field` that no prior step exports,
  the whole plan is rejected and zero plugins are invoked.
- #4 `$alias.field` resolution is **dict lookup** on prior step outputs keyed
  by `outputs_as`. There is no inference, no fuzzy match, no fallback — a
  missing key fails the step with a typed error.
- #5/#7 Gate enforcement happens here, never in a plugin. The core also injects
  a minimum `confirm` gate for any step whose plugin's blast radius is
  `NETWORK`, regardless of what the model declared. Plugins cannot expand
  their own blast radius (#7).

The pause/resume seam for adapters (REPL, Telegram) is the `GateHandler`
protocol. When a step has an effective gate of `confirm` or `human`, the
executor `await`s the handler with the step and accumulated prior outputs; the
handler returns `CONTINUE` or `ABORT`. `ABORT` yields an `ExecutionResult`
with `halted_at_step` / `halt_reason` set; the loop returns it verbatim and
the adapter renders accordingly.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

from butter_agent.core.loop import ExecutionResult, PlanStep, TaskPlan
from butter_agent.core.registry import (
    BlastRadius,
    CapabilityNotFoundError,
    PluginNotFoundError,
    PluginRegistry,
)

# --- Gate types --------------------------------------------------------------


class Gate(StrEnum):
    """Gate types a plan step may declare.

    `NONE` runs the step immediately. `CONFIRM` pauses for a yes/no decision.
    `HUMAN` pauses to present prior results and wait for free-form input.
    """

    NONE = 'none'
    CONFIRM = 'confirm'
    HUMAN = 'human'


class GateDecision(StrEnum):
    """A handler's response when consulted at a gate."""

    CONTINUE = 'continue'
    ABORT = 'abort'


class GateHandler(Protocol):
    """Adapter-side seam invoked when the executor hits a non-`NONE` gate.

    Implementations bridge to whatever interface is active (REPL prompt,
    Telegram message round-trip, test stub). The executor passes the step
    being gated and a read-only view of all prior step outputs; the handler
    decides whether to continue or abort.
    """

    async def on_gate(
        self,
        step: PlanStep,
        effective_gate: Gate,
        prior_outputs: Mapping[str, Mapping[str, object]],
    ) -> GateDecision: ...


# --- Errors ------------------------------------------------------------------


class ExecutorError(Exception):
    """Base class for task executor errors."""


class PlanValidationError(ExecutorError):
    """Raised when atomic plan validation rejects a plan before execution."""


class VariableResolutionError(ExecutorError):
    """Raised when a `$alias.field` reference cannot be resolved at run time.

    Distinct from `PlanValidationError`: validation rejects references to
    aliases that no prior step declares; this fires when the alias was
    declared but the runtime output dict is missing the named field.
    """


# --- Variable references -----------------------------------------------------

# `$alias.field` — alias and field are identifiers (letters/digits/underscore).
_VAR_REF = re.compile(r'^\$([A-Za-z_][A-Za-z0-9_]*)\.([A-Za-z_][A-Za-z0-9_]*)$')


@dataclass(frozen=True, slots=True)
class _VarRef:
    alias: str
    field: str


def _parse_ref(value: object) -> _VarRef | None:
    """Return the parsed reference if `value` is a `$alias.field` string, else None."""
    if not isinstance(value, str):
        return None
    match = _VAR_REF.match(value)
    if match is None:
        return None
    return _VarRef(alias=match.group(1), field=match.group(2))


# --- Executor ---------------------------------------------------------------


class DefaultTaskExecutor:
    """Concrete `TaskExecutor` implementation.

    Constructed once at startup with the frozen `PluginRegistry` and the
    active `GateHandler`. Stateless across `execute()` calls — each plan
    carries its own variable pool.
    """

    def __init__(self, registry: PluginRegistry, gate_handler: GateHandler) -> None:
        self._registry = registry
        self._gate_handler = gate_handler

    async def execute(self, plan: TaskPlan) -> ExecutionResult:
        """Validate atomically, then walk the plan.

        Returns:
            An `ExecutionResult` whose `outputs` map collects every step
            output that declared an `outputs_as` alias. If a gate handler
            aborts, `halted_at_step` and `halt_reason` are populated and no
            further steps run.

        Raises:
            PlanValidationError: If the plan is structurally invalid. Raised
                before any plugin is invoked.
            VariableResolutionError: If a step's resolved inputs reference a
                field missing from the corresponding prior step's output.
        """
        self._validate(plan)

        outputs: dict[str, dict[str, object]] = {}
        for step in plan.steps:
            effective_gate = self._effective_gate(step)

            if effective_gate is not Gate.NONE:
                decision = await self._gate_handler.on_gate(step, effective_gate, outputs)
                if decision is GateDecision.ABORT:
                    return ExecutionResult(
                        plan=plan,
                        outputs=dict(outputs),
                        halted_at_step=step.step,
                        halt_reason=f'gate {effective_gate.value!r} aborted at step {step.step}',
                    )

            resolved_inputs = self._resolve_inputs(step, outputs)
            registered = self._registry.get(step.plugin)
            step_output = await registered.plugin.execute(step.capability, resolved_inputs)

            if step.outputs_as is not None:
                outputs[step.outputs_as] = dict(step_output)

        return ExecutionResult(plan=plan, outputs=dict(outputs))

    # --- Validation (invariant #3) ------------------------------------------

    def _validate(self, plan: TaskPlan) -> None:
        if not plan.steps:
            raise PlanValidationError('plan has no steps')

        declared_aliases: set[str] = set()
        for index, step in enumerate(plan.steps):
            expected_step_num = index + 1
            if step.step != expected_step_num:
                raise PlanValidationError(
                    f'step at position {expected_step_num} declares step={step.step} (expected sequential 1..N)',
                )

            try:
                Gate(step.gate)
            except ValueError as exc:
                valid = ', '.join(g.value for g in Gate)
                raise PlanValidationError(
                    f'step {step.step}: invalid gate {step.gate!r} (expected one of: {valid})',
                ) from exc

            try:
                capability = self._registry.capability(step.plugin, step.capability)
            except PluginNotFoundError as exc:
                raise PlanValidationError(f'step {step.step}: {exc}') from exc
            except CapabilityNotFoundError as exc:
                raise PlanValidationError(f'step {step.step}: {exc}') from exc

            self._validate_step_inputs(step, capability.input_schema, declared_aliases)

            if step.outputs_as is not None:
                if step.outputs_as in declared_aliases:
                    raise PlanValidationError(
                        f'step {step.step}: outputs_as alias {step.outputs_as!r} already declared',
                    )
                declared_aliases.add(step.outputs_as)

    def _validate_step_inputs(
        self,
        step: PlanStep,
        input_schema: dict[str, object],
        declared_aliases: set[str],
    ) -> None:
        for required_key in input_schema:
            if required_key not in step.inputs:
                raise PlanValidationError(
                    f'step {step.step}: missing required input {required_key!r}',
                )

        for key, value in step.inputs.items():
            ref = _parse_ref(value)
            if ref is None:
                continue
            if ref.alias not in declared_aliases:
                raise PlanValidationError(
                    f'step {step.step}: input {key!r} references unknown alias ${ref.alias!r} (no prior step declared this outputs_as)',
                )

    # --- Variable resolution (invariant #4) ---------------------------------

    def _resolve_inputs(
        self,
        step: PlanStep,
        outputs: Mapping[str, Mapping[str, object]],
    ) -> dict[str, object]:
        resolved: dict[str, object] = {}
        for key, value in step.inputs.items():
            ref = _parse_ref(value)
            if ref is None:
                resolved[key] = value
                continue
            try:
                alias_outputs = outputs[ref.alias]
            except KeyError as exc:
                raise VariableResolutionError(
                    f'step {step.step}: input {key!r} references ${ref.alias}.{ref.field} but alias {ref.alias!r} has no recorded output',
                ) from exc
            try:
                resolved[key] = alias_outputs[ref.field]
            except KeyError as exc:
                raise VariableResolutionError(
                    f'step {step.step}: input {key!r} references ${ref.alias}.{ref.field} but field {ref.field!r} is not in output of alias {ref.alias!r}',
                ) from exc
        return resolved

    # --- Gate enforcement (invariants #5, #7) -------------------------------

    def _effective_gate(self, step: PlanStep) -> Gate:
        declared = Gate(step.gate)
        radius = self._registry.get(step.plugin).manifest.blast_radius
        if radius is BlastRadius.NETWORK and declared is Gate.NONE:
            return Gate.CONFIRM
        return declared


# --- Built-in gate handlers --------------------------------------------------


class AlwaysContinueGateHandler:
    """Handler that approves every gate without prompting.

    Useful for tests of the happy path and for non-interactive batch contexts
    where the operator has pre-approved everything. Production adapters
    (REPL, Telegram) supply their own handler.
    """

    async def on_gate(
        self,
        step: PlanStep,
        effective_gate: Gate,
        prior_outputs: Mapping[str, Mapping[str, object]],
    ) -> GateDecision:
        return GateDecision.CONTINUE
