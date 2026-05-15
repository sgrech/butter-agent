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

import logging
import re
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import Protocol

from butter_agent.core.loop import ExecutionResult, PlanStep, TaskPlan
from butter_agent.core.registry import (
    BlastRadius,
    CapabilityNotFoundError,
    PluginNotFoundError,
    PluginRegistry,
)

logger = logging.getLogger(__name__)


# --- PluginContext ----------------------------------------------------------


class PluginContextError(Exception):
    """Raised when a plugin misuses its `PluginContext`.

    Deliberately **not** an `ExecutorError`. An `ExecutorError` is a core
    contract fault (bad plan validation, missing alias) and must surface to
    the caller. A bad `ctx.call` — naming a capability the plugin never
    declared in its manifest `requires` — is *plugin-author* misbehaviour,
    indistinguishable from any other exception third-party plugin code may
    raise (invariant #6). It therefore flows through the executor's broad
    plugin-failure path and is recorded as a `failure_reason`, so the loop
    can still synthesise a reply instead of tearing the REPL down (the
    2026-05-14 user correction). The exception type is preserved verbatim in
    `failure_reason`, so a plugin author still sees `PluginContextError` in
    debug output and knows to fix their manifest — it is a programming bug,
    not a runtime condition a plugin should `except`.
    """


#: Hard ceiling on `PluginContext.call` nesting. The `requires` graph is a
#: DAG (cycles rejected at registry build, invariant #1), so legitimate
#: chains are short. This is defence-in-depth: a regressed build-time check
#: or a pathologically deep (still acyclic) chain would otherwise recurse
#: until Python's own recursion limit and surface as an opaque deep
#: `RecursionError` traceback with no plugin context. Failing fast here
#: yields a `PluginContextError` that names the owner, capability, and the
#: ceiling — actionable instead of cryptic.
_MAX_CTX_DEPTH: int = 32


#: Name of the built-in store whose `table` input core namespaces by caller
#: identity before dispatch (task #381 slice 3, invariant #6 — hard
#: isolation). Kept as a core constant rather than importing the plugin so
#: core has no dependency edge on a plugin module. It MUST equal
#: `butter_agent.plugins.database.PLUGIN_NAME`; a test asserts the two agree
#: so the contract can't silently drift.
_NAMESPACED_DB_PLUGIN: str = 'database'

#: The single input key every `database.*` capability uses to name its
#: table. Core rewrites exactly this key to `{caller}__{table}`; the plugin
#: never sees the un-prefixed name or the caller (see database plugin
#: module docstring). One key for one blessed plugin keeps this boundary
#: from coupling core to the plugin's per-capability schema.
_DB_TABLE_KEY: str = 'table'


class _PluginContext:
    """The real per-invocation `PluginContext` (task #381 slice 2).

    Closes over the **owner** — the plugin currently executing — sourced
    from the frozen registry, never from call arguments, so a plugin cannot
    forge identity by handing a different context to another plugin
    (invariant #6). `call` enforces that the requested capability is in the
    owner's manifest `requires`, then dispatches to the target plugin with a
    *fresh* `_PluginContext` whose owner is the target. Dispatch is therefore
    recursive: an `a -> b -> c` chain composes, and each hop is independently
    restricted to its own `requires` set. Cycles are impossible — the
    registry builder rejects `requires` cycles at startup (invariant #1), so
    this recursion always terminates. A `_MAX_CTX_DEPTH` ceiling is enforced
    anyway as defence-in-depth: a regressed build-time check or an
    excessively deep (still acyclic) chain would otherwise recurse to
    Python's recursion limit and surface as an opaque `RecursionError`. The
    ceiling fails fast with a `PluginContextError` naming the owner and
    capability — caught and recorded on the same plugin-failure path as any
    other `Exception` (so the loop still synthesises; the 2026-05-14
    correction holds), just with an actionable message instead of a cryptic
    deep traceback.

    Namespacing of an internal store (e.g. the `database` plugin prefixing
    tables by caller) is deliberately *not* done here — that is task #381
    slice 3. The seam it will use is `self._owner` at the dispatch point,
    which is exactly the calling plugin's identity.
    """

    __slots__ = ('_depth', '_owner', '_registry', '_step')

    def __init__(
        self,
        *,
        owner: str,
        registry: PluginRegistry,
        step: int,
        depth: int = 0,
    ) -> None:
        self._owner = owner
        self._registry = registry
        self._step = step
        self._depth = depth

    @property
    def config(self) -> Mapping[str, object]:
        """This plugin's own operator-supplied config, read-only.

        Sourced from the frozen registry keyed by `self._owner` — the
        same closed-over identity `call` uses, never a call argument
        (invariant #6). Wrapped in a `MappingProxyType` so third-party
        plugin code cannot mutate the registry's snapshot. For a nested
        `call`, the child context's owner is the *target* plugin, so the
        target transparently sees its own config and never the caller's.
        """
        return MappingProxyType(dict(self._registry.get(self._owner).config))

    async def call(self, capability: str, inputs: dict[str, object]) -> dict[str, object]:
        """Invoke an internal capability declared in the owner's `requires`.

        Args:
            capability: Fully-qualified `plugin.capability` reference,
                identical to how it appears in the owner's manifest
                `requires`.
            inputs: Inputs forwarded verbatim to the target capability.

        Returns:
            The target capability's output dict.

        Raises:
            PluginContextError: If the nesting depth ceiling is exceeded, if
                `capability` is not in the owner's declared `requires`, or
                (defensively) it does not resolve to an `internal=True`
                capability. All are plugin-author / manifest bugs the
                registry builder normally catches at startup, but a clear
                error here beats an obscure one if an invariant regresses.
        """
        if self._depth >= _MAX_CTX_DEPTH:
            raise PluginContextError(
                f'plugin {self._owner!r}: PluginContext.call nesting exceeded {_MAX_CTX_DEPTH} (calling {capability!r}) — a requires cycle that escaped registry-build validation, or an excessively deep requires chain',
            )

        owner_manifest = self._registry.get(self._owner).manifest
        if capability not in owner_manifest.requires:
            raise PluginContextError(
                f'plugin {self._owner!r} called {capability!r} via PluginContext but did not declare it in manifest requires',
            )

        # `capability` passed manifest validation as `plugin.capability`
        # and is a member of `requires`, so the split is total.
        target_plugin, target_capability = capability.split('.', 1)
        target = self._registry.get(target_plugin)
        cap = target.manifest.capability(target_capability)
        if not cap.internal:
            raise PluginContextError(
                f'plugin {self._owner!r} called {capability!r} but it is not internal — only internal=true capabilities are callable plugin-to-plugin (registry build should have rejected this)',
            )

        # Nested call: log indented one level below the owner so a reader
        # sees `database.insert` sitting under its parent plan step.
        logger.debug(
            '%sstep %d: %s -> %s',
            '  ' * (self._depth + 1),
            self._step,
            self._owner,
            capability,
        )

        child = _PluginContext(
            owner=target_plugin,
            registry=self._registry,
            step=self._step,
            depth=self._depth + 1,
        )
        # Copy inputs across the plugin boundary: the target is third-party
        # code (invariant #6) and must not be able to mutate the caller's
        # dict. Mirrors the executor's defensive copies elsewhere and the
        # FakePluginContext test helper, so fake and real behave alike.
        dispatch_inputs = dict(inputs)
        self._apply_db_namespace(target_plugin, dispatch_inputs)
        return await target.plugin.execute(target_capability, dispatch_inputs, child)

    def _apply_db_namespace(self, target_plugin: str, inputs: dict[str, object]) -> None:
        """Rewrite the database `table` input to the caller's namespace.

        Invariant #6 in its strongest form: a plugin can only ever touch
        `{its-own-name}__{table}`. The caller's namespace IS `self._owner`
        — closed over by core, never an argument the caller can set,
        override, or lie about. The `__` separator is reserved, so a
        caller-supplied name containing it is rejected (otherwise
        `notes` could pass `other__secret` and escape its namespace).

        Mutates `inputs` in place (already a private per-dispatch copy).
        No-op for any target other than the built-in store, and for a
        call that omits `table` (the plugin then raises its own
        missing-input error).
        """
        if target_plugin != _NAMESPACED_DB_PLUGIN or _DB_TABLE_KEY not in inputs:
            return
        if '__' in self._owner:
            # parse_manifest forbids `__` in plugin names, so this is
            # unreachable in a correctly built registry. Assert it anyway:
            # if that validation ever regressed, an owner like `a__b` would
            # silently collide with another namespace. Defence-in-depth,
            # same stance as the internal-capability and depth guards.
            raise PluginContextError(
                f'plugin name {self._owner!r} contains "__" — the database namespace separator is reserved to core; registry validation should have rejected this',
            )
        table = inputs[_DB_TABLE_KEY]
        if not isinstance(table, str) or not table:
            raise PluginContextError(
                f'plugin {self._owner!r}: database call {_DB_TABLE_KEY!r} must be a non-empty string, got {table!r}',
            )
        if '__' in table:
            raise PluginContextError(
                f'plugin {self._owner!r}: database table {table!r} may not contain "__" — the namespace separator is reserved to core',
            )
        inputs[_DB_TABLE_KEY] = f'{self._owner}__{table}'


# --- Gate types --------------------------------------------------------------


class Gate(StrEnum):
    """Gate types a plan step may declare.

    `NONE` runs the step immediately. `CONFIRM` pauses for an explicit yes/no
    approval before running the step. `HUMAN` pauses to present prior step
    outputs to the operator and waits for an approve-or-abort decision (it
    does not yet round-trip free-form input back into the plan — that would
    require extending the gate API).
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


_FAILURE_REASON_MAX_CHARS = 300


def _format_failure_reason(step: PlanStep, exc: BaseException) -> str:
    """Build a single-line, length-bounded failure_reason string.

    Plugin exceptions are user-controlled-ish (third-party code,
    invariant #6) and their messages can carry newlines or be huge.
    `failure_reason` is interpolated into the synthesis prompt and the
    debug output — a stray newline breaks the prompt structure, an
    enormous message wastes context. Normalise here so downstream
    renderers can interpolate without escaping. Plugin and capability
    names are already restricted to a safe charset at manifest-parse
    time (see `core/registry.py`), so quoting them is enough.
    """
    raw = str(exc)
    one_line = ' '.join(raw.split()) if raw else ''
    if len(one_line) > _FAILURE_REASON_MAX_CHARS:
        one_line = one_line[: _FAILURE_REASON_MAX_CHARS - 1] + '…'
    return f'plugin {step.plugin!r} capability {step.capability!r} raised {type(exc).__name__}: {one_line}'


def _freeze_outputs(outputs: Mapping[str, Mapping[str, object]]) -> Mapping[str, Mapping[str, object]]:
    """Return a read-only snapshot of the executor's output pool.

    Wraps each inner dict in a MappingProxyType so a misbehaving gate handler
    cannot mutate prior step outputs and corrupt variable resolution for
    later steps. The outer mapping is also a proxy.
    """
    return MappingProxyType({alias: MappingProxyType(dict(value)) for alias, value in outputs.items()})


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
                snapshot = _freeze_outputs(outputs)
                decision = await self._gate_handler.on_gate(step, effective_gate, snapshot)
                if decision is GateDecision.ABORT:
                    return ExecutionResult(
                        plan=plan,
                        outputs=dict(outputs),
                        halted_at_step=step.step,
                        halt_reason=f'gate {effective_gate.value!r} aborted at step {step.step}',
                    )

            resolved_inputs = self._resolve_inputs(step, outputs)
            registered = self._registry.get(step.plugin)
            logger.debug('step %d: executing %s.%s', step.step, step.plugin, step.capability)
            context = _PluginContext(
                owner=step.plugin,
                registry=self._registry,
                step=step.step,
            )
            try:
                step_output = await registered.plugin.execute(step.capability, resolved_inputs, context)
            except ExecutorError:
                # Don't swallow our own contract errors — those are
                # programming faults (bad plan validation, missing alias)
                # and must surface to the caller, not the model.
                raise
            except Exception as exc:
                # Plugins are third-party code (invariant #6) and may raise
                # for any reason. Record the failure as a value on the
                # ExecutionResult so the loop can still run synthesis and
                # let the model acknowledge it to the user. Stop here:
                # downstream steps usually reference this step's outputs
                # via $alias.field and would cascade-fail anyway.
                return ExecutionResult(
                    plan=plan,
                    outputs=dict(outputs),
                    failed_at_step=step.step,
                    failure_reason=_format_failure_reason(step, exc),
                )

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

            if capability.internal:
                # Internal capabilities are infrastructure surface (e.g.
                # database.insert) and are only legal targets of
                # PluginContext.call. The model must never place them in a
                # plan — gates and variable-pool semantics don't apply to
                # internal calls, so accepting them would corrupt both.
                raise PlanValidationError(
                    f'step {step.step}: capability {step.plugin}.{step.capability} is internal and cannot appear in a plan (callable only plugin-to-plugin via PluginContext)',
                )

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
                    f'step {step.step}: input {key!r} references unknown alias ${ref.alias} (no prior step declared this outputs_as)',
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
                    f'step {step.step}: input {key!r} references ${ref.alias}.{ref.field} but alias ${ref.alias} has no recorded output',
                ) from exc
            try:
                resolved[key] = alias_outputs[ref.field]
            except KeyError as exc:
                raise VariableResolutionError(
                    f'step {step.step}: input {key!r} references ${ref.alias}.{ref.field} but field {ref.field!r} is not in output of alias ${ref.alias}',
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
