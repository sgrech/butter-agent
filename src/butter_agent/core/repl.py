"""REPL — first-class interactive interface adapter.

The REPL is the reference adapter that drives `AgentLoop` end-to-end. It
owns three things:

- The read-print loop: prompt → `AgentLoop.run_turn` → render `TurnResult`.
- `ReplGateHandler`: translates non-`NONE` plan-step gates (`confirm`,
  `human`) into a terminal yes/no prompt, surfacing prior step outputs on
  `human` gates so the operator can make an informed call.
- IO seams (`InputSource`, `Output` Protocols) so the same loop is
  testable without a TTY and so future adapters (Telegram, web) can share
  the contract — only the IO surface changes.

What the REPL does NOT do:

- Validate plans, resolve `$variables`, or enforce blast-radius — those
  belong to `core/task_executor.py` (invariants #3-#7).
- Choose a `ModelClient` — the loop is constructed with one and the REPL
  treats it as opaque. The Ollama adapter is a separate module.
"""

from __future__ import annotations

import sys
from collections.abc import Mapping
from typing import Protocol, TextIO

from butter_agent.core.loop import (
    AgentLoop,
    ExecutionResult,
    ModelProtocolError,
    PlanStep,
    TurnResult,
)
from butter_agent.core.task_executor import (
    Gate,
    GateDecision,
)

# --- IO seams ----------------------------------------------------------------


class InputSource(Protocol):
    """Source of user input lines.

    Implementations raise `EOFError` to signal end-of-input (e.g. Ctrl-D on
    stdin, closed socket on a remote adapter). The REPL treats `EOFError`
    as a graceful shutdown signal — not an error.
    """

    async def read_line(self, prompt: str) -> str: ...


class Output(Protocol):
    """Destination for rendered text.

    Implementations are responsible for any flushing required by their
    underlying stream. The REPL writes complete, newline-terminated
    fragments and does not buffer.
    """

    def write(self, text: str) -> None: ...


class StdioInputSource:
    """`InputSource` backed by a text stream (default `sys.stdin`).

    The blocking `readline()` call is acceptable here: the REPL is a
    single-user interactive loop and nothing else runs on the event loop
    while the prompt is open. Adapters that need true concurrency (e.g.
    Telegram) supply their own `InputSource` rather than reusing this.
    """

    def __init__(self, stream: TextIO | None = None, *, prompt_stream: TextIO | None = None) -> None:
        self._stream = stream if stream is not None else sys.stdin
        self._prompt_stream = prompt_stream if prompt_stream is not None else sys.stdout

    async def read_line(self, prompt: str) -> str:
        self._prompt_stream.write(prompt)
        self._prompt_stream.flush()
        line = self._stream.readline()
        if not line:
            raise EOFError
        return line.rstrip('\n')


class StdioOutput:
    """`Output` writing to a text stream (default `sys.stdout`) and flushing each write."""

    def __init__(self, stream: TextIO | None = None) -> None:
        self._stream = stream if stream is not None else sys.stdout

    def write(self, text: str) -> None:
        self._stream.write(text)
        self._stream.flush()


# --- Gate handler ------------------------------------------------------------


class ReplGateHandler:
    """`GateHandler` that prompts the operator via the REPL's IO seams.

    `confirm` gates print the step being gated and ask for approval.
    `human` gates additionally print the accumulated prior-step outputs so
    the operator can review what the plan will feed into the gated step
    before approving. Any answer other than `y`/`yes` (case-insensitive)
    is treated as an abort — including blank input and EOF. This errs
    toward halting rather than executing, which matches the gate's intent.
    """

    def __init__(self, input_source: InputSource, output: Output) -> None:
        self._input = input_source
        self._output = output

    async def on_gate(
        self,
        step: PlanStep,
        effective_gate: Gate,
        prior_outputs: Mapping[str, Mapping[str, object]],
    ) -> GateDecision:
        self._output.write(
            f'\n[gate:{effective_gate.value}] step {step.step}: {step.plugin}.{step.capability}\n',
        )
        if effective_gate is Gate.HUMAN and prior_outputs:
            self._output.write('  prior outputs:\n')
            for alias, fields in prior_outputs.items():
                self._output.write(f'    ${alias}: {dict(fields)!r}\n')
        try:
            answer = await self._input.read_line('  approve? [y/N] ')
        except EOFError:
            self._output.write('\n')
            return GateDecision.ABORT
        if answer.strip().lower() in {'y', 'yes'}:
            return GateDecision.CONTINUE
        return GateDecision.ABORT


# --- The REPL ----------------------------------------------------------------


class Repl:
    """Interactive read-loop driving an `AgentLoop`.

    A single `Repl` instance wraps one configured `AgentLoop` (which
    already holds the context manager, model client, and task executor
    per invariant #1). `run()` blocks until the input source signals EOF.

    Blank input is ignored (no turn dispatched). `ModelProtocolError`
    raised by the loop is rendered as a single-line diagnostic and the
    REPL keeps running — the loop itself does not retry or guess, so the
    user can correct and retry.
    """

    def __init__(
        self,
        loop: AgentLoop,
        input_source: InputSource,
        output: Output,
        *,
        prompt: str = '> ',
        banner: str = 'butter-agent. Ready.\n',
    ) -> None:
        self._loop = loop
        self._input = input_source
        self._output = output
        self._prompt = prompt
        self._banner = banner

    async def run(self) -> None:
        """Drive the read-print loop until EOF on input."""
        self._output.write(self._banner)
        while True:
            try:
                line = await self._input.read_line(self._prompt)
            except EOFError:
                self._output.write('\n')
                return
            user_input = line.strip()
            if not user_input:
                continue
            try:
                result = await self._loop.run_turn(user_input)
            except ModelProtocolError as exc:
                self._output.write(f'[error] model adapter: {exc}\n')
                continue
            self._render(result)

    def _render(self, result: TurnResult) -> None:
        if result.reply is not None:
            self._output.write(f'{result.reply.text}\n')
            return
        # By construction of TurnResult, exactly one of reply / executed_plan is set.
        assert result.executed_plan is not None
        self._render_execution(result.executed_plan)

    def _render_execution(self, execution: ExecutionResult) -> None:
        if execution.halted_at_step is not None:
            self._output.write(f'[halted at step {execution.halted_at_step}] {execution.halt_reason}\n')
            return
        self._output.write(f'[plan executed: {len(execution.plan.steps)} step(s)]\n')
        for alias, fields in execution.outputs.items():
            self._output.write(f'  ${alias}: {dict(fields)!r}\n')
