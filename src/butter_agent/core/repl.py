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

import os
import sys
from collections.abc import AsyncIterator, Callable, Mapping
from contextlib import AbstractAsyncContextManager, asynccontextmanager, nullcontext
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Protocol, TextIO

from butter_agent.core.loop import (
    AgentLoop,
    ExecutionResult,
    ModelProtocolError,
    PlanStep,
    TurnResult,
)
from butter_agent.core.task_executor import (
    ExecutorError,
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


class IndicatorControl(Protocol):
    """The pause/resume surface of a running inference indicator.

    Lives in core so that any code that takes over the terminal (gate
    handler today; future progress-bar plugins, interactive editors)
    can suspend the indicator without depending on the concrete
    `InferenceIndicator` implementation in `repl_prompt_toolkit`.
    """

    def pause(self) -> None: ...

    def resume(self) -> None: ...


_active_indicator: ContextVar[IndicatorControl | None] = ContextVar('butter_agent_active_indicator', default=None)


def register_active_indicator(indicator: IndicatorControl) -> object:
    """Mark `indicator` as the active one for `suspend_indicator()`.

    Called by `InferenceIndicator.__aenter__`. Returns a token to pass
    to `unregister_active_indicator` for the matching reset.
    """
    return _active_indicator.set(indicator)


def unregister_active_indicator(token: object) -> None:
    """Counterpart to `register_active_indicator`. Restores the prior value."""
    _active_indicator.reset(token)  # type: ignore[arg-type]


@asynccontextmanager
async def suspend_indicator() -> AsyncIterator[None]:
    """Pause the currently active inference indicator for the body.

    No-op when no indicator is registered (tests, non-TTY runs). The
    gate handler wraps its prompt in this so the spinner doesn't keep
    drawing "thinking…" frames while the REPL is actually blocked on
    a y/N answer.
    """
    indicator = _active_indicator.get()
    if indicator is not None:
        indicator.pause()
    try:
        yield
    finally:
        if indicator is not None:
            indicator.resume()


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


# --- Slash command dispatch --------------------------------------------------


@dataclass(frozen=True, slots=True)
class CommandResult:
    """Outcome of running a slash command.

    `exit=True` signals the REPL should stop accepting input after the
    command completes — used by `/quit` and reserved for any future
    command that needs to halt the loop without raising.
    """

    exit: bool = False


class Command(Protocol):
    """A slash command available from the REPL.

    Commands are looked up by `name` (no leading slash). `description` is
    surfaced by `/help`. `run` receives the trailing args string (already
    stripped of the command token) plus IO seams so commands can prompt
    interactively when needed.
    """

    name: str
    description: str

    async def run(self, args: str, io_in: InputSource, output: Output) -> CommandResult: ...


class CommandRegistry:
    """Lookup table of slash commands, frozen at construction time.

    The dispatcher itself lives in `Repl.run`; this class is the
    immutable lookup table the loop consults. Duplicate names raise at
    construction so the registry never silently masks one binding with
    another.
    """

    def __init__(self, commands: tuple[Command, ...]) -> None:
        entries: dict[str, Command] = {}
        for command in commands:
            if command.name in entries:
                raise ValueError(f'duplicate slash command: {command.name!r}')
            entries[command.name] = command
        self._entries = entries

    def get(self, name: str) -> Command | None:
        """Look up a command by name (no leading slash). Returns `None` if absent."""
        return self._entries.get(name)

    def all(self) -> tuple[Command, ...]:
        """Return all registered commands in registration order."""
        return tuple(self._entries.values())


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
        # Pause the inference spinner for the duration of the gate
        # interaction. Without this, the stderr spinner keeps drawing
        # "thinking…" frames while the REPL is actually blocked on the
        # operator's y/N answer — both misleading and visually noisy
        # against the gate prompt on stdout.
        async with suspend_indicator():
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
    (bad model output) and `ExecutorError` (plan validation or variable
    resolution failures) are rendered as single-line diagnostics and the
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
        commands: CommandRegistry | None = None,
        indicator_factory: Callable[[], AbstractAsyncContextManager[object]] | None = None,
    ) -> None:
        self._loop = loop
        self._input = input_source
        self._output = output
        self._prompt = prompt
        self._banner = banner
        self._commands = commands if commands is not None else CommandRegistry(())
        # `indicator_factory` is consulted once per turn to wrap the
        # awaited `run_turn` call. Default is `nullcontext` — no-op for
        # tests and non-TTY runs. A TTY-aware caller wires in
        # `InferenceIndicator` from `repl_prompt_toolkit`.
        self._indicator_factory: Callable[[], AbstractAsyncContextManager[object]] = indicator_factory if indicator_factory is not None else _no_indicator

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
            if user_input.startswith('/'):
                if await self._dispatch_command(user_input):
                    return
                continue
            try:
                async with self._indicator_factory():
                    result = await self._loop.run_turn(user_input)
            except ModelProtocolError as exc:
                self._output.write(f'[error] model adapter: {exc}\n')
                continue
            except ExecutorError as exc:
                self._output.write(f'[error] plan rejected: {exc}\n')
                continue
            self._render(result)

    async def _dispatch_command(self, line: str) -> bool:
        """Run a slash command. Returns `True` when the REPL should exit."""
        # Split off the leading '/' and the first whitespace-bounded token —
        # `str.split(maxsplit=1)` handles tabs and other whitespace, not just
        # literal spaces.
        head, *rest = line[1:].split(maxsplit=1)
        args = rest[0] if rest else ''
        command = self._commands.get(head)
        if command is None:
            self._output.write(f'[error] unknown command: /{head}\n')
            return False
        result = await command.run(args, self._input, self._output)
        return result.exit

    def _render(self, result: TurnResult) -> None:
        if result.reply is not None:
            if _debug_enabled():
                self._output.write('[debug] model emitted: reply\n')
            self._output.write(f'{result.reply.text}\n')
            return
        # By construction of TurnResult, exactly one of reply / executed_plan is set.
        assert result.executed_plan is not None
        self._render_execution(result.executed_plan)

    def _render_execution(self, execution: ExecutionResult) -> None:
        if execution.halted_at_step is not None:
            self._output.write(f'[halted at step {execution.halted_at_step}] {execution.halt_reason}\n')
            return
        if _debug_enabled():
            # Make plan execution visible on the happy path too — without
            # this the operator cannot tell whether a precise-looking reply
            # came from a real plugin invocation or model confabulation.
            self._output.write(f'[debug] plan executed: {len(execution.plan.steps)} step(s)\n')
            for step in execution.plan.steps:
                marker = ''
                if execution.failed_at_step == step.step:
                    marker = ' [FAILED]'
                elif execution.failed_at_step is not None and step.step > execution.failed_at_step:
                    marker = ' [skipped]'
                self._output.write(f'[debug]   step {step.step}: {step.plugin}.{step.capability}{marker} inputs={step.inputs!r}\n')
            if execution.failure_reason is not None:
                self._output.write(f'[debug]   failure: {execution.failure_reason}\n')
            for alias, fields in execution.outputs.items():
                self._output.write(f'[debug]   ${alias} = {dict(fields)!r}\n')
        if execution.synthesis_reply is not None:
            # Successful execution went through the synthesis pass. The
            # synthesized reply is the assistant surface; the executed plan
            # is implementation detail that's already in conversation history.
            self._output.write(f'{execution.synthesis_reply.text}\n')
            return
        # Fallback for executions that bypassed synthesis (e.g. test wiring
        # without a synthesis step). Render the raw outputs so the operator
        # still sees what ran.
        self._output.write(f'[plan executed: {len(execution.plan.steps)} step(s)]\n')
        for alias, fields in execution.outputs.items():
            self._output.write(f'  ${alias}: {dict(fields)!r}\n')


def _no_indicator() -> AbstractAsyncContextManager[object]:
    # Default factory for `Repl(indicator_factory=...)`. `nullcontext()`
    # returns an async-compatible context manager that does nothing,
    # so the indicator seam adds zero overhead when unused (tests, non-
    # TTY runs, custom adapters that own their own progress UI).
    return nullcontext()


def _debug_enabled() -> bool:
    # BUTTER_DEBUG=1/true/yes turns on per-turn visibility into what the
    # model emitted and which plugin call(s) actually ran. Off by default
    # because the markers pollute scripted use of the REPL.
    return os.environ.get('BUTTER_DEBUG', '').lower() in {'1', 'true', 'yes'}
