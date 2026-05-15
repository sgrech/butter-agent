"""Tests for the REPL adapter.

The REPL is the reference adapter driving `AgentLoop` end-to-end. These
tests cover three concerns:

- The read-print loop: prompt cadence, blank-input handling, rendering of
  `ModelReply` and `ExecutionResult` (success and halted), graceful EOF.
- `ReplGateHandler`: yes/no parsing, abort-on-anything-else, `human` gate
  surfacing prior outputs, EOF mid-gate aborts cleanly.
- IO seam defaults (`StdioInputSource`, `StdioOutput`): correct stream
  plumbing, EOF on closed stdin, prompt flushing.
"""

from __future__ import annotations

import io
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass, field

import pytest

from butter_agent.core.loop import (
    AgentLoop,
    ContextManager,
    DiscoverySelection,
    ExecutionResult,
    ModelContext,
    ModelOutput,
    ModelProtocolError,
    ModelReply,
    PlanStep,
    TaskPlan,
    Turn,
)
from butter_agent.core.repl import (
    Repl,
    ReplGateHandler,
    StdioInputSource,
    StdioOutput,
)
from butter_agent.core.task_executor import Gate, GateDecision, PlanValidationError

# --- Test plumbing -----------------------------------------------------------


@dataclass
class _ScriptedInput:
    """Returns queued lines, then raises EOFError. Captures prompts seen."""

    lines: deque[str]
    prompts: list[str] = field(default_factory=list)

    async def read_line(self, prompt: str) -> str:
        self.prompts.append(prompt)
        if not self.lines:
            raise EOFError
        return self.lines.popleft()


@dataclass
class _CapturingOutput:
    chunks: list[str] = field(default_factory=list)

    def write(self, text: str) -> None:
        self.chunks.append(text)

    @property
    def text(self) -> str:
        return ''.join(self.chunks)


@dataclass
class _StubContextManager:
    discovery_active = False

    async def assemble(
        self,
        turn: Turn,
        execution: ExecutionResult | None = None,
        selection: DiscoverySelection | None = None,
    ) -> ModelContext:
        del selection
        payload: dict[str, object] = {}
        if execution is not None:
            payload['execution'] = execution
        return ModelContext(turn=turn, payload=payload)


@dataclass
class _ScriptedModel:
    """Yields the next queued output per call. Raises if exhausted."""

    outputs: deque[ModelOutput | Exception]

    async def generate(self, context: ModelContext) -> ModelOutput:
        item = self.outputs.popleft()
        if isinstance(item, Exception):
            raise item
        return item


@dataclass
class _PassthroughExecutor:
    """Records plans seen and returns the queued ExecutionResult per call."""

    results: deque[ExecutionResult]
    seen: list[TaskPlan] = field(default_factory=list)

    async def execute(self, plan: TaskPlan) -> ExecutionResult:
        self.seen.append(plan)
        return self.results.popleft()


def _wire(
    *,
    model_outputs: list[ModelOutput | Exception],
    executor_results: list[ExecutionResult] | None = None,
) -> tuple[AgentLoop, _ScriptedModel, _PassthroughExecutor]:
    cm: ContextManager = _StubContextManager()
    model = _ScriptedModel(outputs=deque(model_outputs))
    executor = _PassthroughExecutor(results=deque(executor_results or []))
    loop = AgentLoop(
        context_manager=cm,
        model=model,
        executor=executor,
    )
    return loop, model, executor


# --- Repl: read-print loop ---------------------------------------------------


async def test_repl_renders_direct_reply_and_loops() -> None:
    loop, _, _ = _wire(model_outputs=[ModelReply(text='hello')])
    inp = _ScriptedInput(lines=deque(['hi']))
    out = _CapturingOutput()
    await Repl(loop, inp, out, banner='B\n').run()

    assert out.text.startswith('B\n')
    assert 'hello\n' in out.text
    # Prompt issued twice: once for input, once for EOF after we consumed it.
    assert inp.prompts == ['> ', '> ']


async def test_repl_ignores_blank_input() -> None:
    loop, model, _ = _wire(model_outputs=[ModelReply(text='ok')])
    inp = _ScriptedInput(lines=deque(['', '   ', 'ping']))
    out = _CapturingOutput()
    await Repl(loop, inp, out, banner='').run()

    # Only the non-blank line should have dispatched a turn.
    assert len(model.outputs) == 0  # consumed once
    assert 'ok\n' in out.text


async def test_repl_strips_whitespace_before_dispatch() -> None:
    captured: list[str] = []

    @dataclass
    class _RecordingCM:
        discovery_active = False

        async def assemble(
            self,
            turn: Turn,
            execution: ExecutionResult | None = None,
            selection: DiscoverySelection | None = None,
        ) -> ModelContext:
            del execution, selection
            captured.append(turn.user_input)
            return ModelContext(turn=turn, payload={})

    loop = AgentLoop(
        context_manager=_RecordingCM(),
        model=_ScriptedModel(outputs=deque([ModelReply(text='x')])),
        executor=_PassthroughExecutor(results=deque()),
    )
    inp = _ScriptedInput(lines=deque(['  hello world  ']))
    await Repl(loop, inp, _CapturingOutput(), banner='').run()
    assert captured == ['hello world']


async def test_repl_renders_synthesized_reply_after_plan() -> None:
    # Successful plan execution + synthesis pass: the REPL renders the
    # natural-language reply, NOT the raw output dicts. This is the user-
    # facing behaviour that makes the agent feel conversational rather
    # than like a tool dispatcher.
    step = PlanStep(step=1, plugin='notes', capability='create', inputs={}, gate='none', outputs_as='n')
    plan = TaskPlan(steps=(step,))
    execution = ExecutionResult(plan=plan, outputs={'n': {'id': 7}})
    loop, _, _ = _wire(
        model_outputs=[plan, ModelReply(text='Note created with id 7.')],
        executor_results=[execution],
    )

    inp = _ScriptedInput(lines=deque(['take a note']))
    out = _CapturingOutput()
    await Repl(loop, inp, out, banner='').run()

    assert 'Note created with id 7.\n' in out.text
    # Raw-output dump must NOT appear when synthesis succeeded.
    assert '[plan executed:' not in out.text
    assert '$n:' not in out.text


async def test_repl_falls_back_to_raw_outputs_without_synthesis() -> None:
    # Direct render path for callers that hand the REPL a TurnResult whose
    # ExecutionResult has no synthesis_reply (e.g. legacy adapters or tests).
    # Verifies the fallback is still functional after the synthesis-aware
    # primary path was added.
    from butter_agent.core.loop import TurnResult

    step = PlanStep(step=1, plugin='notes', capability='create', inputs={}, gate='none', outputs_as='n')
    plan = TaskPlan(steps=(step,))
    execution = ExecutionResult(plan=plan, outputs={'n': {'id': 7}}, synthesis_reply=None)
    turn = Turn(turn_id='t', user_input='note', timestamp=0.0)

    out = _CapturingOutput()
    repl = Repl(_wire(model_outputs=[])[0], _ScriptedInput(lines=deque()), out, banner='')
    repl._render(TurnResult(turn=turn, executed_plan=execution))

    assert '[plan executed: 1 step(s)]' in out.text
    assert "$n: {'id': 7}" in out.text


async def test_repl_renders_halted_plan() -> None:
    step = PlanStep(step=1, plugin='notes', capability='create', inputs={}, gate='confirm')
    plan = TaskPlan(steps=(step,))
    execution = ExecutionResult(
        plan=plan,
        outputs={},
        halted_at_step=1,
        halt_reason="gate 'confirm' aborted at step 1",
    )
    loop, _, _ = _wire(model_outputs=[plan], executor_results=[execution])

    inp = _ScriptedInput(lines=deque(['note']))
    out = _CapturingOutput()
    await Repl(loop, inp, out, banner='').run()

    assert '[halted at step 1]' in out.text
    assert "gate 'confirm' aborted at step 1" in out.text


async def test_repl_recovers_from_model_protocol_error() -> None:
    loop, _, _ = _wire(
        model_outputs=[ModelProtocolError('bad json'), ModelReply(text='ok')],
    )
    inp = _ScriptedInput(lines=deque(['first', 'second']))
    out = _CapturingOutput()
    await Repl(loop, inp, out, banner='').run()

    assert '[error] model adapter: bad json' in out.text
    assert 'ok\n' in out.text


async def test_repl_recovers_from_executor_error() -> None:
    # An executor that raises (e.g. PlanValidationError because the model
    # emitted a plan missing a required input) must not crash the REPL.
    @dataclass
    class _RaisingExecutor:
        async def execute(self, plan: TaskPlan) -> ExecutionResult:
            raise PlanValidationError("step 1: missing required input 'tz'")

    step = PlanStep(step=1, plugin='clock', capability='now', inputs={}, gate='none')
    plan = TaskPlan(steps=(step,))
    loop = AgentLoop(
        context_manager=_StubContextManager(),
        model=_ScriptedModel(outputs=deque([plan, ModelReply(text='ok')])),
        executor=_RaisingExecutor(),
    )
    inp = _ScriptedInput(lines=deque(['first', 'second']))
    out = _CapturingOutput()
    await Repl(loop, inp, out, banner='').run()

    assert "[error] plan rejected: step 1: missing required input 'tz'" in out.text
    assert 'ok\n' in out.text


async def test_repl_exits_cleanly_on_eof() -> None:
    loop, _, _ = _wire(model_outputs=[])
    inp = _ScriptedInput(lines=deque())
    out = _CapturingOutput()
    await Repl(loop, inp, out, banner='B\n').run()
    # Banner printed, EOF newline appended, no errors raised.
    assert out.text == 'B\n\n'


async def test_repl_writes_banner_before_first_prompt() -> None:
    loop, _, _ = _wire(model_outputs=[])
    inp = _ScriptedInput(lines=deque())
    out = _CapturingOutput()
    await Repl(loop, inp, out, banner='hello\n').run()
    assert out.chunks[0] == 'hello\n'


async def test_gate_handler_pauses_and_resumes_active_indicator() -> None:
    """PR #18 review (Copilot): the spinner kept animating during gate prompts.

    Fix: `ReplGateHandler.on_gate` wraps its prompt in
    `suspend_indicator()`, which pauses any registered
    `IndicatorControl` for the duration. This test registers a
    recording indicator via the same contextvar API used by
    `InferenceIndicator` and asserts pause/resume fire exactly once
    around the gate interaction.
    """
    from butter_agent.core.repl import register_active_indicator, unregister_active_indicator

    @dataclass
    class _RecordingIndicator:
        pauses: int = 0
        resumes: int = 0

        def pause(self) -> None:
            self.pauses += 1

        def resume(self) -> None:
            self.resumes += 1

    indicator = _RecordingIndicator()
    token = register_active_indicator(indicator)
    try:
        inp = _ScriptedInput(lines=deque(['y']))
        out = _CapturingOutput()
        await ReplGateHandler(inp, out).on_gate(_step(), Gate.CONFIRM, {})
    finally:
        unregister_active_indicator(token)

    assert indicator.pauses == 1
    assert indicator.resumes == 1


# --- ReplGateHandler ---------------------------------------------------------


def _step(gate: str = 'confirm') -> PlanStep:
    return PlanStep(step=1, plugin='notes', capability='create', inputs={}, gate=gate)


@pytest.mark.parametrize('answer', ['y', 'Y', 'yes', 'YES', ' yes '])
async def test_gate_handler_continues_on_yes(answer: str) -> None:
    inp = _ScriptedInput(lines=deque([answer]))
    decision = await ReplGateHandler(inp, _CapturingOutput()).on_gate(
        _step(),
        Gate.CONFIRM,
        {},
    )
    assert decision is GateDecision.CONTINUE


@pytest.mark.parametrize('answer', ['', 'n', 'no', 'maybe', 'sure'])
async def test_gate_handler_aborts_on_anything_else(answer: str) -> None:
    inp = _ScriptedInput(lines=deque([answer]))
    decision = await ReplGateHandler(inp, _CapturingOutput()).on_gate(
        _step(),
        Gate.CONFIRM,
        {},
    )
    assert decision is GateDecision.ABORT


async def test_gate_handler_aborts_on_eof() -> None:
    inp = _ScriptedInput(lines=deque())  # EOF immediately
    out = _CapturingOutput()
    decision = await ReplGateHandler(inp, out).on_gate(_step(), Gate.CONFIRM, {})
    assert decision is GateDecision.ABORT


async def test_gate_handler_surfaces_prior_outputs_on_human_gate() -> None:
    inp = _ScriptedInput(lines=deque(['y']))
    out = _CapturingOutput()
    prior: Mapping[str, Mapping[str, object]] = {'note': {'id': 42, 'title': 'hi'}}
    await ReplGateHandler(inp, out).on_gate(_step('human'), Gate.HUMAN, prior)
    assert '  prior outputs:' in out.text
    assert '$note:' in out.text
    assert "'id': 42" in out.text


async def test_gate_handler_omits_prior_block_on_confirm_gate() -> None:
    inp = _ScriptedInput(lines=deque(['y']))
    out = _CapturingOutput()
    prior: Mapping[str, Mapping[str, object]] = {'note': {'id': 42}}
    await ReplGateHandler(inp, out).on_gate(_step('confirm'), Gate.CONFIRM, prior)
    # CONFIRM is a structural yes/no — don't dump variable pool.
    assert 'prior outputs' not in out.text


async def test_gate_handler_omits_prior_block_when_empty() -> None:
    inp = _ScriptedInput(lines=deque(['y']))
    out = _CapturingOutput()
    await ReplGateHandler(inp, out).on_gate(_step('human'), Gate.HUMAN, {})
    assert 'prior outputs' not in out.text


# --- Stdio defaults ----------------------------------------------------------


async def test_stdio_input_source_reads_until_eof() -> None:
    src = io.StringIO('first\nsecond\n')
    prompts = io.StringIO()
    inp = StdioInputSource(stream=src, prompt_stream=prompts)
    assert await inp.read_line('> ') == 'first'
    assert await inp.read_line('> ') == 'second'
    with pytest.raises(EOFError):
        await inp.read_line('> ')
    assert prompts.getvalue() == '> > > '


async def test_stdio_input_source_strips_trailing_newline_not_inner_whitespace() -> None:
    inp = StdioInputSource(stream=io.StringIO('  hi  \n'), prompt_stream=io.StringIO())
    assert await inp.read_line('') == '  hi  '


def test_stdio_output_writes_and_flushes() -> None:
    @dataclass
    class _TrackingStream:
        buf: list[str] = field(default_factory=list)
        flushes: int = 0

        def write(self, text: str) -> None:
            self.buf.append(text)

        def flush(self) -> None:
            self.flushes += 1

    stream = _TrackingStream()
    StdioOutput(stream=stream).write('hello')  # type: ignore[arg-type]
    assert stream.buf == ['hello']
    assert stream.flushes == 1


# --- indicator_factory seam -------------------------------------------------


async def test_repl_calls_indicator_factory_around_each_turn() -> None:
    """Each non-command turn must enter the indicator context once.

    The indicator is the seam prompt_toolkit's spinner hangs off; tests
    inject a recording stub so the wiring can be verified without
    pulling a TTY into the test environment.
    """
    enter_count = 0
    exit_count = 0

    class _RecordingIndicator:
        async def __aenter__(self) -> _RecordingIndicator:
            nonlocal enter_count
            enter_count += 1
            return self

        async def __aexit__(self, *_: object) -> None:
            nonlocal exit_count
            exit_count += 1

    loop, _, _ = _wire(model_outputs=[ModelReply(text='ok')])
    inp = _ScriptedInput(lines=deque(['ping']))
    out = _CapturingOutput()
    await Repl(loop, inp, out, banner='', indicator_factory=_RecordingIndicator).run()

    assert enter_count == 1
    assert exit_count == 1


async def test_repl_indicator_not_invoked_on_blank_input_or_commands() -> None:
    """Blank lines and slash commands skip the indicator — no inference happens.

    The indicator wraps `AgentLoop.run_turn` only. Slash commands are
    dispatched locally and blank lines are dropped before any model
    call, so neither should spin up the spinner.
    """
    calls = 0

    class _CountingIndicator:
        async def __aenter__(self) -> _CountingIndicator:
            nonlocal calls
            calls += 1
            return self

        async def __aexit__(self, *_: object) -> None:
            return None

    loop, _, _ = _wire(model_outputs=[])
    inp = _ScriptedInput(lines=deque(['', '   ', '/unknown']))
    out = _CapturingOutput()
    await Repl(loop, inp, out, banner='', indicator_factory=_CountingIndicator).run()

    assert calls == 0
