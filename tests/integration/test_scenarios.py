"""End-to-end scenario tests covering the spec's 5 acceptance scenarios.

The model is the only non-deterministic part of butter; everything else
(plan validator, executor, gates, plugins, prompt renderer, REPL) is
deterministic given the model's output. So these tests mock only the
`ModelClient` — every other seam runs the same code path the live REPL
uses, including a real `DefaultTaskExecutor`, real `RegistryBuilder`,
real `ReplGateHandler`, real `Repl`, and a small in-test clock plugin
that returns deterministic values.

What this catches that unit tests don't:
- Plan validation against real `PluginManifest` shapes.
- `$alias.field` resolution across real plugin outputs.
- Gate handler firing on a real `local-write` / `confirm` plan step.
- Synthesis prompt being assembled with real `ExecutionResult` outputs.
- Repl renderer integration with `synthesis_reply` / `halted_at_step` /
  `failed_at_step`.

What this does NOT catch:
- Whether the live model emits a given plan for a given prompt. That
  is model behaviour, not butter behaviour, and lives in
  `tests/integration/test_live_ollama.py` behind `@pytest.mark.live`.

Scenarios (numbered to match
`specs/development/model-baseline-and-input-schema-prompt.md`):
1. Single-tool, single-call: "what time is it?"
2. Single-tool with non-obvious input: "what time is it in Tokyo?"
3. Multi-call without dependency: simulates "time difference between
   here and Tokyo" — two independent `clock.now_in_zone` calls.
4. Halt-and-confirm: a plan with `gate: confirm` where the user
   declines. Plugin must not run; synthesis must not fire.
5. Plain chat with no capability match: model returns a `reply` even
   though capabilities are registered. No plan attempted.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import pytest

from butter_agent.core.context_manager import (
    CapabilityDescriptor,
    DefaultContextManager,
    InMemoryConversationHistory,
    PluginIndexEntry,
)
from butter_agent.core.loop import AgentLoop, DiscoverySelection, ModelContext, ModelOutput, ModelReply, PlanStep, TaskPlan
from butter_agent.core.registry import BlastRadius, Capability, PluginManifest, RegistryBuilder
from butter_agent.core.repl import Repl, ReplGateHandler
from butter_agent.core.task_executor import DefaultTaskExecutor

# --- Test plumbing -----------------------------------------------------------


@dataclass
class _ScriptedInput:
    """Queued input lines; EOFError when exhausted."""

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
class _ScriptedModel:
    """Yields the next queued ModelOutput on each call. Raises if exhausted."""

    outputs: deque[ModelOutput]
    contexts_seen: list[ModelContext] = field(default_factory=list)

    async def generate(self, context: ModelContext) -> ModelOutput:
        self.contexts_seen.append(context)
        return self.outputs.popleft()


class _RecordingClockPlugin:
    """Deterministic stand-in for the real clock plugin.

    Real `clock` plugin lives outside this repo; embedding a tiny
    recording variant here keeps the test self-contained. The
    capability set and output shape match the real manifest so the
    test exercises the same prompt rendering and `$alias.field` paths.
    """

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, object]]] = []

    async def execute(
        self,
        capability: str,
        inputs: dict[str, object],
        context: object,
    ) -> dict[str, object]:
        del context
        self.calls.append((capability, dict(inputs)))
        if capability == 'now':
            return {'time': '2026-05-14T15:00:00+02:00', 'tz': 'CEST'}
        if capability == 'now_in_zone':
            tz = str(inputs.get('tz', 'UTC'))
            # Deterministic: 8 hours ahead of CEST for Tokyo. Real plugin
            # consults zoneinfo; the synthesis turn renders whatever we
            # return so a static value is fine for assertion purposes.
            time = '2026-05-14T22:00:00+09:00' if tz == 'Asia/Tokyo' else '2026-05-14T15:00:00+02:00'
            return {'time': time, 'tz': tz}
        if capability == 'diff':
            return {'seconds': 25200.0, 'human': '7h'}
        raise ValueError(f'unknown capability: {capability}')


def _clock_manifest() -> PluginManifest:
    """Manifest matching the real clock plugin's shape.

    Mirror of `~/Workspace/butter-plugin-clock/manifest.toml`. Kept
    inline so the integration test doesn't depend on the operator's
    workspace layout.
    """
    return PluginManifest(
        name='clock',
        version='0.1.0',
        blast_radius=BlastRadius.READ_ONLY,
        entrypoint='test:Plugin',
        capabilities=(
            Capability(
                name='now',
                description="Current date and time in the system's default timezone (ISO 8601).",
                input_schema={},
                output_schema={'time': 'string', 'tz': 'string'},
            ),
            Capability(
                name='now_in_zone',
                description='Current date and time in the given IANA timezone.',
                input_schema={'tz': 'string'},
                output_schema={'time': 'string', 'tz': 'string'},
            ),
            Capability(
                name='diff',
                description='Duration between two ISO timestamps (b - a).',
                input_schema={'a': 'string', 'b': 'string'},
                output_schema={'seconds': 'number', 'human': 'string'},
            ),
        ),
    )


def _build_repl(
    *,
    user_inputs: list[str],
    model_outputs: list[ModelOutput],
) -> tuple[Repl, _ScriptedInput, _CapturingOutput, _ScriptedModel, _RecordingClockPlugin]:
    """Compose a real Repl with the only seam scripted being the model client.

    Mirrors `build_repl` in `app.py` but without I/O detection, history
    persistence, or Ollama wiring. Returns the recording handles so
    tests can assert on plugin invocations and the rendered transcript.
    """
    plugin = _RecordingClockPlugin()
    builder = RegistryBuilder(max_blast_radius=BlastRadius.NETWORK)
    builder.register(_clock_manifest(), plugin)
    registry = builder.build()

    history = InMemoryConversationHistory()
    context_manager = DefaultContextManager(registry, history)
    model = _ScriptedModel(outputs=deque(model_outputs))

    inp = _ScriptedInput(lines=deque(user_inputs))
    out = _CapturingOutput()
    gate_handler = ReplGateHandler(inp, out)
    executor = DefaultTaskExecutor(registry, gate_handler)
    loop = AgentLoop(context_manager, model, executor, history=history)
    repl = Repl(loop, inp, out, banner='')
    return repl, inp, out, model, plugin


# --- Scenarios --------------------------------------------------------------


async def test_scenario_1_single_tool_single_call() -> None:
    """Spec scenario #1: model emits one-step plan, plugin runs, synthesis replies."""
    plan = TaskPlan(steps=(PlanStep(step=1, plugin='clock', capability='now', inputs={}, gate='none', outputs_as='t'),))
    repl, _inp, out, _model, plugin = _build_repl(
        user_inputs=['what time is it?'],
        model_outputs=[plan, ModelReply(text='It is 15:00 CEST.')],
    )

    await repl.run()

    assert plugin.calls == [('now', {})]
    assert 'It is 15:00 CEST.' in out.text


async def test_scenario_2_single_tool_with_non_obvious_input() -> None:
    """Spec scenario #2: required-input `tz` surfaces in the prompt and the plan supplies it."""
    plan = TaskPlan(
        steps=(PlanStep(step=1, plugin='clock', capability='now_in_zone', inputs={'tz': 'Asia/Tokyo'}, gate='none', outputs_as='t'),),
    )
    repl, _inp, out, model, plugin = _build_repl(
        user_inputs=['what time is it in Tokyo?'],
        model_outputs=[plan, ModelReply(text='It is 22:00 in Tokyo.')],
    )

    await repl.run()

    assert plugin.calls == [('now_in_zone', {'tz': 'Asia/Tokyo'})]
    # The intent-recognition context must carry the `now_in_zone` capability with `tz`
    # in its required_inputs — that's what the prompt rendering uses to print
    # `(requires: tz)` to the model.
    intent_context = model.contexts_seen[0]
    capabilities = intent_context.payload.get('capabilities')
    assert isinstance(capabilities, tuple)
    assert any(isinstance(c, CapabilityDescriptor) and c.capability == 'now_in_zone' and 'tz' in c.required_inputs for c in capabilities)
    assert 'It is 22:00 in Tokyo.' in out.text


async def test_scenario_3_multi_call_without_dependency() -> None:
    """Spec scenario #3: two independent `now_in_zone` calls feed a `diff`.

    Tests `$alias.field` resolution across multiple real plugin calls.
    The diff step's inputs reference earlier steps' `time` fields.
    """
    plan = TaskPlan(
        steps=(
            PlanStep(step=1, plugin='clock', capability='now_in_zone', inputs={'tz': 'CEST'}, gate='none', outputs_as='here'),
            PlanStep(step=2, plugin='clock', capability='now_in_zone', inputs={'tz': 'Asia/Tokyo'}, gate='none', outputs_as='there'),
            PlanStep(step=3, plugin='clock', capability='diff', inputs={'a': '$here.time', 'b': '$there.time'}, gate='none', outputs_as='d'),
        ),
    )
    repl, _inp, out, _model, plugin = _build_repl(
        user_inputs=['time difference between here and Tokyo?'],
        model_outputs=[plan, ModelReply(text='Tokyo is 7 hours ahead.')],
    )

    await repl.run()

    # All three steps ran; step 3 received the resolved `$alias.field` values, not the literal `$...` strings.
    assert plugin.calls == [
        ('now_in_zone', {'tz': 'CEST'}),
        ('now_in_zone', {'tz': 'Asia/Tokyo'}),
        ('diff', {'a': '2026-05-14T15:00:00+02:00', 'b': '2026-05-14T22:00:00+09:00'}),
    ]
    assert 'Tokyo is 7 hours ahead.' in out.text


async def test_scenario_4_halt_and_confirm_aborts_plan_without_running_step() -> None:
    """Spec scenario #4: `gate: confirm` user-rejection halts the plan.

    First user input is the question; second input is the "n" reply at
    the gate prompt. The plugin must not be called, the synthesis turn
    must not fire, and the REPL must render `[halted ...]`.
    """
    plan = TaskPlan(
        steps=(PlanStep(step=1, plugin='clock', capability='now', inputs={}, gate='confirm', outputs_as='t'),),
    )
    repl, _inp, out, model, plugin = _build_repl(
        user_inputs=['what time is it?', 'n'],
        model_outputs=[plan],  # only the intent-recognition call should occur
    )

    await repl.run()

    assert plugin.calls == []
    # Synthesis must not have been called — only the intent-recognition turn.
    assert len(model.contexts_seen) == 1
    assert '[halted at step 1]' in out.text


async def test_scenario_5_plain_chat_with_no_capability_match() -> None:
    """Spec scenario #5: model returns a `reply`, not a plan, even with capabilities registered.

    Pins the contract that the prompt does not force planning when the
    user's request doesn't match any registered capability. The clock
    plugin is loaded; the model is scripted to chat anyway.
    """
    repl, _inp, out, _model, plugin = _build_repl(
        user_inputs=['tell me a joke'],
        model_outputs=[ModelReply(text='Why did the chicken cross the road? Synthesizers were on the other side.')],
    )

    await repl.run()

    # No plugin invocation, no plan attempted.
    assert plugin.calls == []
    assert 'chicken' in out.text


# --- Capability discovery acceptance (spec migration step 4) ----------------


class _RecordingPlugin:
    """Records (capability, inputs); returns a canned dict per capability."""

    def __init__(self, returns: dict[str, dict[str, object]]) -> None:
        self._returns = returns
        self.calls: list[tuple[str, dict[str, object]]] = []

    async def execute(self, capability: str, inputs: dict[str, object], context: object) -> dict[str, object]:
        del context
        self.calls.append((capability, dict(inputs)))
        return self._returns.get(capability, {})


def _wide_manifest(name: str, caps: tuple[tuple[str, str, dict[str, object]], ...]) -> PluginManifest:
    return PluginManifest(
        name=name,
        version='0.1.0',
        blast_radius=BlastRadius.READ_ONLY,
        entrypoint='test:Plugin',
        capabilities=tuple(Capability(name=cn, description=cd, input_schema=ci, output_schema={}) for cn, cd, ci in caps),
    )


async def test_discovery_recovers_filesystem_registered_last() -> None:
    """Spec acceptance: the 2026-05-15 failing prompt plans `filesystem.

    read_file{path}` with `filesystem` registered LAST and sharing no
    keyword with the request — the exact case `KeywordCapabilityFilter`
    blanked (registration-order fallback). With discovery active the
    model is shown a Tier-1 index naming `filesystem`, selects it, and
    the Tier-2 context carries `read_file`'s schema (incl. required
    `path`) regardless of registration position. The model is scripted —
    this pins butter's plumbing, not model behaviour.
    """
    clock = _RecordingPlugin({'now': {'time': '15:00', 'tz': 'CEST'}})
    notes = _RecordingPlugin({'create': {'id': 1}})
    files = _RecordingPlugin({'read_file': {'content': '[project]\ndependencies = ["pytest"]'}})

    builder = RegistryBuilder(max_blast_radius=BlastRadius.NETWORK)
    builder.register(_clock_manifest(), clock)
    builder.register(
        _wide_manifest('notes', (('create', 'Create a note', {'body': 'string'}), ('list', 'List notes', {}), ('delete', 'Delete a note', {'id': 'integer'}))),
        notes,
    )
    # filesystem registered LAST and intentionally keyword-disjoint from
    # "what dependencies does pyproject have".
    builder.register(
        _wide_manifest(
            'filesystem',
            (
                ('read_file', 'Return the contents of a file at a path', {'path': 'string'}),
                ('list_dir', 'List entries in a directory', {'path': 'string'}),
                ('write_file', 'Write contents to a path', {'path': 'string', 'content': 'string'}),
            ),
        ),
        files,
    )
    registry = builder.build()
    # 3 + 3 + 3 = 9 user-facing caps > default threshold 8, 3 plugins → active.
    history = InMemoryConversationHistory()
    context_manager = DefaultContextManager(registry, history, capability_discovery=True)
    assert context_manager.discovery_active is True

    model = _ScriptedModel(
        outputs=deque(
            [
                DiscoverySelection(plugins=('filesystem',)),
                TaskPlan(steps=(PlanStep(step=1, plugin='filesystem', capability='read_file', inputs={'path': 'pyproject.toml'}, gate='none', outputs_as='f'),)),
                ModelReply(text='pyproject depends on pytest.'),
            ],
        ),
    )
    inp = _ScriptedInput(lines=deque(['what dependencies does pyproject have']))
    out = _CapturingOutput()
    executor = DefaultTaskExecutor(registry, ReplGateHandler(inp, out))
    repl = Repl(AgentLoop(context_manager, model, executor, history=history), inp, out, banner='')

    await repl.run()

    # Tier-1 index named filesystem despite last registration / no keyword overlap.
    tier1 = model.contexts_seen[0].payload['plugin_index']
    assert isinstance(tier1, tuple)
    assert any(isinstance(e, PluginIndexEntry) and e.name == 'filesystem' for e in tier1)
    assert 'capabilities' not in model.contexts_seen[0].payload
    # Tier-2 carried read_file's full schema, incl. required `path`.
    tier2 = model.contexts_seen[1].payload['capabilities']
    assert isinstance(tier2, tuple)
    read_file = next(c for c in tier2 if isinstance(c, CapabilityDescriptor) and c.capability == 'read_file')
    assert read_file.plugin == 'filesystem'
    assert 'path' in read_file.required_inputs
    # Plan executed against the real executor; synthesis replied.
    assert files.calls == [('read_file', {'path': 'pyproject.toml'})]
    assert 'pyproject depends on pytest.' in out.text


# --- Live-model placeholder (opt-in via `-m live`) --------------------------


@pytest.mark.live
async def test_live_ollama_placeholder_clock_now() -> None:
    """Placeholder for live-Ollama integration tests.

    Hits the configured homelab Ollama with the real OllamaModelClient
    and asserts the model emits a plan invoking `clock.now`. Excluded
    from the default test run via `pytest -m "not live"` in
    pyproject.toml; runnable with `uv run pytest -m live`.

    Implementation deferred until we want a regression net for
    prompt-vs-model behaviour; see `plugin-failure-recovery.md`.
    """
    pytest.skip('Live-Ollama integration not yet implemented; placeholder for future work.')
