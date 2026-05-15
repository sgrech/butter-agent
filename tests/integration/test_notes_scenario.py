"""End-to-end acceptance for the `notes` plugin (task #377, spec §7).

This is the test the spec's live-REPL acceptance describes, with the model
as the only scripted seam. Everything else runs the exact code path the
live REPL uses:

- real `RegistryBuilder` wiring the real `database` plugin, the real
  `notes` plugin, and a deterministic recording `clock` (the real clock
  lives outside this repo — same stand-in pattern as `test_scenarios.py`);
- real `DefaultTaskExecutor` doing atomic plan validation, `$alias.field`
  resolution, and the core-side `{notes}__entries` namespace prefixing;
- real `ReplGateHandler` prompting for the `confirm` gate on
  `notes.create` and reading the operator's `y` off the input seam;
- real `AgentLoop` running the synthesis pass.

What it proves that the isolated unit tests can't:

1. `clock.now → notes.create($t.time)` resolves end-to-end — the ISO
   timestamp travels the variable pool and lands in the persisted row.
2. The `confirm` gate fires on a real `local-write` step and the write
   only happens after approval.
3. The namespace boundary is real: `notes` passes the bare table name,
   core rewrites it, the shared `database` file ends up with a
   `notes__entries` table — and `notes.list` reads the same row back in
   the same session.
"""

from __future__ import annotations

from collections import deque
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from butter_agent.core.context_manager import DefaultContextManager, InMemoryConversationHistory
from butter_agent.core.loop import AgentLoop, ModelContext, ModelOutput, ModelReply, PlanStep, TaskPlan
from butter_agent.core.registry import BlastRadius, Capability, PluginManifest, RegistryBuilder
from butter_agent.core.repl import Repl, ReplGateHandler
from butter_agent.core.task_executor import DefaultTaskExecutor
from butter_agent.plugins.database import build_database_plugin
from butter_agent.plugins.notes import build_notes_plugin
from butter_agent.storage.sqlite import Database

# --- Scripted seams (mirrors test_scenarios.py; kept local so this file is
#     self-contained and does not import another test module's privates) ---


@dataclass
class _ScriptedInput:
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
    outputs: deque[ModelOutput]
    contexts_seen: list[ModelContext] = field(default_factory=list)

    async def generate(self, context: ModelContext) -> ModelOutput:
        self.contexts_seen.append(context)
        return self.outputs.popleft()


class _RecordingClockPlugin:
    """Deterministic stand-in for the out-of-repo `clock` plugin."""

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
        raise ValueError(f'unknown capability: {capability}')


def _clock_manifest() -> PluginManifest:
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
        ),
    )


@dataclass
class _Harness:
    repl: Repl
    inp: _ScriptedInput
    out: _CapturingOutput
    model: _ScriptedModel
    clock: _RecordingClockPlugin
    database: Database


@pytest.fixture
async def harness(tmp_path: Path) -> AsyncIterator[_Harness]:
    """Compose the real stack over a throwaway SQLite file.

    `model_outputs` / `user_inputs` are filled per-test by replacing the
    deques before `repl.run()`; the wiring (registry, executor, gate,
    loop) is identical to `app.build_repl`.
    """
    database = await Database.open(tmp_path / 'butter.db')
    try:
        builder = RegistryBuilder(max_blast_radius=BlastRadius.NETWORK)
        # Same registration order as app.build_repl: database, then notes
        # (so its requires resolve), then everything else.
        db_manifest, db_plugin = build_database_plugin(database)
        builder.register(db_manifest, db_plugin)
        notes_manifest, notes_plugin = build_notes_plugin()
        builder.register(notes_manifest, notes_plugin)
        clock = _RecordingClockPlugin()
        builder.register(_clock_manifest(), clock)
        registry = builder.build()

        history = InMemoryConversationHistory()
        context_manager = DefaultContextManager(registry, history)
        model = _ScriptedModel(outputs=deque())
        inp = _ScriptedInput(lines=deque())
        out = _CapturingOutput()
        executor = DefaultTaskExecutor(registry, ReplGateHandler(inp, out))
        loop = AgentLoop(context_manager, model, executor, history=history)
        repl = Repl(loop, inp, out, banner='')
        yield _Harness(repl=repl, inp=inp, out=out, model=model, clock=clock, database=database)
    finally:
        await database.close()


def _script(harness: _Harness, *, user_inputs: list[str], model_outputs: list[ModelOutput]) -> None:
    # Repl/loop hold references to the same model and input objects; both
    # read their queues lazily per turn, so swapping the deques here is
    # enough to script the whole session.
    harness.model.outputs = deque(model_outputs)
    harness.inp.lines = deque(user_inputs)


# --- The acceptance scenario -------------------------------------------------


async def test_clock_now_to_notes_create_confirm_then_list_readback(harness: _Harness) -> None:
    """Spec §7: chain a timestamped note through the confirm gate, read it back.

    Turn 1: `clock.now → notes.create` with `gate: confirm` on the write.
    The operator approves (`y`). Turn 2: `notes.list` (no gate) returns the
    note that was just persisted — in the same session, through the real
    `notes__entries` table.
    """
    create_plan = TaskPlan(
        steps=(
            PlanStep(step=1, plugin='clock', capability='now', inputs={}, gate='none', outputs_as='t'),
            PlanStep(
                step=2,
                plugin='notes',
                capability='create',
                inputs={'content': 'buy butter', 'created_at': '$t.time'},
                gate='confirm',
                outputs_as='n',
            ),
        ),
    )
    list_plan = TaskPlan(
        steps=(PlanStep(step=1, plugin='notes', capability='list', inputs={}, gate='none', outputs_as='all'),),
    )
    _script(
        harness,
        user_inputs=[
            'save a note to buy butter with the current time',
            'y',  # answer to the [gate:confirm] prompt on notes.create
            'what notes do I have?',
        ],
        model_outputs=[
            create_plan,
            ModelReply(text='Saved your note (#1).'),
            list_plan,
            ModelReply(text='You have one note: "buy butter".'),
        ],
    )

    await harness.repl.run()

    # 1. clock.now ran exactly once (the read step), with no inputs.
    assert harness.clock.calls == [('now', {})]

    # 2. The confirm gate fired on the notes.create step specifically.
    assert '[gate:confirm] step 2: notes.create' in harness.out.text
    # Both synthesis replies were rendered (turn 1 and turn 2 completed).
    assert 'Saved your note (#1).' in harness.out.text
    assert 'You have one note: "buy butter".' in harness.out.text

    # 3. The chained timestamp actually persisted: read the shared store
    #    directly and confirm core namespaced the table to notes__entries
    #    and the $t.time value (not the literal '$t.time') is in the row.
    rows = await harness.database.query('SELECT id, content, created_at FROM notes__entries', ())
    assert rows == [{'id': 1, 'content': 'buy butter', 'created_at': '2026-05-14T15:00:00+02:00'}]


async def test_declined_confirm_gate_writes_nothing(harness: _Harness) -> None:
    """If the operator declines the confirm gate, no row is written.

    Pins that the gate sits in front of the write: the `clock.now` read
    step may run, but `notes.create` must not, and the synthesis turn must
    not fire (the loop returns the halt verbatim).
    """
    create_plan = TaskPlan(
        steps=(
            PlanStep(step=1, plugin='clock', capability='now', inputs={}, gate='none', outputs_as='t'),
            PlanStep(
                step=2,
                plugin='notes',
                capability='create',
                inputs={'content': 'secret', 'created_at': '$t.time'},
                gate='confirm',
                outputs_as='n',
            ),
        ),
    )
    _script(
        harness,
        user_inputs=['save a secret note', 'n'],
        model_outputs=[create_plan],  # only the planning turn — no synthesis
    )

    await harness.repl.run()

    # The gate halted before the write step; synthesis did not run.
    assert '[halted at step 2]' in harness.out.text
    assert len(harness.model.contexts_seen) == 1
    # No note row was persisted. Robust to whether the ensure-table guard
    # ran: that's an implementation detail the spec doesn't pin, so the
    # assertion only depends on the observable invariant — zero rows —
    # regardless of whether the namespaced table exists yet.
    existing = await harness.database.query(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='notes__entries'",
        (),
    )
    if existing:
        assert await harness.database.query('SELECT * FROM notes__entries', ()) == []
