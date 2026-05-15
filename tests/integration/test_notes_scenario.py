"""End-to-end host proof for a `local-write` plugin over the real `database`.

`notes` itself now lives in its own repo (`butter-plugin-notes`, loaded
via `[[plugin]]`); its capability logic is proven there. What stays
in-repo is the proof of the **host machinery** a write-plugin depends on,
which the `clock` scenario in `test_scenarios.py` cannot cover (clock is
read-only and calls nothing):

- the real `DefaultTaskExecutor` building a real `_PluginContext`,
- core-side namespace prefixing (`entries` → `notes__entries`) applied by
  that context, not the plugin,
- `requires` enforcement on a real plugin-to-plugin `ctx.call`,
- the `confirm` gate firing on a real `local-write` step,
- `$alias.field` variable-pool resolution (`clock.now → notes.create`),

all against the real bundled `database` plugin and a real SQLite file.

`_RecordingNotesStub` is a deliberately minimal stand-in — the same
pattern as `_RecordingClockPlugin` standing in for the out-of-repo clock.
It is *not* the real notes plugin and is not asserting notes' behaviour;
it is the smallest `local-write` plugin that exercises the host seams
above so this coverage doesn't depend on an external checkout.
"""

from __future__ import annotations

from collections import deque
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from butter_agent.core.context_manager import DefaultContextManager, InMemoryConversationHistory
from butter_agent.core.loop import AgentLoop, ModelContext, ModelOutput, ModelReply, PlanStep, TaskPlan
from butter_agent.core.registry import BlastRadius, Capability, PluginContext, PluginManifest, RegistryBuilder
from butter_agent.core.repl import Repl, ReplGateHandler
from butter_agent.core.task_executor import DefaultTaskExecutor
from butter_agent.plugins.database import build_database_plugin
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


# Column schema the real notes plugin defines; duplicated here only so the
# stub persists rows the post-condition can read back. Not a contract —
# the real schema lives in butter-plugin-notes.
_NOTES_COLUMNS: dict[str, object] = {
    'content': {'type': 'text', 'not_null': True},
    'created_at': {'type': 'datetime', 'not_null': True},
}


class _RecordingNotesStub:
    """Smallest `local-write` plugin that drives the real `database` plugin.

    Stands in for the externalized notes plugin the way
    `_RecordingClockPlugin` stands in for clock. It persists via
    `context.call` into `database.*` with the **bare** table name so the
    real `_PluginContext` applies the `notes__` prefix — exercising the
    host's namespace boundary, `requires` check, gate, and variable pool.
    Only the two capabilities the scenarios use are implemented.
    """

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, object]]] = []

    async def execute(
        self,
        capability: str,
        inputs: dict[str, object],
        context: PluginContext,
    ) -> dict[str, object]:
        self.calls.append((capability, dict(inputs)))
        if capability == 'create':
            await context.call('database.define_table', {'table': 'entries', 'columns': _NOTES_COLUMNS})
            inserted = await context.call(
                'database.insert',
                {'table': 'entries', 'row': {'content': inputs['content'], 'created_at': inputs['created_at']}},
            )
            return {'note_id': inserted['id'], 'created_at': inputs['created_at']}
        if capability == 'list':
            result = await context.call('database.select', {'table': 'entries', 'order_by': 'id'})
            return {'notes': result['rows']}
        raise ValueError(f'unknown capability: {capability}')


def _notes_manifest() -> PluginManifest:
    return PluginManifest(
        name='notes',
        version='0.1.0',
        blast_radius=BlastRadius.LOCAL_WRITE,
        entrypoint='test:Plugin',
        capabilities=(
            Capability(
                name='create',
                description='Save a free-form note. Optionally accepts a chained created_at.',
                input_schema={'content': 'string'},
                output_schema={'note_id': 'integer', 'created_at': 'string'},
            ),
            Capability(
                name='list',
                description='List saved notes oldest-first.',
                input_schema={},
                output_schema={'notes': 'array'},
            ),
        ),
        requires=('database.define_table', 'database.insert', 'database.select'),
    )


@dataclass
class _Harness:
    repl: Repl
    inp: _ScriptedInput
    out: _CapturingOutput
    model: _ScriptedModel
    clock: _RecordingClockPlugin
    notes: _RecordingNotesStub
    database: Database


@pytest.fixture
async def harness(tmp_path: Path) -> AsyncIterator[_Harness]:
    """Compose the real stack over a throwaway SQLite file.

    `model_outputs` / `user_inputs` are filled per-test by replacing the
    deques before `repl.run()`; the wiring (registry, executor, gate,
    loop) is identical to `app.build_repl`: the built-in `database`
    registers first, the external-like plugins after, so notes' `requires`
    resolve at build.
    """
    database = await Database.open(tmp_path / 'butter.db')
    try:
        builder = RegistryBuilder(max_blast_radius=BlastRadius.NETWORK)
        db_manifest, db_plugin = build_database_plugin(database)
        builder.register(db_manifest, db_plugin)
        notes = _RecordingNotesStub()
        builder.register(_notes_manifest(), notes)
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
        yield _Harness(repl=repl, inp=inp, out=out, model=model, clock=clock, notes=notes, database=database)
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
    """Chain a timestamped note through the confirm gate, read it back.

    Turn 1: `clock.now → notes.create` with `gate: confirm` on the write.
    The operator approves (`y`). Turn 2: `notes.list` (no gate) returns
    the note that was just persisted — in the same session, through the
    real `notes__entries` table the core namespaced on the plugin's
    behalf.
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
    # The notes plugin received the chained $t.time resolved to clock's
    # output (not the literal '$t.time' string).
    assert ('create', {'content': 'buy butter', 'created_at': '2026-05-14T15:00:00+02:00'}) in harness.notes.calls

    # 2. The confirm gate fired on the notes.create step specifically.
    assert '[gate:confirm] step 2: notes.create' in harness.out.text
    # Both synthesis replies were rendered (turn 1 and turn 2 completed).
    assert 'Saved your note (#1).' in harness.out.text
    assert 'You have one note: "buy butter".' in harness.out.text

    # 3. The chained timestamp actually persisted, and core namespaced the
    #    table to notes__entries (the plugin only ever passed 'entries').
    rows = await harness.database.query('SELECT id, content, created_at FROM notes__entries', ())
    assert rows == [{'id': 1, 'content': 'buy butter', 'created_at': '2026-05-14T15:00:00+02:00'}]


async def test_declined_confirm_gate_writes_nothing(harness: _Harness) -> None:
    """If the operator declines the confirm gate, no row is written.

    The `clock.now` read step may run, but `notes.create` must not, and
    the synthesis turn must not fire (the loop returns the halt verbatim).
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
    assert harness.notes.calls == []  # the write plugin was never invoked
    # No note row was persisted. Robust to whether the table was created:
    # the stub only defines it inside `create`, which never ran here, but
    # the assertion holds regardless of that implementation detail.
    existing = await harness.database.query(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='notes__entries'",
        (),
    )
    if existing:
        assert await harness.database.query('SELECT * FROM notes__entries', ()) == []
