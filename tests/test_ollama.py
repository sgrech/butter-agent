"""Tests for the Ollama `ModelClient` adapter.

The adapter has two non-trivial responsibilities that need separate
coverage:

- **Prompt construction**: given a `ModelContext` populated by
  `DefaultContextManager`, the adapter renders a chat-style messages
  array that surfaces capabilities, history, and memory without
  dumping the full registry.
- **Output parsing**: given a JSON response from Ollama, the adapter
  discriminates `{"type": "reply"}` vs `{"type": "plan"}`, validates
  shape strictly, and raises `ModelProtocolError` on any structural
  problem — never guesses, never returns a partial result.

Tests inject a fake `Transport` so no real Ollama is required.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import pytest

from butter_agent.core.context_manager import (
    CapabilityDescriptor,
    ConversationEntry,
    MemorySnippet,
    PluginIndexEntry,
)
from butter_agent.core.loop import (
    DiscoverySelection,
    ExecutionResult,
    ModelContext,
    ModelProtocolError,
    ModelReply,
    PlanStep,
    TaskPlan,
    Turn,
)
from butter_agent.model.ollama import OllamaModelClient

# --- Test plumbing -----------------------------------------------------------


@dataclass
class _FakeTransport:
    """Returns canned responses, records each request for inspection."""

    response: dict[str, object]
    calls: list[tuple[str, dict[str, object], float]] = field(default_factory=list)

    async def post(self, url: str, body: dict[str, object], timeout: float) -> dict[str, object]:
        self.calls.append((url, body, timeout))
        return self.response


def _ollama_response(content: object) -> dict[str, object]:
    """Wrap `content` (JSON-encoded by the model) in an Ollama-shaped envelope."""
    serialised = content if isinstance(content, str) else json.dumps(content)
    return {'message': {'role': 'assistant', 'content': serialised}}


def _ctx(user_input: str = 'hello', **payload: object) -> ModelContext:
    turn = Turn(turn_id='t1', user_input=user_input, timestamp=0.0)
    return ModelContext(turn=turn, payload=dict(payload))


def _user_message(transport: _FakeTransport) -> str:
    """Pull the user message content out of the last recorded request body."""
    body = transport.calls[-1][1]
    messages = body['messages']
    assert isinstance(messages, list)
    user = messages[1]
    assert isinstance(user, dict)
    content = user['content']
    assert isinstance(content, str)
    return content


def _client(response: dict[str, object]) -> tuple[OllamaModelClient, _FakeTransport]:
    transport = _FakeTransport(response=response)
    client = OllamaModelClient(
        host='http://test-host:11434',
        model='qwen3:8b',
        timeout_seconds=5.0,
        transport=transport,
    )
    return client, transport


# --- Output parsing: replies -------------------------------------------------


async def test_parses_direct_reply() -> None:
    client, _ = _client(_ollama_response({'type': 'reply', 'text': 'hi there'}))
    result = await client.generate(_ctx())
    assert result == ModelReply(text='hi there')


async def test_reply_missing_text_field_raises() -> None:
    client, _ = _client(_ollama_response({'type': 'reply'}))
    with pytest.raises(ModelProtocolError, match=r'reply\.text'):
        await client.generate(_ctx())


async def test_reply_non_string_text_raises() -> None:
    client, _ = _client(_ollama_response({'type': 'reply', 'text': 42}))
    with pytest.raises(ModelProtocolError, match=r'reply\.text'):
        await client.generate(_ctx())


# --- Output parsing: plans ---------------------------------------------------


async def test_parses_single_step_plan() -> None:
    plan_json = {
        'type': 'plan',
        'steps': [
            {
                'step': 1,
                'plugin': 'notes',
                'capability': 'create',
                'inputs': {'title': 'shopping'},
                'gate': 'none',
                'outputs_as': 'note',
            },
        ],
    }
    client, _ = _client(_ollama_response(plan_json))
    result = await client.generate(_ctx())
    assert isinstance(result, TaskPlan)
    assert len(result.steps) == 1
    step = result.steps[0]
    assert step.step == 1
    assert step.plugin == 'notes'
    assert step.capability == 'create'
    assert step.inputs == {'title': 'shopping'}
    assert step.gate == 'none'
    assert step.outputs_as == 'note'


async def test_parses_multi_step_plan_preserves_order() -> None:
    plan_json = {
        'type': 'plan',
        'steps': [
            {'step': 1, 'plugin': 'a', 'capability': 'x', 'inputs': {}, 'gate': 'none', 'outputs_as': 'first'},
            {'step': 2, 'plugin': 'b', 'capability': 'y', 'inputs': {'ref': '$first.id'}, 'gate': 'confirm'},
        ],
    }
    client, _ = _client(_ollama_response(plan_json))
    plan = await client.generate(_ctx())
    assert isinstance(plan, TaskPlan)
    assert [s.plugin for s in plan.steps] == ['a', 'b']
    assert plan.steps[1].outputs_as is None
    assert plan.steps[1].gate == 'confirm'


async def test_plan_inputs_default_to_empty_dict_when_omitted() -> None:
    plan_json = {
        'type': 'plan',
        'steps': [{'step': 1, 'plugin': 'p', 'capability': 'c', 'gate': 'none'}],
    }
    client, _ = _client(_ollama_response(plan_json))
    plan = await client.generate(_ctx())
    assert isinstance(plan, TaskPlan)
    assert plan.steps[0].inputs == {}


async def test_plan_gate_defaults_to_none_when_omitted() -> None:
    plan_json = {
        'type': 'plan',
        'steps': [{'step': 1, 'plugin': 'p', 'capability': 'c'}],
    }
    client, _ = _client(_ollama_response(plan_json))
    plan = await client.generate(_ctx())
    assert isinstance(plan, TaskPlan)
    assert plan.steps[0].gate == 'none'


async def test_plan_with_no_steps_rejected() -> None:
    client, _ = _client(_ollama_response({'type': 'plan', 'steps': []}))
    with pytest.raises(ModelProtocolError, match='non-empty array'):
        await client.generate(_ctx())


async def test_plan_steps_not_a_list_rejected() -> None:
    client, _ = _client(_ollama_response({'type': 'plan', 'steps': 'oops'}))
    with pytest.raises(ModelProtocolError, match=r'plan\.steps'):
        await client.generate(_ctx())


async def test_plan_step_must_match_position() -> None:
    plan_json = {
        'type': 'plan',
        'steps': [{'step': 2, 'plugin': 'p', 'capability': 'c'}],
    }
    client, _ = _client(_ollama_response(plan_json))
    with pytest.raises(ModelProtocolError, match='position is 1'):
        await client.generate(_ctx())


async def test_plan_rejects_invalid_gate() -> None:
    plan_json = {
        'type': 'plan',
        'steps': [{'step': 1, 'plugin': 'p', 'capability': 'c', 'gate': 'maybe'}],
    }
    client, _ = _client(_ollama_response(plan_json))
    with pytest.raises(ModelProtocolError, match='gate must be one of'):
        await client.generate(_ctx())


async def test_plan_rejects_non_string_outputs_as() -> None:
    plan_json = {
        'type': 'plan',
        'steps': [{'step': 1, 'plugin': 'p', 'capability': 'c', 'outputs_as': 7}],
    }
    client, _ = _client(_ollama_response(plan_json))
    with pytest.raises(ModelProtocolError, match='outputs_as'):
        await client.generate(_ctx())


async def test_plan_rejects_empty_plugin_name() -> None:
    plan_json = {
        'type': 'plan',
        'steps': [{'step': 1, 'plugin': '', 'capability': 'c'}],
    }
    client, _ = _client(_ollama_response(plan_json))
    with pytest.raises(ModelProtocolError, match='plugin must be a non-empty string'):
        await client.generate(_ctx())


async def test_plan_rejects_bool_as_step_number() -> None:
    # Python treats `True` as an instance of `int` — guard against it
    # so a JSON `true` value in the step field doesn't sneak through.
    plan_json = {
        'type': 'plan',
        'steps': [{'step': True, 'plugin': 'p', 'capability': 'c'}],
    }
    client, _ = _client(_ollama_response(plan_json))
    with pytest.raises(ModelProtocolError, match='step must be an integer'):
        await client.generate(_ctx())


# --- Output parsing: top-level discrimination -------------------------------


async def test_plan_rejects_non_string_gate() -> None:
    # `gate not in _VALID_GATES` would raise TypeError for unhashable JSON
    # shapes (dict/list). Adapter must surface this as ModelProtocolError.
    plan_json: dict[str, object] = {
        'type': 'plan',
        'steps': [{'step': 1, 'plugin': 'p', 'capability': 'c', 'gate': {'nested': 'object'}}],
    }
    client, _ = _client(_ollama_response(plan_json))
    with pytest.raises(ModelProtocolError, match='gate must be one of'):
        await client.generate(_ctx())


async def test_unknown_type_rejected() -> None:
    client, _ = _client(_ollama_response({'type': 'mystery'}))
    with pytest.raises(ModelProtocolError, match="'type' must be 'reply', 'plan', or 'discover'"):
        await client.generate(_ctx())


async def test_missing_type_rejected() -> None:
    client, _ = _client(_ollama_response({'text': 'hi'}))
    with pytest.raises(ModelProtocolError, match="'type' must be"):
        await client.generate(_ctx())


async def test_content_not_json_rejected() -> None:
    client, _ = _client(_ollama_response('not json {'))
    with pytest.raises(ModelProtocolError, match='not valid JSON'):
        await client.generate(_ctx())


async def test_content_json_but_not_object_rejected() -> None:
    client, _ = _client(_ollama_response('[1, 2, 3]'))
    with pytest.raises(ModelProtocolError, match='must be a JSON object'):
        await client.generate(_ctx())


async def test_response_missing_message_field_rejected() -> None:
    client, _ = _client({'unrelated': 'oops'})
    with pytest.raises(ModelProtocolError, match="missing 'message'"):
        await client.generate(_ctx())


async def test_response_empty_content_rejected() -> None:
    client, _ = _client({'message': {'content': ''}})
    with pytest.raises(ModelProtocolError, match='non-empty string'):
        await client.generate(_ctx())


# --- Request shape -----------------------------------------------------------


async def test_request_targets_chat_endpoint_with_json_mode() -> None:
    client, transport = _client(_ollama_response({'type': 'reply', 'text': 'ok'}))
    await client.generate(_ctx('what time is it'))
    assert len(transport.calls) == 1
    url, body, timeout = transport.calls[0]
    assert url == 'http://test-host:11434/api/chat'
    assert body['model'] == 'qwen3:8b'
    assert body['format'] == 'json'
    assert body['stream'] is False
    # Defaults to False - chain-of-thought adds 2-3x latency in the
    # two-call planning loop and isn't needed for plan-emitting models.
    assert body['think'] is False
    assert timeout == 5.0


async def test_think_flag_propagates_to_request_body() -> None:
    """Constructor `think=True` must round-trip into the request body.

    Per-deployment opt-in for CoT models that genuinely produce better
    plans with thinking enabled. Default stays False; this verifies the
    override path doesn't silently drop the flag.
    """
    transport = _FakeTransport(response=_ollama_response({'type': 'reply', 'text': 'ok'}))
    client = OllamaModelClient(host='http://h', model='qwen3:8b', timeout_seconds=5.0, think=True, transport=transport)
    await client.generate(_ctx('q'))
    assert transport.calls[0][1]['think'] is True


async def test_request_includes_system_and_user_messages() -> None:
    client, transport = _client(_ollama_response({'type': 'reply', 'text': 'ok'}))
    await client.generate(_ctx('plan a trip'))
    messages = transport.calls[0][1]['messages']
    assert isinstance(messages, list)
    assert len(messages) == 2
    assert messages[0] == {'role': 'system', 'content': messages[0]['content']}
    assert messages[1]['role'] == 'user'
    # The raw user input must appear verbatim in the user message.
    assert 'plan a trip' in _user_message(transport)


async def test_default_transport_wraps_malformed_url_as_protocol_error() -> None:
    # Default _UrllibTransport hits urllib.request.urlopen, which raises
    # ValueError (not URLError) for unsupported / malformed URLs. The
    # adapter promises uniform ModelProtocolError surfacing.
    client = OllamaModelClient(host='not-a-url', timeout_seconds=1.0)
    with pytest.raises(ModelProtocolError, match='transport error'):
        await client.generate(_ctx())


async def test_host_trailing_slash_normalised() -> None:
    transport = _FakeTransport(response=_ollama_response({'type': 'reply', 'text': 'ok'}))
    client = OllamaModelClient(host='http://h:11434/', transport=transport)
    await client.generate(_ctx())
    assert transport.calls[0][0] == 'http://h:11434/api/chat'


# --- Prompt rendering -------------------------------------------------------


async def test_prompt_surfaces_capabilities() -> None:
    cap = CapabilityDescriptor(plugin='notes', capability='create', description='Create a note')
    client, transport = _client(_ollama_response({'type': 'reply', 'text': 'ok'}))
    await client.generate(_ctx('remind me', capabilities=(cap,), history=(), memory=()))
    user_msg = _user_message(transport)
    assert 'plugin "notes"' in user_msg
    assert 'capability "create"' in user_msg
    assert 'Create a note' in user_msg
    assert 'remind me' in user_msg


async def test_prompt_renders_required_inputs_next_to_capability() -> None:
    """Capabilities with `required_inputs` render `(requires: a, b)` inline.

    Discovered during user-test on 2026-05-13: the model produced plans
    that omitted required inputs (e.g. `tz` for `clock.now`) because the
    prompt only showed descriptions. Surfacing the required-key list
    inline is the minimum the model needs to produce a valid plan.
    """
    cap = CapabilityDescriptor(
        plugin='clock',
        capability='now',
        description='Return the current wall-clock time.',
        required_inputs=('tz',),
    )
    client, transport = _client(_ollama_response({'type': 'reply', 'text': 'ok'}))
    await client.generate(_ctx('time?', capabilities=(cap,), history=(), memory=()))
    user_msg = _user_message(transport)
    assert '- plugin "clock" capability "now" (requires: tz): Return the current wall-clock time.' in user_msg


async def test_prompt_omits_requires_clause_for_empty_required_inputs() -> None:
    """No-input capabilities render without a `(requires: ...)` parenthetical."""
    cap = CapabilityDescriptor(plugin='notes', capability='list', description='List all notes.')
    client, transport = _client(_ollama_response({'type': 'reply', 'text': 'ok'}))
    await client.generate(_ctx('show notes', capabilities=(cap,), history=(), memory=()))
    user_msg = _user_message(transport)
    assert '- plugin "notes" capability "list": List all notes.' in user_msg
    assert 'requires' not in user_msg


async def test_prompt_surfaces_history_and_memory() -> None:
    entry = ConversationEntry(turn_id='prev', user_input='hi', assistant_reply='hello', timestamp=0.0)
    snip = MemorySnippet(source='memory-mcp', content='user likes lists')
    client, transport = _client(_ollama_response({'type': 'reply', 'text': 'ok'}))
    await client.generate(_ctx('next', capabilities=(), history=(entry,), memory=(snip,)))
    user_msg = _user_message(transport)
    assert 'user: hi' in user_msg
    assert 'assistant: hello' in user_msg
    assert 'user likes lists' in user_msg
    assert '[memory-mcp]' in user_msg


async def test_prompt_omits_empty_history_and_memory_sections() -> None:
    """History and memory headers are suppressed when empty.

    Capabilities is the exception — see
    `test_prompt_renders_empty_capabilities_explicitly` — because a missing
    section let the model confabulate plausible plugins.
    """
    client, transport = _client(_ollama_response({'type': 'reply', 'text': 'ok'}))
    await client.generate(_ctx('only input', capabilities=(), history=(), memory=()))
    user_msg = _user_message(transport)
    assert 'Recent conversation' not in user_msg
    assert 'Relevant memory' not in user_msg
    assert 'only input' in user_msg


async def test_prompt_renders_empty_capabilities_explicitly() -> None:
    """Empty registry must surface the `Available capabilities:` header
    followed by `(none)` on the next line.

    Discovered during user-test on 2026-05-13: with the header omitted,
    the model invented capabilities ("file system access", "web search")
    when asked what it could do, because it had no signal the absence was
    intentional. Rendering `(none)` forces honesty.
    """
    client, transport = _client(_ollama_response({'type': 'reply', 'text': 'ok'}))
    await client.generate(_ctx('what can you do?', capabilities=()))
    user_msg = _user_message(transport)
    assert 'Available capabilities:' in user_msg
    assert '(none)' in user_msg


async def test_payload_with_wrong_type_raises_protocol_error() -> None:
    # If somebody wires up a context manager that produces the wrong
    # value shapes, surface a model-protocol-level error rather than
    # silently mangling the prompt.
    client, _ = _client(_ollama_response({'type': 'reply', 'text': 'ok'}))
    bad_ctx = _ctx('hi', capabilities=['not a descriptor'])
    with pytest.raises(ModelProtocolError, match="'capabilities'"):
        await client.generate(bad_ctx)


# --- System prompt content (PR C: identity + capability honesty) ------------


def _system_prompt(transport: _FakeTransport) -> str:
    """Return the system message content with newlines folded to spaces.

    The prompt is hard-wrapped for readability, but the assertions below
    care about logical phrases, not column-80 layout — folding whitespace
    keeps the tests stable under future reflows.
    """
    body = transport.calls[-1][1]
    messages = body['messages']
    assert isinstance(messages, list)
    system = messages[0]
    assert isinstance(system, dict)
    content = system['content']
    assert isinstance(content, str)
    return ' '.join(content.split())


async def test_system_prompt_identifies_butter_agent() -> None:
    client, transport = _client(_ollama_response({'type': 'reply', 'text': 'ok'}))
    await client.generate(_ctx())
    folded = _system_prompt(transport)
    assert 'butter-agent' in folded
    assert 'local-first' in folded
    assert 'personal assistant' in folded


async def test_system_prompt_forbids_fabricating_plans_without_capability() -> None:
    """The no-plan-without-capability rule must be present verbatim enough
    that future edits to the prompt cannot quietly drop it — empty-registry
    installs depend on this rule to keep the model honest about what it
    can and cannot do.
    """
    client, transport = _client(_ollama_response({'type': 'reply', 'text': 'ok'}))
    await client.generate(_ctx(capabilities=()))
    folded = _system_prompt(transport)
    assert 'no listed capability matches' in folded
    assert 'Do not invent' in folded
    assert 'fabricate a plan' in folded


async def test_system_prompt_requires_plan_when_capability_matches() -> None:
    """When a matching capability exists, the model must plan — not chat.

    User-test on 2026-05-14 with hermes3:8b: asked "Can you give me the
    current time?" with `clock.now` registered, the model fabricated a
    time in a `ModelReply` instead of emitting a plan. The earlier
    qwen3 failure mode was the same class — text reply containing the
    literal placeholder `"$(clock.now)"` instead of a plan. Both bypass
    the plan path because the original prompt framed plans as optional.
    """
    client, transport = _client(_ollama_response({'type': 'reply', 'text': 'ok'}))
    await client.generate(_ctx())
    folded = _system_prompt(transport)
    assert 'MUST return a plan' in folded
    assert 'Do not fabricate values' in folded
    # Worked example anchors the rule with a concrete time-question→plan
    # pair. Without it, 8B-class models read the rule as advisory and
    # default to chat. Don't drop the example without a replacement.
    assert 'Worked example' in folded
    assert '"type":"plan"' in folded


async def test_system_prompt_states_capability_list_is_exhaustive() -> None:
    """The system prompt must tell the model the rendered list is exhaustive.

    Without this, an empty-registry install lets the model riff on a
    plausible plugin universe when asked "what can you do?". The rule
    explicitly forbids speculation about files, the web, calendars, or
    email unless a matching capability is listed.
    """
    client, transport = _client(_ollama_response({'type': 'reply', 'text': 'ok'}))
    await client.generate(_ctx())
    folded = _system_prompt(transport)
    assert 'complete and exhaustive' in folded
    assert 'no plugins installed' in folded


# --- Synthesis-mode prompting -----------------------------------------------


def _execution_result(outputs: dict[str, dict[str, object]] | None = None) -> ExecutionResult:
    plan = TaskPlan(
        steps=(PlanStep(step=1, plugin='clock', capability='now', inputs={}, gate='none', outputs_as='t'),),
    )
    return ExecutionResult(plan=plan, outputs=outputs if outputs is not None else {'t': {'time': '17:00', 'tz': 'CEST'}})


async def test_synthesis_uses_reply_only_system_prompt() -> None:
    # Presence of `execution` in the payload swaps the system prompt to the
    # synthesis variant, which forbids returning a plan.
    client, transport = _client(_ollama_response({'type': 'reply', 'text': 'It is 17:00 CEST.'}))
    await client.generate(_ctx('what time is it', execution=_execution_result()))
    folded = _system_prompt(transport)
    assert 'plugin task plan' in folded
    assert 'plan has already run' in folded
    assert 'Do not return "type": "plan"' in folded


async def test_synthesis_renders_tool_results_section() -> None:
    client, transport = _client(_ollama_response({'type': 'reply', 'text': 'ok'}))
    await client.generate(_ctx('what time is it', execution=_execution_result()))
    user_msg = _user_message(transport)
    assert 'Tool results:' in user_msg
    assert 'step 1: clock.now → $t' in user_msg
    # Output fields are surfaced with their values so the model can quote them.
    assert "time: '17:00'" in user_msg
    assert "tz: 'CEST'" in user_msg


async def test_synthesis_marks_failed_step_and_following_steps() -> None:
    """Failed step is rendered FAILED; subsequent steps as 'did not run'.

    Spec: `plugin-failure-recovery.md`. Without explicit markers, the
    model would either invent successful outputs for the failed step
    or treat trailing steps as if they ran.
    """
    plan = TaskPlan(
        steps=(
            PlanStep(step=1, plugin='clock', capability='now', inputs={}, gate='none', outputs_as='t'),
            PlanStep(step=2, plugin='clock', capability='diff', inputs={'a': '$t.time', 'b': 'x'}, gate='none', outputs_as='d'),
            PlanStep(step=3, plugin='clock', capability='now', inputs={}, gate='none', outputs_as='t2'),
        ),
    )
    failed = ExecutionResult(
        plan=plan,
        outputs={'t': {'time': '17:00', 'tz': 'CEST'}},
        failed_at_step=2,
        failure_reason="plugin 'clock' capability 'diff' raised: bad iso",
    )
    client, transport = _client(_ollama_response({'type': 'reply', 'text': 'ok'}))
    await client.generate(_ctx('diff', execution=failed))
    user_msg = _user_message(transport)
    # Step 1 ran normally with its outputs surfaced.
    assert 'step 1: clock.now → $t' in user_msg
    assert "time: '17:00'" in user_msg
    # Step 2 carries the FAILED marker and the reason.
    assert 'step 2: clock.diff → $d — FAILED' in user_msg
    assert 'raised: bad iso' in user_msg
    # Step 3 explicitly did not run — no fabricated outputs.
    assert 'step 3: clock.now → $t2 — did not run (prior step failed)' in user_msg


async def test_synthesis_system_prompt_requires_verbatim_value_quoting() -> None:
    """Synthesis must preserve tool-output values verbatim when quoting them.

    Live-REPL on 2026-05-14 showed qwen3:8b paraphrasing
    `'2026-05-14T15:12:18+02:00'` into `'2026-05-14T15:12:18'` (offset
    stripped) in its reply. That reply then surfaced in the next turn's
    history; the model copied the stripped value into a new
    `clock.diff` plan and the diff returned wall-clock skew instead of
    absolute-time distance. See
    `specs/development/plugin-failure-recovery.md` ("Known limitations").
    Pin the prompt clause so a future edit cannot quietly drop it.
    """
    client, transport = _client(_ollama_response({'type': 'reply', 'text': 'ok'}))
    await client.generate(_ctx('q', execution=_execution_result()))
    folded = _system_prompt(transport)
    assert 'reproduce it verbatim' in folded
    assert 'timezone offsets' in folded


async def test_synthesis_system_prompt_acknowledges_failed_steps() -> None:
    """Synthesis prompt instructs the model to acknowledge FAILED steps."""
    client, transport = _client(_ollama_response({'type': 'reply', 'text': 'ok'}))
    await client.generate(_ctx('q', execution=_execution_result()))
    folded = _system_prompt(transport)
    assert 'FAILED' in folded
    assert 'acknowledge the failure' in folded
    assert 'Do not invent successful outputs' in folded


async def test_synthesis_omits_capabilities_section() -> None:
    # Capabilities are the menu for planning; in synthesis the plan already
    # ran, so the menu is irrelevant and could nudge the model toward
    # proposing follow-up actions instead of replying.
    client, transport = _client(_ollama_response({'type': 'reply', 'text': 'ok'}))
    await client.generate(_ctx('q', execution=_execution_result()))
    user_msg = _user_message(transport)
    assert 'Available capabilities:' not in user_msg


async def test_synthesis_rejects_non_execution_value() -> None:
    # Defense against a context manager that puts the wrong thing under
    # the 'execution' key — surface a protocol error instead of letting
    # the prompt render garbage.
    client, _ = _client(_ollama_response({'type': 'reply', 'text': 'ok'}))
    bad = _ctx('q', execution='not-an-execution')
    with pytest.raises(ModelProtocolError, match="'execution'"):
        await client.generate(bad)


# --- Capability discovery (Tier-1) ------------------------------------------


async def test_discovery_pass_uses_discovery_system_prompt() -> None:
    # A `plugin_index` payload key swaps in the discovery system prompt:
    # pick plugins, do not plan yet.
    client, transport = _client(_ollama_response({'type': 'discover', 'plugins': ['notes']}))
    await client.generate(_ctx('save a note', plugin_index=(PluginIndexEntry(name='notes', summary='Notes.'),)))
    folded = _system_prompt(transport)
    assert 'choose which plugins you need' in folded
    assert 'NOT been shown individual capabilities' in folded
    assert 'Do NOT return "type": "plan"' in folded


async def test_discovery_prompt_renders_plugin_index() -> None:
    client, transport = _client(_ollama_response({'type': 'discover', 'plugins': ['files']}))
    index = (
        PluginIndexEntry(name='notes', summary='Short text notes.'),
        PluginIndexEntry(name='files', summary='Read and write local files.'),
    )
    await client.generate(_ctx('what deps does pyproject have', plugin_index=index))
    user_msg = _user_message(transport)
    assert 'Available plugins:' in user_msg
    assert '- "notes": Short text notes.' in user_msg
    assert '- "files": Read and write local files.' in user_msg
    assert 'Available capabilities:' not in user_msg


async def test_discovery_prompt_renders_empty_index_explicitly() -> None:
    # An empty index must surface "(none)" so the model admits it has no
    # plugins rather than confabulating one — same honesty guard the
    # capability list uses.
    client, transport = _client(_ollama_response({'type': 'reply', 'text': 'I have no plugins.'}))
    await client.generate(_ctx('do something', plugin_index=()))
    user_msg = _user_message(transport)
    assert 'Available plugins:' in user_msg
    assert '(none)' in user_msg


async def test_parses_discovery_selection() -> None:
    client, _ = _client(_ollama_response({'type': 'discover', 'plugins': ['files', 'clock']}))
    result = await client.generate(_ctx('q', plugin_index=()))
    assert result == DiscoverySelection(plugins=('files', 'clock'))


async def test_parses_empty_discovery_selection() -> None:
    # Structurally valid — the context manager treats an empty selection
    # as the keyword-filter fallback, so this is not an adapter error.
    client, _ = _client(_ollama_response({'type': 'discover', 'plugins': []}))
    result = await client.generate(_ctx('q', plugin_index=()))
    assert result == DiscoverySelection(plugins=())


async def test_discovery_non_list_plugins_raises() -> None:
    client, _ = _client(_ollama_response({'type': 'discover', 'plugins': 'files'}))
    with pytest.raises(ModelProtocolError, match=r'discover\.plugins must be an array'):
        await client.generate(_ctx('q', plugin_index=()))


async def test_discovery_non_string_plugin_entry_raises() -> None:
    client, _ = _client(_ollama_response({'type': 'discover', 'plugins': ['ok', 42]}))
    with pytest.raises(ModelProtocolError, match=r'discover\.plugins\[1\]'):
        await client.generate(_ctx('q', plugin_index=()))


async def test_discovery_reply_still_parses() -> None:
    # The discovery pass may also yield a plain reply (conversational turn).
    client, _ = _client(_ollama_response({'type': 'reply', 'text': 'hello'}))
    result = await client.generate(_ctx('hi', plugin_index=()))
    assert result == ModelReply(text='hello')


# --- Transport error wrapping (timeouts) ------------------------------------


async def test_transport_wraps_timeout_as_protocol_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """TimeoutError from `urlopen(timeout=...)` must surface as ModelProtocolError.

    Surfaced during the 2026-05-13 user-test: timeouts were escaping as
    raw OSError because the original except clause only caught URLError /
    ValueError. cli._cmd_start then treated them as fatal startup errors
    and exited the REPL, instead of letting the loop print a model-error
    line and accept another prompt.

    Patched at the urlopen seam rather than relying on a real socket —
    ECONNREFUSED on an unbound port surfaces as URLError, not
    TimeoutError, so a "talk to port 1" approach would not exercise
    this branch deterministically.
    """
    import urllib.request

    from butter_agent.model.ollama import _UrllibTransport

    def _raise_timeout(*args: object, **kwargs: object) -> object:
        raise TimeoutError('socket timed out')

    monkeypatch.setattr(urllib.request, 'urlopen', _raise_timeout)
    transport = _UrllibTransport()
    # 12.5s must render as "12.5s" (not "12s" — `:.0f` would round it) and
    # the diagnostic must NOT hardcode a specific config filename because
    # `--config` may point somewhere other than the XDG default.
    with pytest.raises(ModelProtocolError, match=r'timed out after 12\.5s') as info:
        await transport.post('http://example.invalid/api/chat', {'model': 'x', 'messages': []}, timeout=12.5)
    assert 'config.toml' not in str(info.value)
    assert 'timeout_seconds' in str(info.value)
