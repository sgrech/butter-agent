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
)
from butter_agent.core.loop import (
    ModelContext,
    ModelProtocolError,
    ModelReply,
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
    with pytest.raises(ModelProtocolError, match="'type' must be 'reply' or 'plan'"):
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
    assert timeout == 5.0


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
    assert 'notes.create' in user_msg
    assert 'Create a note' in user_msg
    assert 'remind me' in user_msg


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
    assert 'no matching capability' in folded
    assert 'do not invent' in folded
    assert 'fabricate a plan' in folded


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
    assert 'you can only chat' in folded
