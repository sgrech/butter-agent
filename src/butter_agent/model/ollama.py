"""Ollama `ModelClient` adapter — local-first default per scope.

Implements the `ModelClient` Protocol from `core/loop.py` against
Ollama's `/api/chat` endpoint in JSON mode. The adapter is responsible
for two things the rest of `core/` deliberately stays out of:

- **Prompt construction.** Takes `ModelContext.payload` (capabilities,
  history, memory — keys established by `DefaultContextManager`) and
  renders the chat-style messages handed to Ollama. The loop never
  inspects payload contents, so each model adapter owns its own
  prompt-assembly conventions.
- **Output discrimination.** Parses the JSON the model returns into
  exactly one of `ModelReply` or `TaskPlan`. Any structural problem —
  malformed JSON, missing discriminator, wrong types, non-sequential
  step numbers — raises `ModelProtocolError` so the REPL surfaces a
  clean diagnostic and does not guess.

Validation is **structural only**. Semantic checks (does the named
plugin exist? do declared inputs match the registered schema?) are the
task executor's job and are run after the loop hands the plan off.

Transport is behind a `Transport` Protocol so tests inject canned
responses without spinning up Ollama. The default `_UrllibTransport`
uses stdlib `urllib.request` wrapped in `asyncio.to_thread` — no
runtime dependencies beyond Python itself, per the scope's
"dependency minimalism" constraint.
"""

from __future__ import annotations

import asyncio
import json
import urllib.error
import urllib.request
from typing import Final, Protocol

from butter_agent.core.context_manager import (
    CapabilityDescriptor,
    ConversationEntry,
    MemorySnippet,
)
from butter_agent.core.loop import (
    ExecutionResult,
    ModelContext,
    ModelOutput,
    ModelProtocolError,
    ModelReply,
    PlanStep,
    TaskPlan,
)

# --- Defaults ----------------------------------------------------------------

DEFAULT_HOST: Final = 'http://localhost:11434'
DEFAULT_MODEL: Final = 'qwen3:8b'
DEFAULT_TIMEOUT_SECONDS: Final = 60.0

_VALID_GATES: Final[frozenset[str]] = frozenset({'none', 'confirm', 'human'})

# --- Transport seam ----------------------------------------------------------


class Transport(Protocol):
    """HTTP transport seam — abstracted so tests can stub it cleanly.

    Implementations POST a JSON `body` to `url` with the given `timeout`
    (seconds) and return the parsed JSON response as a dict. They MUST
    raise `ModelProtocolError` on any transport-level failure (network
    error, non-2xx status, non-JSON response body) so the loop sees a
    single uniform error type.
    """

    async def post(self, url: str, body: dict[str, object], timeout: float) -> dict[str, object]: ...


class _UrllibTransport:
    """Default `Transport` backed by stdlib `urllib.request`.

    Sync I/O wrapped in `asyncio.to_thread` so the async caller is not
    blocked. The REPL is single-task while a turn is in flight, so this
    is acceptable for v1 — a future Telegram adapter that wants true
    concurrency can supply an aiohttp-backed transport without touching
    this module.
    """

    async def post(self, url: str, body: dict[str, object], timeout: float) -> dict[str, object]:
        return await asyncio.to_thread(self._post_sync, url, body, timeout)

    def _post_sync(self, url: str, body: dict[str, object], timeout: float) -> dict[str, object]:
        try:
            request = urllib.request.Request(
                url,
                data=json.dumps(body).encode('utf-8'),
                headers={'Content-Type': 'application/json'},
                method='POST',
            )
            with urllib.request.urlopen(request, timeout=timeout) as response:
                raw = response.read()
        except TimeoutError as exc:
            # `urlopen(timeout=...)` raises socket.timeout (aliased to
            # TimeoutError in Python 3.10+), which is an OSError subclass and
            # NOT a URLError. Without this branch it would escape the
            # adapter's "uniform ModelProtocolError" promise and bubble up to
            # cli._cmd_start as a fatal startup error, killing the REPL
            # instead of letting the loop print the error and continue.
            #
            # `:g` formats 12 as "12" and 12.5 as "12.5" — `:.0f` rounds and
            # would misreport non-integer timeouts. The diagnostic doesn't
            # name a specific config file because `--config` may point
            # somewhere other than the XDG default.
            raise ModelProtocolError(
                f'ollama request timed out after {timeout:g}s — the model may be cold-loading; increase [model] timeout_seconds in your config if this is the first call',
            ) from exc
        except (urllib.error.URLError, ValueError) as exc:
            # Both Request(url=...) and urlopen() can raise ValueError for
            # unsupported / malformed URLs (e.g. missing scheme). Treat
            # alongside URLError so the adapter's promise of uniform
            # ModelProtocolError surfacing holds.
            raise ModelProtocolError(f'ollama transport error: {exc}') from exc
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ModelProtocolError(f'ollama returned non-JSON body: {exc}') from exc
        if not isinstance(parsed, dict):
            raise ModelProtocolError(f'ollama response must be a JSON object, got {type(parsed).__name__}')
        return parsed


# --- The adapter -------------------------------------------------------------


class OllamaModelClient:
    """`ModelClient` for Ollama's `/api/chat` endpoint in JSON mode."""

    def __init__(
        self,
        *,
        host: str = DEFAULT_HOST,
        model: str = DEFAULT_MODEL,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        think: bool = False,
        transport: Transport | None = None,
    ) -> None:
        self._host = host.rstrip('/')
        self._model = model
        self._timeout = timeout_seconds
        self._think = think
        self._transport: Transport = transport if transport is not None else _UrllibTransport()

    async def generate(self, context: ModelContext) -> ModelOutput:
        # Presence of an `execution` payload entry flips the adapter into
        # synthesis mode: different system prompt (reply-only) and the
        # rendered user prompt includes the tool results.
        synthesis = context.payload.get('execution') is not None
        system_prompt = _SYNTHESIS_SYSTEM_PROMPT if synthesis else _SYSTEM_PROMPT
        body: dict[str, object] = {
            'model': self._model,
            'messages': [
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': _render_user_prompt(context)},
            ],
            'format': 'json',
            'stream': False,
            # Ollama field for qwen3 / deepseek-r1 / other CoT models.
            # False = no <think>...</think> emission, ~2-3x faster on
            # qwen3:8b per 2026-05-14 testing. Older Ollama builds and
            # non-CoT models ignore the field.
            'think': self._think,
        }
        response = await self._transport.post(f'{self._host}/api/chat', body, self._timeout)
        content = _extract_content(response)
        return _parse_output(content)


# --- Prompt construction -----------------------------------------------------

_SYSTEM_PROMPT = """\
You are butter-agent, a local-first personal assistant. Your primary
job is to invoke the plugins listed in the "Available capabilities"
section of the user message — chat is the fallback, not the default.

Decision rule (apply in this order):

1. If any listed capability could produce the answer the user is
   asking for, you MUST return a plan that invokes it. Do not answer
   from your own knowledge. Do not fabricate values (times, dates,
   prices, locations, file contents, search results) that a listed
   capability could produce. You do not know the current time, the
   weather, or any other live value — the plugins do.
2. If no listed capability matches, return a "reply" explaining what
   is missing. Do not invent a plugin name or fabricate a plan.
3. If the request is purely conversational (greeting, opinion,
   clarification), return a "reply".

The "Available capabilities" section is the complete and exhaustive
list of plugin actions available to you. If it is empty (shown as
"(none)"), you have no plugins installed. Never list, describe, imply,
or speculate about capabilities beyond what is shown there, and never
claim to access files, the web, calendars, email, or any other external
system unless a matching capability is listed.

Output schemas — your response must be exactly one of these:

Reply:
  {"type": "reply", "text": "..."}

Plan:
  {"type": "plan", "steps": [
    {
      "step": 1,
      "plugin": "<plugin name>",
      "capability": "<capability name>",
      "inputs": {"<key>": <value>},
      "gate": "none" | "confirm" | "human",
      "outputs_as": "<alias>" | null
    }
  ]}

Step numbers start at 1 and increase by 1. Reference a prior step's
output with the string "$alias.field" — only aliases declared by an
earlier step's "outputs_as" are valid. Use "gate": "confirm" for any
step the user should approve before it runs; use "gate": "human" when
the user should review prior outputs first.

Worked example. If the available capabilities include:
  - plugin "clock" capability "now": Current date and time...
and the user asks "What time is it?", the correct response is:
  {"type":"plan","steps":[{"step":1,"plugin":"clock","capability":"now","inputs":{},"gate":"none","outputs_as":"t"}]}
NOT a reply that states a time. You have no way to know the time
without invoking the plugin.

Return JSON only. No prose outside the object.
"""


_SYNTHESIS_SYSTEM_PROMPT = """\
You are butter-agent. The system has just executed a plugin task plan on
the user's behalf in response to their request. The "Tool results"
section in the user message contains the steps that ran and the outputs
each step produced.

Produce a JSON object: {"type": "reply", "text": "..."} where text is a
short, natural-language answer to the user's original request, grounded
in the tool results. Quote concrete values from the outputs where they
help; convert units or timezones if the user asked for that.

When you quote a value that came directly from a tool output (a
timestamp, an ID, a filename, a numeric measurement), reproduce it
verbatim — including timezone offsets like `+02:00`, fractional
seconds, and any other suffix the tool emitted. The user's next turn
may reuse these values in a follow-up plan, and a stripped or
reformatted value will round-trip incorrectly. If you want to express
the same value in a friendlier form, quote the original first, then
add the rephrasing in parentheses.

If a step is marked "FAILED" in the tool results, acknowledge the
failure plainly in your reply — say which step failed and why. Do not
invent successful outputs for it. If earlier steps in the plan
succeeded, you may still use their values; if the failure left the
user's question unanswered, say so and suggest a next step.

Do not return "type": "plan". Do not propose new plugin actions. The
plan has already run. Return JSON only. No prose outside the object.
"""


def _render_user_prompt(context: ModelContext) -> str:
    """Render the user-facing prompt from the assembled `ModelContext`.

    Keys consumed (`capabilities`, `history`, `memory`) are the contract
    established by `DefaultContextManager` — if a different context
    manager populates a different shape, it pairs with a different
    `ModelClient`.
    """
    payload = context.payload
    parts: list[str] = []
    execution = payload.get('execution')

    if execution is None:
        # Intent-recognition pass. Capabilities are always rendered — even
        # when empty — so the model sees the absence rather than inferring
        # it. A missing section let the model confabulate plausible plugins
        # ("file system access", "web search") when asked what it could do;
        # rendering "(none)" forces honesty.
        capabilities = _expect_tuple(payload.get('capabilities', ()), CapabilityDescriptor, 'capabilities')
        parts.append('Available capabilities:')
        if capabilities:
            parts.extend(_render_capability(c) for c in capabilities)
        else:
            parts.append('(none)')
        parts.append('')

    history = _expect_tuple(payload.get('history', ()), ConversationEntry, 'history')
    if history:
        parts.append('Recent conversation:')
        for entry in history:
            parts.append(f'user: {entry.user_input}')
            if entry.assistant_reply is not None:
                parts.append(f'assistant: {entry.assistant_reply}')
        parts.append('')

    memory = _expect_tuple(payload.get('memory', ()), MemorySnippet, 'memory')
    if memory:
        parts.append('Relevant memory:')
        parts.extend(f'- [{m.source}] {m.content}' for m in memory)
        parts.append('')

    if execution is not None:
        if not isinstance(execution, ExecutionResult):
            raise ModelProtocolError(
                f"context payload 'execution' must be an ExecutionResult, got {type(execution).__name__}",
            )
        parts.append('Tool results:')
        for step in execution.plan.steps:
            alias = f' → ${step.outputs_as}' if step.outputs_as else ''
            if execution.failed_at_step is not None and step.step == execution.failed_at_step:
                parts.append(f'- step {step.step}: {step.plugin}.{step.capability}{alias} — FAILED: {execution.failure_reason}')
            elif execution.failed_at_step is not None and step.step > execution.failed_at_step:
                # Steps after the failure point did not run; surface that
                # explicitly so the model doesn't fabricate their outputs.
                parts.append(f'- step {step.step}: {step.plugin}.{step.capability}{alias} — did not run (prior step failed)')
            else:
                parts.append(f'- step {step.step}: {step.plugin}.{step.capability}{alias}')
                if step.outputs_as is not None and step.outputs_as in execution.outputs:
                    fields = execution.outputs[step.outputs_as]
                    for key, value in fields.items():
                        parts.append(f'    {key}: {value!r}')
        parts.append('')

    parts.append(f'User: {context.turn.user_input}')
    return '\n'.join(parts)


def _render_capability(cap: CapabilityDescriptor) -> str:
    # Plugin and capability are rendered with the same field names the JSON
    # plan schema uses (`plugin`, `capability`). User-test on 2026-05-14
    # with mistral-nemo:12b: the previous `- clock.now: ...` shorthand
    # caused the model to put the dotted form into the `plugin` field
    # ("plugin": "clock.now") instead of splitting at the dot. Naming the
    # fields explicitly removes the ambiguity.
    head = f'- plugin "{cap.plugin}" capability "{cap.capability}"'
    if cap.required_inputs:
        required = ', '.join(cap.required_inputs)
        return f'{head} (requires: {required}): {cap.description}'
    return f'{head}: {cap.description}'


def _expect_tuple[T](value: object, item_type: type[T], label: str) -> tuple[T, ...]:
    if isinstance(value, tuple) and all(isinstance(item, item_type) for item in value):
        return value
    raise ModelProtocolError(
        f'context payload {label!r} must be a tuple of {item_type.__name__}, got {type(value).__name__}',
    )


# --- Response parsing --------------------------------------------------------


def _extract_content(response: dict[str, object]) -> str:
    message = response.get('message')
    if not isinstance(message, dict):
        raise ModelProtocolError("ollama response missing 'message' object")
    content = message.get('content')
    if not isinstance(content, str) or not content:
        raise ModelProtocolError("ollama response 'message.content' must be a non-empty string")
    return content


def _parse_output(content: str) -> ModelOutput:
    try:
        data = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ModelProtocolError(f'model output is not valid JSON: {exc}') from exc
    if not isinstance(data, dict):
        raise ModelProtocolError(f'model output must be a JSON object, got {type(data).__name__}')

    kind = data.get('type')
    if kind == 'reply':
        return _parse_reply(data)
    if kind == 'plan':
        return _parse_plan(data)
    raise ModelProtocolError(
        f"model output 'type' must be 'reply' or 'plan', got {kind!r}",
    )


def _parse_reply(data: dict[str, object]) -> ModelReply:
    text = data.get('text')
    if not isinstance(text, str):
        raise ModelProtocolError('reply.text must be a string')
    return ModelReply(text=text)


def _parse_plan(data: dict[str, object]) -> TaskPlan:
    raw_steps = data.get('steps')
    if not isinstance(raw_steps, list) or not raw_steps:
        raise ModelProtocolError('plan.steps must be a non-empty array')
    steps = tuple(_parse_step(idx, item) for idx, item in enumerate(raw_steps))
    return TaskPlan(steps=steps)


def _parse_step(index: int, raw: object) -> PlanStep:
    position = index + 1
    if not isinstance(raw, dict):
        raise ModelProtocolError(f'plan.steps[{index}] must be an object')

    step_num = raw.get('step')
    if not isinstance(step_num, int) or isinstance(step_num, bool):
        raise ModelProtocolError(f'plan.steps[{index}].step must be an integer')

    plugin = raw.get('plugin')
    if not isinstance(plugin, str) or not plugin:
        raise ModelProtocolError(f'plan.steps[{index}].plugin must be a non-empty string')

    capability = raw.get('capability')
    if not isinstance(capability, str) or not capability:
        raise ModelProtocolError(f'plan.steps[{index}].capability must be a non-empty string')

    inputs = raw.get('inputs', {})
    if not isinstance(inputs, dict):
        raise ModelProtocolError(f'plan.steps[{index}].inputs must be an object')

    gate = raw.get('gate', 'none')
    if not isinstance(gate, str) or gate not in _VALID_GATES:
        # Guard the isinstance check first: `in _VALID_GATES` would raise
        # TypeError for unhashable JSON shapes (dict/list), which would
        # leak past the adapter's structural-error contract.
        valid = ', '.join(sorted(_VALID_GATES))
        raise ModelProtocolError(
            f'plan.steps[{index}].gate must be one of: {valid} (got {gate!r})',
        )

    outputs_as_raw = raw.get('outputs_as')
    if outputs_as_raw is None:
        outputs_as: str | None = None
    elif isinstance(outputs_as_raw, str) and outputs_as_raw:
        outputs_as = outputs_as_raw
    else:
        raise ModelProtocolError(
            f'plan.steps[{index}].outputs_as must be a non-empty string or null',
        )

    # Structural sanity: keep the executor's atomic validation responsible
    # for cross-step checks (alias uniqueness, $ref resolution), but reject
    # an obviously misnumbered step here so the user gets the error from
    # the adapter rather than after the executor walks the plan.
    if step_num != position:
        raise ModelProtocolError(
            f'plan.steps[{index}].step={step_num} but position is {position} (steps must be 1..N in order)',
        )

    return PlanStep(
        step=step_num,
        plugin=plugin,
        capability=capability,
        inputs=dict(inputs),
        gate=gate,
        outputs_as=outputs_as,
    )


__all__: Final[tuple[str, ...]] = (
    'DEFAULT_HOST',
    'DEFAULT_MODEL',
    'DEFAULT_TIMEOUT_SECONDS',
    'OllamaModelClient',
    'Transport',
)
