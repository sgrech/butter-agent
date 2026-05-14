# Plugin failure recovery + alias persistence across turns

Status: planned
Owner: shanegrech
Created: 2026-05-14
Follows from: `model-baseline-and-input-schema-prompt.md` (user-testing
on 2026-05-14 surfaced the findings recorded here).

## Findings from live-REPL testing on 2026-05-14

1. **Qwen3:8b works end-to-end.** With the `(requires: ...)` prompt fix
   (PR-in-flight on `feat/capability-input-schema-prompt`) and the
   `plugin "x" capability "y"` rendering disambiguation, qwen3:8b
   reliably invokes `clock.now` / `clock.now_in_zone`. Hermes3:8b and
   mistral-nemo:12b both confabulate plausible-looking values
   (mistral's timezone arithmetic was internally inconsistent —
   conclusive evidence). The change-2 plan to swap the default away
   from qwen3 is on hold: the only correct local model at this size is
   the thinking-tag one we tried to leave behind. Cost concerns from
   the synthesis turn doubling stand, but correctness wins over cost
   for v1.
2. **Persisted history poisons the next session.** SQLite-backed
   conversation history replayed prior confabulations as evidence of
   "normal" assistant behaviour. Already addressed on
   `feat/capability-input-schema-prompt`: `SqliteConversationHistory`
   deleted, `InMemoryConversationHistory` is the only wiring in
   `app.py`. Every butter invocation starts a fresh chat session;
   persisted sessions are explicitly future work.
3. **Plugin exceptions tore the REPL down.** A `clock.diff` call with
   bare `$now_iso` strings (model emitted an undocumented reference
   form) raised `ValueError` from inside the plugin. The exception
   propagated through `task_executor.execute` → `loop.run_turn` →
   `repl.run` → `asyncio.run`, killing the process. Partial fix
   already in: `PluginExecutionError(ExecutorError)` wraps the
   exception so the REPL's existing `[error] plan rejected: ...`
   channel catches it. **But the model sees nothing** — the turn aborts
   before synthesis runs and before history is recorded.
4. **The model invented a `$alias` reference form.** On turn 3 (after
   two successful turns each declaring `outputs_as`), qwen3 emitted
   `inputs={'a': '$now_iso', 'b': '$ny_iso'}` — bare aliases, not the
   documented `$alias.field` form. Two distinct causes blended here:
   (a) the spec is `$alias.field` and the model used `$alias`, (b) the
   aliases were declared in *prior* turns, but in-memory history only
   carries the rendered assistant text, not the `outputs_as` bindings —
   so even with the correct syntax, the alias would have been
   undeclared in the new turn's plan.

## Scope

### 1. Synthesis-on-failure (this spec)

Today plugin exceptions short-circuit the loop and the model never
sees the failure. Change the executor's failure-handling contract so
the model can apologise, retry intelligently, or fall back to a
reply.

- `ExecutionResult`: add `failed_at_step: int | None` and
  `failure_reason: str | None`, mirroring the existing
  `halted_at_step` / `halt_reason` pair. An `ExecutionResult` with
  either pair set carries partial outputs (from steps that ran
  *before* the failure point).
- `DefaultTaskExecutor.execute`: catch any `Exception` from
  `registered.plugin.execute(...)` and return an `ExecutionResult` with
  `failed_at_step` / `failure_reason` populated, instead of raising
  `PluginExecutionError`. Subsequent steps do not run — the partial
  outputs are what they are. Delete `PluginExecutionError`; the
  failure is now a value, not an exception.
- `AgentLoop.run_turn`: if `execution.failed_at_step is not None`,
  still run the synthesis turn. The synthesis reply is the model's
  natural-language acknowledgement of the failure (the model can
  apologise, ask the user for clarification, or note what *did* run).
  History records the synthesis reply, so the next turn has full
  context.
- `DefaultContextManager.assemble`: no change to the seam — `execution`
  already carries everything synthesis needs.
- `OllamaModelClient._render_user_prompt` (synthesis branch): when
  `failed_at_step` is set, render a `Tool results:` section that shows
  the successful steps' outputs and ends with an explicit
  `step N: FAILED — <reason>` line. The synthesis system prompt picks
  up an extra clause: "if a step is marked FAILED, acknowledge the
  failure in your reply — do not invent successful outputs for it."
- `Repl._render_execution`: no change. Synthesis reply is rendered
  the same way it is on success. The `failed_at_step` is implementation
  detail; the user sees the synthesised acknowledgement plus any
  `[debug] plan executed: ...` markers when `BUTTER_DEBUG=1`.

Acceptance: re-run the `clock.diff` failure scenario from
2026-05-14. The REPL must (a) not crash, (b) print a coherent reply
from the model acknowledging the failure, and (c) record the reply in
history so the next user turn has the context. Verify with `BUTTER_DEBUG=1`.

### 2. Alias persistence across turns (future, captured here)

Today `outputs_as` aliases are scoped to a single plan inside one
turn. The model has no way to reference a value produced in turn N-1
from turn N. Symptom from 2026-05-14: model invents bare `$alias`
references hoping prior aliases survive.

Options:

- **Don't persist aliases at all.** Render prior-turn outputs into the
  user-prompt history block as plain values (e.g.
  `step 1 outputs: time=2026-05-14T12:01... tz=CEST`). The model
  sees the values inline and uses them as input literals, not
  references. Simplest; aligns with invariant #4 (variable resolution
  is dict lookup *within a plan*).
- **Persist aliases in conversation history.** Each `ConversationEntry`
  carries the `outputs` dict from the executed plan. The next turn's
  context exposes it under a `Prior outputs:` section. The model
  references via a new `$$turn-1.alias.field` form (or similar). More
  expressive but requires a new reference syntax and runtime resolver
  scope.

Decision deferred. Pick after the synthesis-on-failure work lands and
we have a few weeks of usage data on whether multi-turn alias use is
actually common.

## Known limitations (observed 2026-05-14)

- **Cross-turn alias resolution resolves itself via copy-paste.** Turn
  3 of the user-test asked "time difference between here and Australia"
  after two prior turns had emitted timestamps. Instead of declaring a
  3-step plan with `$alias.field` chaining, qwen3:8b read the prior
  turns' rendered outputs from history and pasted the literal
  timestamp strings into `clock.diff` inputs. This validates option
  (a) under "Alias persistence across turns" above: surfacing prior
  outputs as values in history is sufficient; a cross-turn `$$alias`
  syntax may not be needed at all.
- **Timezone-offset stripping when transcribing values.** ~~In the same
  turn 3, the model dropped the `+02:00` and `+10:00` offsets when
  copying timestamps from history…~~ **Addressed** on
  `feat/repl-prompt-toolkit`: synthesis system prompt gained a clause
  requiring tool-output values to be reproduced verbatim, including
  timezone offsets, fractional seconds, and any other suffix. The
  model paraphrases the same value in parentheses if it wants a
  friendlier form. Pinned by
  `test_synthesis_system_prompt_requires_verbatim_value_quoting`. The
  real test is live-REPL — re-run the
  here-vs-Australia diff after this lands. If the stripped form
  returns, the next step is to render *raw plugin outputs* into
  conversation history rather than just the synthesis text, so the
  model can copy from a non-paraphrased source.

## Out of scope

- Re-planning on failure (model gets the error and emits a new plan
  same turn). Risk of infinite loops; v2 at earliest.
- Structured tool-calling (Ollama's `tools` API). Still tracked as
  scope of `model-baseline-and-input-schema-prompt.md`. The findings
  here do not change that decision.
- Per-step error categorisation (transient vs permanent, retriable
  vs not). Plugins raise `Exception`; the executor wraps the message.
  Categorisation is a plugin-API extension.

## References

- `feat/capability-input-schema-prompt` (in flight): prompt-surface
  fix, prompt restructure, history non-persistence, plugin-exception
  wrap, BUTTER_DEBUG flag.
- `core/task_executor.py:185` — plugin invocation site.
- `core/loop.py:247` — `run_turn` execution branch.
- `model/ollama.py:_SYNTHESIS_SYSTEM_PROMPT` — synthesis prompt that
  needs the FAILED-step clause.
