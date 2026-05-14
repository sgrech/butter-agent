# Model baseline + capability `input_schema` in prompt

Status: change 1 implemented; change 2 superseded by 2026-05-14 findings
Owner: shanegrech
Created: 2026-05-14
Follow-up: see `plugin-failure-recovery.md` for findings from live-REPL
testing that reshaped change 2.

## Motivation

User-testing on 2026-05-13 (post PR #16, synthesis turn merged) surfaced
two failure modes on the homelab Ollama running `qwen3:8b`:

1. The model occasionally returned `ModelReply` text with literal
   placeholder syntax (`"The current time is $(clock.now)."`) instead of
   a structured `TaskPlan`. Symptom of poor JSON-mode discipline + CoT
   overhead with no tool-use training.
2. The model emitted multi-step plans where step N omitted a required
   input (e.g. `clock.now` step missing `tz`), which the executor
   atomically rejected with `PlanValidationError`. The model had no way
   to know `tz` was required — the prompt only renders the capability's
   *description*, never its `input_schema`. See `context_manager.py`:
   `CapabilityDescriptor` deliberately drops schema, with the comment
   *"Input/output schemas are the executor's concern... not the
   model's"*. That decision is wrong in retrospect — the model cannot
   produce a valid plan without knowing required inputs.

PR #16 made the second failure non-fatal (REPL recovers via
`[error] plan rejected: ...`) but the underlying gap remains. Both
failures will recur on every plan that needs a non-obvious input key.

## Scope

Two coupled changes, ordered so the prompt fix lands first and the model
swap is benchmarked against a fixed prompt surface.

### 1. Surface `input_schema` in capability prompt section

- Extend `CapabilityDescriptor` (`core/context_manager.py`) with the
  required key set — minimally just the list of required input names,
  not the full JSON Schema. Token budget matters; "requires: tz" beats
  a verbatim schema blob.
- Update `_render_user_prompt` in `model/ollama.py` to render each
  capability as e.g.
  `- clock.now (requires: tz): Return the current wall-clock time.`
  Empty-required-set capabilities render unchanged.
- `KeywordCapabilityFilter` should also tokenise the required-keys list
  into the haystack so "what time is it in china" still ranks
  `clock.now` highly when the description doesn't mention timezones.
- Tests: extend `test_ollama.py` (`test_prompt_surfaces_capabilities`
  family) and add a `_manifest_toml` helper variant in
  `test_context_manager.py` that registers a capability with a non-empty
  `input_schema` so the descriptor surface is exercised end-to-end.

Acceptance: the model can produce a valid 2-step plan for "time
difference between here and china" without `PlanValidationError`. Verify
in the live REPL.

### 2. ~~Switch default model to `hermes3:8b`~~ — superseded 2026-05-14

Live-REPL testing on 2026-05-14 invalidated the premise. Hermes3:8b
confabulates plausible values without invoking plugins. Mistral-Nemo
12B confabulates with internally inconsistent timezone math.
**Qwen3:8b is the only local 8–12B model that reliably invokes
plugins** — confirmed via `BUTTER_DEBUG=1` showing `[plan executed:
1 step(s)] step 1: clock.now ...`. The thinking-tag CoT cost is real
but acceptable for v1 correctness. Keep qwen3:8b as the default.
Revisit the model swap when one of: (a) a non-CoT 8–12B model
demonstrates equivalent plugin-call discipline in live REPL, or (b)
butter adopts Ollama's `tools` API (structured function calling),
which removes the prompt-only reliance.

Original (now-stale) reasoning preserved below for context:



- Both models pulled on 2026-05-13 to the homelab Ollama at
  `192.168.4.33:11434` (pulls were running in background at debrief
  time; verify completion with `ollama list` against that host before
  benchmarking).
- Update `config.toml` (the committed default) and `core/config.py`
  default to `hermes3:8b`. `local-config.toml` is gitignored and not
  touched by this change.
- Reasoning recorded so the next session does not re-litigate:
  - Workload is two model calls per plan turn (intent + synthesis). CoT
    models double the cost twice.
  - Hermes3 (Llama-3.1-8b fine-tune by Nous) is explicitly trained on
    function calling and JSON output. No thinking-tag overhead.
  - Mistral-Nemo 12B is the step-up if VRAM allows (~14GB at Q4_K_M);
    bigger context, strong JSON, no CoT. The homelab has headroom per
    the user.
  - Avoid `qwen3`, `deepseek-r1`, and other thinking-tag models for the
    *planning* call — Ollama JSON mode discards the reasoning trace but
    you still pay for it. Acceptable for synthesis, but a two-model loop
    is out of scope for v1.

Acceptance: default install runs against `hermes3:8b` out of the box;
`README` / `config.toml` comment block names `mistral-nemo:12b` as the
sanctioned step-up. User can verify with the same scenario set used to
validate change 1.

### 3. Benchmark scenarios (informal, not automated)

Run these against both models from the REPL after change 1 ships. Goal
is feel-grade comparison, not a test fixture.

- Single-tool, single-call: "what time is it?"
- Single-tool, with non-obvious input: "what time is it in Tokyo?"
- Multi-call without dependency: "time difference between Italy and
  China" (the failing scenario from 2026-05-13)
- Halt-and-confirm: any plan with `gate: confirm` — synthesis must not
  fire on abort.
- Plain chat with no capability match: "tell me a joke" — must produce
  a `ModelReply`, never a fabricated plan.

## Out of scope

- Function-calling protocol (Ollama's `tools` API). Butter's `format:
  'json'` + structured schema in the prompt is the v1 contract; the
  function-calling path is a later spec.
- Per-call model selection (different model for intent vs synthesis).
- `tasks-mcp` / `memory-mcp` / `knowledgebase-mcp` integration —
  servers were disconnected at debrief time. If they remain so when
  this spec is implemented, capture follow-up findings in this spec
  rather than the MCP stores.

## References

- PR #16 (merged 2026-05-13): synthesis turn + REPL recovery from
  plan errors. Commit `7e2e01f`.
- `src/butter_agent/core/context_manager.py:60-67` —
  `CapabilityDescriptor` definition that drops `input_schema` today.
- `src/butter_agent/model/ollama.py:226` — capability rendering
  block that needs the `(requires: ...)` extension.
