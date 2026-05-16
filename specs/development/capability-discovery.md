# Capability discovery — progressive disclosure of the plugin menu

Status: implemented (issue #30) — behind `[core] capability_discovery`,
default off; `KeywordCapabilityFilter` retained as the off/fallback path.
Open question follow-ups (skip-threshold token measurement; model-authored
vs neutral Tier-1 purpose) deferred — see task #390 / issue #30.
Owner: shanegrech
Created: 2026-05-15
Follows from: live-REPL testing on 2026-05-15 (filesystem plugin,
`butter-plugin-filesystem` v0.1.0) surfaced the keyword-filter blindspot
recorded under "Motivation".

## Motivation

`KeywordCapabilityFilter` (`core/context_manager.py`) is the sole
mechanism deciding which capabilities the model sees when planning. It
ranks the registry's capability descriptors by token overlap with the
turn text and returns the top `top_k = 8`. When no capability shares a
token with the request it falls back to the **first 8 in registration
order**.

Live testing on 2026-05-15 exposed two failure modes from this single
mechanism, both with `qwen3:8b`:

1. **Truncation.** With `database`(internal, excluded) + `clock`(4) +
   `notes`(5) + `filesystem`(10) = 19 user-facing capabilities and
   `top_k = 8`, a plan can never see more than 8 — and which 8 is a
   lexical guess made *before* the model expresses intent.
2. **Registration-order fallback.** "what dependencies does pyproject
   have" shares no token with any `filesystem` description, so the
   filter fell back to the first 8 by registration order — all
   `clock`+`notes` — and the model never saw `filesystem.read_file`.
   It hallucinated `filesystem.read` (no such capability) and emitted a
   `read_file` step with no `path`. The executor correctly rejected both
   (`[error] plan rejected`), but the user hit a dead end. The only
   workaround was reordering `[[plugin]]` entries in `local-config.toml`
   so `filesystem` registers first — config order should not decide
   which plugin the planner is blind to.

The lexical filter is a v1 expedient. The correct model is **progressive
disclosure**: surface a tiny always-present index, let the model pull
detail for the plugin it actually intends to use.

## Goals

- The model can always discover *every* loaded plugin (no truncation,
  no lexical blindspot, no config-order dependence).
- Per-turn context stays small — the explicit purpose of
  `context_manager` ("filtered, not dumped"). Detail is pulled by
  intent, not pushed by guess.
- `qwen3:8b`-class local models reliably emit the correct capability
  name + required inputs (the two things they got wrong on 2026-05-15).
- No weakening of the architecture invariants.

## Non-goals

- Same-turn re-planning on a rejected plan (owned by
  `plugin-failure-recovery.md`, deferred to v2). Discovery happens
  *before* the planning pass, not as recovery after it.
- Semantic/embedding capability ranking. Discovery is by explicit model
  request, not a smarter scorer (a better scorer is a fallback nicety,
  not the design).
- Removing `KeywordCapabilityFilter` — it is retained as the fallback
  selector and the single-plugin / discovery-skipped path.

## Design — two-tier disclosure

### Tier 1: plugin index (always in context)

The intent-pass context carries a compact index: one row per loaded
plugin — `name` + a one-line plugin purpose — and **no capabilities**.
Bounded by plugin count (~3–5), not capability count (19+). Costs a
handful of lines, never truncates, has no keyword blindspot. Source: a
new optional `summary` field on the plugin manifest `[plugin]` section
(falls back to a generated "N capabilities: a, b, c…" line when absent,
so existing plugins need no change).

### Tier 2: on-demand capability schemas

Before the planning pass the model emits a discovery selection — the
plugin name(s) it intends to use. The loop returns those plugins' full
capability descriptors (`description` + `input_schema`, so required
inputs like `path` are always visible for exactly the plugin being
planned against), then runs the planning pass with that detail in
context. The model only ever loads the 1–2 plugins relevant to the
turn.

### Loop-shape delta vs invariant #1

Invariant #1 ("core loop never changes shape at runtime") forbids
*runtime reconfiguration* of the loop, not a deliberately redesigned
loop. This adds a fixed discovery phase to the loop's static shape:

```
turn → context(plugin index) → model: discovery selection
     → context(selected capability schemas) → model: plan
     → executor → synthesis → reply
```

This shape is fixed for every turn (no runtime branching of loop
structure), so it is consistent with invariant #1 once adopted as the
loop's definition. It is still a core redesign and must land as such —
`core/loop.py`, `core/context_manager.py`, the `CapabilityFilter` seam.

### Cost & mitigations

One extra model round-trip per turn. On `qwen3:8b` with JSON + CoT that
is seconds (`plugin-failure-recovery.md` finding 1 already flags the
synthesis turn doubling cost; this adds a third call). Mitigations,
each independently shippable:

- **Skip discovery** when ≤1 plugin is loaded, or when the registry's
  total user-facing capability count ≤ a threshold (the whole menu fits
  cheaply — surface it directly, no round-trip). This makes the common
  small-install case pay nothing.
- Allow the discovery selection to name multiple plugins in one turn so
  a cross-plugin plan (`clock.now → notes.create`) needs only one
  discovery round-trip.
- Cache the Tier-1 index on the frozen registry (invariant #2 makes it
  immutable, so it is built once).

## Invariant audit

- **#1 loop shape:** redefined deliberately, fixed for all turns, not
  runtime-reconfigured. Compatible (see above).
- **#2 frozen registry:** Tier-1 index and Tier-2 descriptors are
  derived from the frozen registry; built once, no mutation.
- **#3 atomic plan validation:** unchanged — discovery feeds the
  planning pass; the plan is still validated atomically before any
  execution.
- **#4 variable resolution:** untouched.
- **#5 gate enforcement:** untouched.
- **#6 isolation:** the index exposes only `name` + purpose; Tier-2
  exposes only the selected plugin's own public (`internal=False`)
  capability schemas. No plugin learns another's internals; internal
  capabilities stay excluded (as `_all_descriptors` already does).
- **#7 blast radius:** untouched (discovery is read-only context
  assembly).

## Migration

1. Add optional `summary` to manifest `[plugin]` (backward compatible;
   generated fallback when absent). No existing plugin repo must change.
2. Add a `CapabilityFilter`-side index selector + the discovery phase in
   the loop, behind a config switch (`[core] capability_discovery =
   true|false`, default `false` initially) so it can be A/B'd on local
   models without a flag-day.
3. `KeywordCapabilityFilter` remains the implementation when discovery
   is off and the fallback when the model declines/!skips discovery or a
   discovery selection is empty.
4. Live-REPL acceptance: the 2026-05-15 failing prompt ("what
   dependencies does pyproject have") yields a valid
   `filesystem.read_file{path: …}` plan with `filesystem` registered in
   any `[[plugin]]` position, on `qwen3:8b`, with no `local-config.toml`
   reordering.

## Open questions

- Discovery selection wire format: a dedicated model output variant
  (cleanest, matches the `ModelOutput` discriminated union) vs a
  reserved pseudo-capability the model "calls". The former keeps plans
  and discovery structurally distinct and is preferred; confirm against
  the `ModelClient` contract.
- Threshold for "skip discovery, just dump the menu" — pick by measured
  token cost of the full descriptor set on a representative install, not
  a guessed constant.
- Whether the Tier-1 plugin purpose should be model-authored guidance
  ("use this for X") vs a neutral description — the former plans better
  but is a prompt-injection surface from third-party manifests; lean
  neutral, sanitise at the trust boundary (same stance as the
  identifier-charset fix in `registry.py`).
