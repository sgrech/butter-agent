# Plugin Config Injection Spec

> **Status: implemented.** Landed in butter-agent core. This spec is the
> authoritative description of the per-plugin config contract that
> third-party plugins (starting with
> [`butter-plugin-filesystem`](https://github.com/sgrech/butter-plugin-filesystem))
> rely on for operator-controlled, plugin-scoped settings.

## 1. Purpose

Until now a plugin received data only through task-plan step inputs — there
was no channel for *operator policy* that the model must not be able to set
(e.g. "is `filesystem.delete` allowed at all?"). This feature adds a
read-only, plugin-scoped config table the operator declares in `config.toml`
and the plugin reads at execute time via `PluginContext.config`. It is the
mechanism the filesystem plugin uses to gate destructive operations behind
an operator flag rather than model discretion.

## 2. Scope

### In Scope

- A `config` inline table on each `[[plugin]]` entry in `config.toml`,
  parsed onto `PluginSource` / `PluginPath` (`core/config.py`).
- `dump_config` round-trip of the `config` table (`/configure` must not
  drop it): `load_config(dump_config(c)) == c` holds for all admitted
  value types.
- `PluginContext.config: Mapping[str, object]` — the executing plugin's
  own config, closed over by core, exposed read-only.
- Threading: `PluginDeclaration.config → LoadedPlugin → RegistryBuilder.
  register(config=…) → RegisteredPlugin → _PluginContext.config`.

### Out of Scope

- Interpreting or schema-validating config *values* — keys are
  plugin-private; core only enforces serialisability. A plugin validates
  its own keys (mirrors how manifests keep input/output schemas
  plugin-side).
- Secrets management / encryption — `config.toml` is plaintext, operator
  owned.
- Live reload — config is read once at startup; the registry is frozen
  (invariant #2).

## 3. Contract

### config.toml

```toml
[[plugin]]
path = "~/Workspace/butter-plugin-filesystem"
config = { allow_delete = true, allow_recursive_delete = false }
```

- `config` is optional; absent ⇒ empty mapping (never `None`).
- Values may be string, integer, float, bool, array, or nested table.
  Any other type (notably TOML date/time → `datetime`) is rejected at
  parse time with `ConfigError` — guarantees the `dump_config`
  round-trip invariant can never be silently violated.

### Plugin surface

```python
class PluginContext(Protocol):
    @property
    def config(self) -> Mapping[str, object]: ...
    async def call(self, capability: str, inputs: dict[str, object]) -> dict[str, object]: ...
```

- `config` is keyed by the **executing** plugin's identity — the same
  closed-over owner `call` uses, never a call argument.
- The returned mapping is a `MappingProxyType` over a private copy:
  third-party plugin code cannot mutate the registry's snapshot.
- Backward compatible: `call`-only plugins (notes, clock at their pinned
  versions) never touch `config`; `register()` keeps its two-argument
  form via a keyword-only `config=` default.

## 4. Invariants

- **#6 (isolation):** a plugin can read only its own config. In a nested
  `PluginContext.call`, the child context's owner is the *target* plugin,
  so the target sees its own config and never the caller's — verified by
  `test_plugin_context_config_isolated_per_owner`.
- **#2 (frozen registry):** config is captured into `RegisteredPlugin` at
  build time; no post-startup mutation path exists.
- **#7 (policy is core-side):** because config is operator-declared and
  model-invisible, a plugin can use it as a hard gate the model cannot
  talk its way past — the basis for `filesystem.delete`'s `allow_delete`.

## 5. Critical Files

| File | Change |
|------|--------|
| `core/config.py` | `config` field on `PluginSource`/`PluginPath`; `_parse_plugin_config`; inline-table serialiser in `dump_config` |
| `core/registry.py` | `config` on `RegisteredPlugin`; `PluginContext.config` Protocol member; `register(config=…)` |
| `core/task_executor.py` | `_PluginContext.config` property (owner-keyed, `MappingProxyType`) |
| `core/plugin_source.py` | `LoadedPlugin.config` carried from declaration |
| `app.py` | `builder.register(..., config=entry.config)` |
| `tests/support.py` | `FakePluginContext.config` for Protocol conformance |

## 6. Verification

- `just check` (ruff + format + mypy strict + pytest) green.
- `tests/test_config.py` — parse, default-empty, table-required,
  nested/list values, unserialisable-type rejection.
- `tests/test_cli_commands.py` — `dump_config` round-trip with a config
  table covering every admitted value type (incl. integer-valued float).
- `tests/test_task_executor.py` — own-config exposure, default empty,
  per-owner isolation, read-only enforcement.
- Plugin authors validate the shipped `manifest.toml` + config contract
  via `butter_agent.plugin_api.parse_manifest` in their own repo tests;
  `FakePluginContext(config={...})` exercises config-gated capabilities.
