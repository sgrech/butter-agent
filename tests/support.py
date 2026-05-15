"""Shared test helpers (task #381 slice 2, checklist #395).

`FakePluginContext` lets a plugin be unit-tested in isolation — without
standing up a real `PluginRegistry` + `DefaultTaskExecutor` just to satisfy
the three-arg `Plugin.execute(capability, inputs, context)` signature. It
satisfies the `PluginContext` Protocol structurally: a programmable `call`
that records every invocation and returns a canned response (or raises a
preconfigured exception so a plugin's error-handling path is exercisable).

This is the seam future plugin tests (the `database` plugin in slice 3, then
`notes`) build on, so it lives in one place rather than being re-derived
inline per test module.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field


@dataclass
class FakePluginContext:
    """Stand-in `PluginContext` for unit-testing a plugin in isolation.

    Configure `responses` with the canned output for each fully-qualified
    `plugin.capability` ref the plugin under test will call. Optionally
    configure `errors` to make a given ref raise, exercising the plugin's
    failure path. Every `call` is appended to `calls` for assertions.

    A missing canned response is a test-wiring bug, not a runtime
    condition, so `call` raises `AssertionError` rather than returning an
    empty dict — that would silently mask an unstubbed dependency.

    `config` mirrors the real `PluginContext.config` — the plugin's own
    operator-supplied settings table. Defaults to empty; a test
    exercising a config-gated capability sets it explicitly.
    """

    responses: dict[str, dict[str, object]] = field(default_factory=dict)
    errors: dict[str, BaseException] = field(default_factory=dict)
    calls: list[tuple[str, dict[str, object]]] = field(default_factory=list)
    config: Mapping[str, object] = field(default_factory=dict)

    async def call(self, capability: str, inputs: dict[str, object]) -> dict[str, object]:
        self.calls.append((capability, dict(inputs)))
        if capability in self.errors:
            raise self.errors[capability]
        try:
            return dict(self.responses[capability])
        except KeyError as exc:
            raise AssertionError(
                f'FakePluginContext: no canned response for {capability!r}; configure FakePluginContext(responses={{{capability!r}: {{...}}}})',
            ) from exc
