"""Minimal plugin fixture for plugin_source tests."""

from __future__ import annotations


class FakePlugin:
    async def execute(self, capability: str, inputs: dict[str, object]) -> dict[str, object]:
        del inputs
        if capability == 'ping':
            return {'reply': 'pong'}
        raise ValueError(f'fake: unknown capability {capability!r}')
