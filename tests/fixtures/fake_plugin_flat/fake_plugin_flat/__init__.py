"""Flat-layout plugin fixture — no src/ directory."""

from __future__ import annotations


class FlatPlugin:
    async def execute(self, capability: str, inputs: dict[str, object]) -> dict[str, object]:
        del inputs
        return {'reply': f'flat-{capability}'}
