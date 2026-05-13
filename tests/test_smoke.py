"""Smoke test — confirms the package imports."""

from __future__ import annotations

import butter_agent


def test_version_string() -> None:
    assert isinstance(butter_agent.__version__, str)
    assert butter_agent.__version__
