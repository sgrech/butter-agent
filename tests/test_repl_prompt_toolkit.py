"""Tests for the prompt_toolkit-backed REPL components.

The `PromptToolkitInputSource` itself is hard to test without a real
TTY (prompt_toolkit's event loop expects one). These tests focus on
the parts that can be exercised in isolation:

- `InferenceIndicator` state machine: spinner-on-TTY, no-op-off-TTY,
  line-clear on exit, robustness when the awaited block raises.
- `PromptToolkitInputSource` construction: history wiring, completer
  population from slash commands.

Integration of the spinner into `Repl.run` is covered by tests in
`test_repl.py` via the `indicator_factory` seam — those tests inject
a recording stub so they pass without a TTY.
"""

from __future__ import annotations

import asyncio
import io

import pytest
from prompt_toolkit.completion import CompleteEvent
from prompt_toolkit.document import Document

from butter_agent.repl_prompt_toolkit import (
    InferenceIndicator,
    PromptToolkitInputSource,
    _SlashCommandCompleter,
)


class _FakeTTYStream(io.StringIO):
    """StringIO that lies about being a TTY.

    Lets the indicator render its frames into a buffer we can inspect.
    """

    def isatty(self) -> bool:
        return True


# --- InferenceIndicator -----------------------------------------------------


async def test_indicator_writes_frames_to_tty_stream() -> None:
    """Spinner runs in the background and updates the stream."""
    stream = _FakeTTYStream()
    async with InferenceIndicator(message='thinking', stream=stream, interval=0.01):
        # Give the animator a couple of ticks to render frames.
        await asyncio.sleep(0.05)
    output = stream.getvalue()
    # At least one frame must have been rendered.
    assert 'thinking' in output
    # The exit path clears the line so the next stdout write starts clean.
    assert output.endswith('\r\x1b[K')


async def test_indicator_disabled_for_non_tty_stream() -> None:
    """When the stream is not a TTY (CI logs, pipes), the indicator emits nothing.

    Keeps non-interactive output free of cursor-control escape sequences.
    """
    stream = io.StringIO()  # not a TTY
    async with InferenceIndicator(message='thinking', stream=stream, interval=0.01):
        await asyncio.sleep(0.03)
    assert stream.getvalue() == ''


async def test_indicator_clears_line_when_awaited_block_raises() -> None:
    """The line is wiped on exit even if the wrapped code raises.

    Without this, the spinner glyph would sit next to the error message
    the REPL prints when bubbling the exception up.
    """
    stream = _FakeTTYStream()
    with pytest.raises(RuntimeError, match='boom'):
        async with InferenceIndicator(stream=stream, interval=0.01):
            await asyncio.sleep(0.02)
            raise RuntimeError('boom')
    # Final write to stream is the clear sequence.
    assert stream.getvalue().endswith('\r\x1b[K')


async def test_indicator_does_not_emit_when_awaited_returns_immediately() -> None:
    """A turn that completes before the first sleep tick still exits cleanly.

    Tightest race: the background animator may not have written any
    frame before being cancelled. The clear-on-exit still runs because
    the indicator was enabled on a TTY stream.
    """
    stream = _FakeTTYStream()
    async with InferenceIndicator(stream=stream, interval=10.0):
        # No await — exit immediately.
        pass
    # Whatever was written (probably nothing, or one frame), the last
    # bytes must be the clear sequence so the next line starts clean.
    assert stream.getvalue().endswith('\r\x1b[K')


async def test_indicator_pause_stops_frame_emission_and_clears_line() -> None:
    """While paused, the animator must not write spinner frames.

    PR #18 review (Copilot): the spinner kept animating during
    `confirm`/`human` gate prompts because the indicator wraps the
    entire `run_turn`. `pause()` is the suspension hook the gate
    handler calls via `suspend_indicator()`.
    """
    stream = _FakeTTYStream()
    async with InferenceIndicator(message='thinking', stream=stream, interval=0.01) as indicator:
        # Let one or two frames render first.
        await asyncio.sleep(0.03)
        baseline = stream.getvalue()
        indicator.pause()
        # Whatever was on the line gets cleared as part of pause().
        assert stream.getvalue().endswith('\r\x1b[K')
        await asyncio.sleep(0.03)
        # No new frames should have been written after the clear.
        assert stream.getvalue() == baseline + '\r\x1b[K'


async def test_indicator_resume_restarts_frame_emission() -> None:
    """After resume(), the animator writes frames again."""
    stream = _FakeTTYStream()
    async with InferenceIndicator(message='thinking', stream=stream, interval=0.01) as indicator:
        indicator.pause()
        await asyncio.sleep(0.02)
        paused_state = stream.getvalue()
        indicator.resume()
        await asyncio.sleep(0.03)
        # At least one new frame written after resume.
        assert len(stream.getvalue()) > len(paused_state)


async def test_suspend_indicator_pauses_and_resumes_active_indicator() -> None:
    """`suspend_indicator()` is the public API the gate handler uses.

    Round-trip test: spinner runs, suspend, no frames during the
    suspended block, frames resume after. The contextvar wiring lives
    in `core.repl`; this test confirms the indicator and the helper
    cooperate.
    """
    from butter_agent.core.repl import suspend_indicator

    stream = _FakeTTYStream()
    async with InferenceIndicator(message='thinking', stream=stream, interval=0.01):
        await asyncio.sleep(0.03)
        before = stream.getvalue()
        async with suspend_indicator():
            # `suspend_indicator` should have paused the active indicator.
            # Allow a tick for the animator to observe the pause flag.
            await asyncio.sleep(0.02)
            mid = stream.getvalue()
            # Only the clear-line sequence appended after entry.
            assert mid.endswith('\r\x1b[K')
            # No further spinner frames written during the suspended block.
            assert mid.count('thinking') == before.count('thinking')
        # After suspend exits, frames resume.
        await asyncio.sleep(0.03)
        after = stream.getvalue()
        assert after.count('thinking') > mid.count('thinking')


async def test_suspend_indicator_is_noop_when_no_indicator_registered() -> None:
    """In tests / non-TTY runs no indicator is active; suspend must not blow up."""
    from butter_agent.core.repl import suspend_indicator

    async with suspend_indicator():
        # Body runs normally; no exception, no registered indicator to touch.
        pass


# --- _SlashCommandCompleter -------------------------------------------------


def _completions(completer: _SlashCommandCompleter, text: str) -> list[str]:
    """Run the completer for `text` and return the completion strings."""
    return [c.text for c in completer.get_completions(Document(text=text, cursor_position=len(text)), CompleteEvent())]


def test_slash_completer_returns_nothing_for_plain_text() -> None:
    """User-test on 2026-05-14: completions used to pop up on every space.

    The fix is to ignore any input that does not start with a `/` so
    completion only appears in the contexts where it's meaningful
    (typing a slash command).
    """
    completer = _SlashCommandCompleter(['help', 'quit', 'status'])
    assert _completions(completer, '') == []
    assert _completions(completer, 'what') == []
    assert _completions(completer, 'what about ') == []
    assert _completions(completer, 'hello /not-at-start') == []


def test_slash_completer_returns_matching_commands_for_slash_prefix() -> None:
    completer = _SlashCommandCompleter(['help', 'quit', 'status'])
    assert _completions(completer, '/') == ['/help', '/quit', '/status']
    assert _completions(completer, '/q') == ['/quit']
    assert _completions(completer, '/h') == ['/help']
    # Typo / no match: empty.
    assert _completions(completer, '/zzz') == []


# --- PromptToolkitInputSource construction ----------------------------------


def test_input_source_constructs_with_no_history_and_no_completer() -> None:
    """Minimal construction (no history, no commands) must succeed.

    Lets callers opt out of persistence and completion without
    fabricating placeholder values.
    """
    src = PromptToolkitInputSource()
    # The session is created lazily-but-eagerly during __init__; just
    # confirm the object exists and didn't raise on optional features.
    assert src is not None


def test_input_source_constructs_with_history_and_slash_commands(tmp_path: object) -> None:
    """History path + completer are accepted without error."""
    from pathlib import Path

    history = Path(str(tmp_path)) / 'history'
    src = PromptToolkitInputSource(
        history_path=history,
        bottom_toolbar_text='butter-agent · qwen3:8b',
        slash_commands=('help', 'configure', 'quit'),
    )
    assert src is not None
