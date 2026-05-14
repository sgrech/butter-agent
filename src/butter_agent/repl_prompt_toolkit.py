"""prompt_toolkit-backed `InputSource` + inference indicator for the REPL.

Why this module exists separately from `core/repl.py`:

- `core/repl.py` owns the `InputSource` / `Output` Protocols and the
  pure stdio implementations used in tests and non-TTY runs. Pulling
  `prompt_toolkit` in there would force every test to load the
  terminal toolkit's curses/asyncio scaffolding.
- This module sits beside core (not inside it) so the dependency
  surface is localised — only the entry-point composition in
  `app.py` imports from here, and only when a TTY is detected.

What this module provides:

- `PromptToolkitInputSource` — drop-in `InputSource` replacement.
  Adds persistent history, up/down arrow navigation, Ctrl-R reverse
  search, tab completion of registered slash commands, and a bottom
  toolbar showing the active model + host.
- `InferenceIndicator` — async context manager that renders a
  rotating braille spinner on stderr while `AgentLoop.run_turn`
  awaits. Disabled automatically when stderr is not a TTY so logs
  stay clean under `>` / `|` redirection.

Neither class touches the existing `Output` Protocol; the stdio
`Output` continues to handle reply rendering. `prompt_toolkit`
redraws the input prompt below any text we emit, so plain stdout
writes from the loop and the spinner clears with an ANSI erase line
on exit.
"""

from __future__ import annotations

import asyncio
import sys
from collections.abc import Iterable
from contextlib import suppress
from pathlib import Path
from types import TracebackType
from typing import TextIO

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import CompleteEvent, Completer, Completion
from prompt_toolkit.document import Document
from prompt_toolkit.history import FileHistory

from butter_agent.core.repl import register_active_indicator, unregister_active_indicator

# --- Slash-command completion -----------------------------------------------


class _SlashCommandCompleter(Completer):
    """Suggest slash commands only when the line starts with `/`.

    `WordCompleter` triggers on any token, so completions popped up
    every time the user typed a space mid-sentence (user-reported on
    2026-05-14). This completer ignores the input until it begins with
    `/`, then offers a filtered list of registered commands matching
    the prefix typed so far.
    """

    def __init__(self, commands: Iterable[str]) -> None:
        # Store with the leading `/` so prefix matching against
        # `document.text_before_cursor` is direct, no string surgery.
        self._commands: tuple[str, ...] = tuple(f'/{name}' for name in commands)

    def get_completions(self, document: Document, complete_event: CompleteEvent) -> Iterable[Completion]:
        text = document.text_before_cursor
        if not text.startswith('/'):
            return
        for cmd in self._commands:
            if cmd.startswith(text):
                # `start_position` is negative — how far back from the
                # cursor the completion replaces. Substituting the
                # whole current word (including the leading `/`) gives
                # the "type /, hit Tab, see /quit /help …" UX.
                yield Completion(cmd, start_position=-len(text))


# --- Input source -----------------------------------------------------------


class PromptToolkitInputSource:
    """`InputSource` backed by a `PromptSession`.

    History persists at the supplied `history_path` so up-arrow works
    across butter invocations. Tab-completion is offered for the
    registered slash commands; the user types `/` then Tab to see what
    is available.

    A `PromptSession` is constructed once and reused — it accumulates
    history entries internally as `prompt_async` returns, so the
    caller does not need to `history.append_string` manually.
    """

    def __init__(
        self,
        *,
        history_path: Path | None = None,
        bottom_toolbar_text: str = '',
        slash_commands: Iterable[str] = (),
    ) -> None:
        history = FileHistory(str(history_path)) if history_path is not None else None
        cmds = tuple(slash_commands)
        completer = _SlashCommandCompleter(cmds) if cmds else None
        # The toolbar callable is invoked on every redraw; keep it pure
        # so prompt_toolkit can call it freely. The text is fixed for
        # this session (set at construction from the active config).
        self._toolbar_text = bottom_toolbar_text
        self._session: PromptSession[str] = PromptSession(
            history=history,
            bottom_toolbar=lambda: self._toolbar_text,
            completer=completer,
        )

    async def read_line(self, prompt: str) -> str:
        # `prompt_async` raises EOFError on Ctrl-D, matching the
        # `InputSource` Protocol contract — Repl.run treats it as a
        # graceful shutdown signal.
        return await self._session.prompt_async(prompt)


# --- Inference indicator ----------------------------------------------------


_SPINNER_FRAMES: tuple[str, ...] = ('⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏')

# `\r` returns the cursor to column 0; `\x1b[K` erases from the cursor
# to end-of-line. Together they wipe the current line without scrolling.
_CLEAR_LINE = '\r\x1b[K'


class InferenceIndicator:
    """Async context manager that animates a spinner on stderr.

    Enters: spawns a background asyncio task that rewrites a single
    stderr line with rotating braille frames + the supplied message.
    Exits: cancels the task and clears the line so the next stdout
    write starts cleanly.

    Disabled automatically when the target stream is not a TTY.
    Keeps non-interactive output (CI logs, piped runs) free of
    cursor-control escape sequences.
    """

    def __init__(
        self,
        message: str = 'thinking',
        *,
        stream: TextIO | None = None,
        interval: float = 0.08,
    ) -> None:
        self._message = message
        self._stream: TextIO = stream if stream is not None else sys.stderr
        self._interval = interval
        self._task: asyncio.Task[None] | None = None
        self._paused = False
        self._token: object | None = None
        # Decided once at construction so unit tests that swap streams
        # behave deterministically — re-checking on enter would let a
        # late TTY change racily enable the spinner mid-turn.
        self._enabled = self._stream.isatty()

    async def __aenter__(self) -> InferenceIndicator:
        if self._enabled:
            # Register before starting the animator so any code that
            # observes the ContextVar (gate handler, future progress
            # plugins) sees a paused-when-needed handle for the full
            # lifetime of the spinner.
            self._token = register_active_indicator(self)
            self._paused = False
            self._task = asyncio.create_task(self._animate())
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if self._token is not None:
            unregister_active_indicator(self._token)
            self._token = None
        if self._task is None:
            return
        self._task.cancel()
        with suppress(asyncio.CancelledError):
            await self._task
        # Always clear the line on exit, even if the awaited work
        # raised — otherwise the spinner glyph would sit next to the
        # error message the REPL is about to print.
        self._stream.write(_CLEAR_LINE)
        self._stream.flush()

    def pause(self) -> None:
        """Stop drawing frames and clear the current line.

        Used by `core.repl.suspend_indicator()` when something else
        takes over the terminal — typically a `confirm`/`human` gate
        prompt that blocks on operator input. The animator task keeps
        running but emits nothing while `_paused` is set; `resume()`
        re-enables frame writes.
        """
        if not self._enabled:
            return
        self._paused = True
        self._stream.write(_CLEAR_LINE)
        self._stream.flush()

    def resume(self) -> None:
        """Counterpart to `pause()` — start emitting frames again."""
        if not self._enabled:
            return
        self._paused = False

    async def _animate(self) -> None:
        idx = 0
        while True:
            if not self._paused:
                frame = _SPINNER_FRAMES[idx % len(_SPINNER_FRAMES)]
                self._stream.write(f'{_CLEAR_LINE}{frame} {self._message}...')
                self._stream.flush()
            await asyncio.sleep(self._interval)
            idx += 1
