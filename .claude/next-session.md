# Next session pickup

Day-2 PR #2 merged on `main` (build order #3 `core/task_executor.py` shipped, 41 tests green). Next: build order #4 — `core/context_manager.py`, tracked in tasks-mcp task **358**; switch memory-mcp context to `butter-agent` and search kb domain `ai-butterbot` for context-footprint constraints (chunks `8ba89593-0375-45b5-846b-c2980dee95f8` and `985b4cce-0546-4adb-b2a1-2613a572be21`) — there is no dedicated executor-style design chunk for this module, derive from constraints + the `ContextManager` Protocol in `core/loop.py`.

Open question for next session: where conversation-history storage lives (in-process dict vs SQLite-backed). Scope chunks point to SQLite as default but a day-3 stub is acceptable. Local merged branch `core/day-2-task-executor` still exists; safe to delete with `git branch -d`.

---
Debriefed: 2026-05-13
Routed: tasks-mcp (1 task), memory-mcp context `butter-agent` (2 decisions)
