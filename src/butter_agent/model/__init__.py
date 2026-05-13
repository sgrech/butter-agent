"""Model provider adapters.

Each module under `model/` exposes a concrete `ModelClient` implementation
for a specific provider (Ollama, future: external APIs). The adapters
live outside `core/` so swapping providers does not touch the loop or
the executor — invariant #1 (loop shape never changes) is preserved.
"""
