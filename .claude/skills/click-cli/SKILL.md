---
name: click-cli
description: "Add or modify Click CLI commands — group structure, error handling, global tool install, serve subcommand"
argument-hint: <command-name>
allowed-tools: Grep, Glob, Read, Write, Edit, Bash
provenance:
  source: skill-library/python-core/click-cli
  catalog-version: "1.0.0"
  synced-at: "2026-05-13T05:43:30Z"
  content-hash: "sha256:7493682ba898a0cf2a25f31408fab7b8d64dcde8c8d2f851f009148c6fdb2f93"
  customized: false
---

## Parameters

- `command-name` (required): Name of the CLI command to create or modify (e.g., `repo-map`, `health`)

## Instructions

Add or modify a Click CLI command in a project that exposes its functionality as a globally installable terminal tool.

### Phase 1: Locate CLI Module

1. Find the CLI entry point: search for `cli/main.py`, `cli.py`, or a `[project.scripts]` entry in `pyproject.toml`
2. Read existing commands to understand the project's specific patterns:
   - What is the Click group name?
   - How are errors handled?
   - What is the delegation pattern (calls to service layer, tool functions, etc.)?
3. Read the implementation modules that commands delegate to

### Phase 2: Create or Modify the Command

Follow these rules exactly:

#### Group Structure
```python
@click.group()
@click.version_option(version=__version__, prog_name='tool-name')
def cli() -> None:
    """tool-name — one-line description."""
```
- One `@click.group()` as the root
- `@click.version_option()` pulls from `__version__` in the package `__init__.py`
- Group docstring is the CLI help text shown to users

#### Command Registration
```python
@cli.command('command-name')  # Explicit name if different from function name
@click.argument('path', default='.')
@click.option('--limit', default=20, show_default=True, help='Max items to show.')
def command_name(path: str, limit: int) -> None:
    """Single-sentence description shown in --help."""
```
- Use `@cli.command()` to register under the group
- Use `@click.argument()` for required positional args (add `default` to make optional)
- Use `@click.option()` for flags with `show_default=True` and `help=`
- Use `click.FloatRange()`, `click.IntRange()` for bounded numeric options
- Use `multiple=True` for repeatable options (returns tuple)

#### Error Handling
- If the underlying function returns error sentinels (e.g., strings starting with `# Error`), detect them and raise `click.ClickException()`:
  ```python
  ERROR_SENTINEL = '# Error'

  def _emit(result: str) -> None:
      if result.startswith(ERROR_SENTINEL):
          message = result.removeprefix(ERROR_SENTINEL).strip()
          raise click.ClickException(message)
      click.echo(result)
  ```
- Use `click.ClickException(str(exc))` for caught exceptions — never `sys.exit(1)` directly
- This ensures Click handles exit codes and stderr formatting consistently

#### Delegation Pattern
- CLI commands are thin wrappers — all logic lives in the implementation module
- Import and call the implementation function directly:
  ```python
  def repo_map(path: str, max_files: int, exclude: tuple[str, ...]) -> None:
      """Generate a markdown repository map."""
      exclude_patterns = list(exclude) if exclude else None
      _emit(generate_repo_map(path, max_files=max_files, exclude_patterns=exclude_patterns))
  ```
- Convert Click types to implementation types where needed (e.g., `tuple` → `list`)

#### Serve Subcommand
- MCP server projects should include a `serve` command:
  ```python
  @cli.command()
  def serve() -> None:
      """Start the MCP server in stdio mode."""
      from package_name.server import mcp
      mcp.run()
  ```
- Lazy-import the server to avoid loading MCP dependencies on every CLI invocation

#### Async Commands
- Click commands are sync. If delegating to async code, use a helper:
  ```python
  def _run_async(coro: Coroutine[Any, Any, None]) -> None:
      try:
          asyncio.run(coro)
      except (ChannelNotFoundError, ConfigError) as exc:
          raise click.ClickException(str(exc)) from exc
  ```
- Most projects with sync implementations don't need this

### Packaging

#### pyproject.toml
```toml
[project.scripts]
tool-name = "package_name.cli.main:cli"
```

#### __main__.py
```python
from package_name.cli.main import cli

if __name__ == '__main__':
    cli()
```

#### justfile
```just
# Install tool-name as a global CLI tool
install-cli:
    uv tool install --reinstall --from . package-name
```

### Canonical Examples

#### Sync Command (delegating to pure function)
```python
@cli.command('repo-map')
@click.argument('path', default='.')
@click.option('--max-files', default=200, show_default=True, help='Maximum files in the tree.')
@click.option('--exclude', multiple=True, help='Glob patterns to exclude (repeatable).')
def repo_map(path: str, max_files: int, exclude: tuple[str, ...]) -> None:
    """Generate a markdown repository map."""
    exclude_patterns = list(exclude) if exclude else None
    _emit(generate_repo_map(path, max_files=max_files, exclude_patterns=exclude_patterns))
```

#### Async Command (delegating to async service)
```python
@cli.command()
@click.argument('channel')
@click.option('--limit', default=20, show_default=True, help='Number of messages.')
def messages(channel: str, limit: int) -> None:
    """Read recent messages from a channel."""
    if not 1 <= limit <= 100:
        raise click.ClickException(f'limit must be between 1 and 100, got {limit}.')

    async def _run() -> None:
        ctx = await build_context()
        try:
            msgs = await ctx.polling.poll_and_read(channel, limit=limit)
            for msg in msgs:
                click.echo(f'[{msg.date}] {msg.sender}: {msg.text}')
        finally:
            await close_context(ctx)

    _run_async(_run())
```

### Phase 3: Verify

- [ ] CLI group has `@click.version_option()` pulling from `__version__`
- [ ] Each command has a single-sentence docstring
- [ ] `@click.argument()` for positional args, `@click.option()` for flags
- [ ] Options include `show_default=True` and `help=`
- [ ] Bounded numerics use `click.FloatRange()` or `click.IntRange()`
- [ ] Errors use `click.ClickException` — no `sys.exit(1)`
- [ ] Error sentinels from implementations detected and raised as ClickException
- [ ] Commands delegate to implementation functions — no business logic in CLI
- [ ] `[project.scripts]` entry in `pyproject.toml`
- [ ] `__main__.py` invokes the CLI group
- [ ] `install-cli` recipe in justfile for global tool install

### Boundaries
- Do NOT put business logic in CLI commands — only delegation and type conversion
- Do NOT use `sys.exit()` — let Click manage exit codes
- Do NOT add middleware or decorators beyond Click's own
- Do NOT import heavy dependencies at module level if only needed by one command (lazy import)
