"""Built-in infrastructure plugins shipped inside butter-agent.

Unlike third-party plugins (fetched via [[plugin]] config and loaded by
`core.plugin_source`), these are constructed directly in `app.build_repl`
and registered into the same frozen `PluginRegistry`. They are reserved
infrastructure (database today; future log/fs/http) — all capabilities
`internal: true`, never visible to the planner.
"""
