# memory-mcp

`memory-mcp` is a small [Model Context Protocol](https://modelcontextprotocol.io)
server over the fleet's **markdown-memory store**. It turns the read/write
*calls* on that store — search, get, list, index, and careful write/update —
into typed tools, with name validation, no-clobber / no-traversal write safety,
and index consistency baked into the server.

The persist/recall **methodology** — when to save a memory, how to dedupe, how
to phrase it — stays a skill (`persist`). This server is only the read/write
call.

## The store format

A memory **root directory** (configurable via `MEMORY_ROOT`) contains:

- One `.md` file per memory. YAML frontmatter with a top-level `name`
  (kebab-case slug) and `description` (one line, used for recall relevance),
  then a `metadata:` block with `type` (`user` | `feedback` | `project` |
  `reference`) plus any other keys (e.g. `node_type`), then a markdown body
  (which may contain `[[other-name]]` wiki-links):

  ```markdown
  ---
  name: feedback-prefer-wif
  description: Prefer Workload Identity Federation over service-account keys
  metadata:
    node_type: memory
    type: feedback
  ---

  Use WIF (keyless OIDC) instead of downloaded SA JSON keys. Related: [[some-other-memory]]
  ```

- `MEMORY.md` — a human-readable index, one pointer line per memory:
  `- [name.md](name.md) — hook`.

The frontmatter is parsed with the standard library only (no PyYAML), so the
core package has zero runtime dependencies. Unknown `metadata` keys
(`node_type`, `originSessionId`, …) are **preserved** across an update.

## Tools

| Tool | Purpose |
| --- | --- |
| `memory_search(query, type?, limit?)` | Ranked search over name + description + body (name match ranks highest; whole-query phrase match is a strong bonus). Optional `type` filter. |
| `memory_get(name)` | One memory's full frontmatter + body. |
| `memory_list(type?)` | Every memory (optionally by `type`), name-sorted. |
| `memory_index()` | The `MEMORY.md` pointer lines (the human-readable index). |
| `memory_write(name, description, type, body, links?)` | Create a new memory; refuses to overwrite an existing one; adds one index pointer. `links` are appended as `[[name]]`. |
| `memory_update(name, description?, type?, body?, links?)` | Edit an existing memory, preserving unspecified fields and unknown metadata; reconciles the index pointer in place. |

`type` is one of `user` / `feedback` / `project` / `reference`. Every search hit
carries `name` / `description` / `type` / `path` / `score`.

### Write safety

- **Validated names.** A `name` must be a kebab-case slug and is resolved to a
  single `<name>.md` inside the root. A name with a separator, `..`, or an
  absolute path is rejected before any byte is written — a write can never
  escape the configured store.
- **No clobber.** `memory_write` refuses to overwrite an existing memory; use
  `memory_update` to change one.
- **Index consistency.** Each write adds or replaces exactly one `MEMORY.md`
  pointer for that memory — never a duplicate.
- **Idempotent.** Rewriting the same content yields the same file and the same
  single index line.
- **No delete.** There is intentionally no delete tool in this version, to avoid
  accidental memory loss.

## Configuration (environment, resolved at call time)

| Variable | Effect |
| --- | --- |
| `MEMORY_ROOT` | Path to the memory store root (the directory of `.md` files + `MEMORY.md`). Defaults to `~/.claude/projects/-home-dev/memory`. No path is hardcoded into the package. |

No credentials are read or stored by this server.

## Install

Run directly from GitHub with the MCP extra:

```bash
uvx --from "git+https://github.com/selamy-labs/memory-mcp@v0.1.0#egg=memory-mcp[mcp]" memory-mcp
```

Or with pipx:

```bash
pipx install "memory-mcp[mcp] @ git+https://github.com/selamy-labs/memory-mcp@v0.1.0"
```

## MCP client config

```json
{
  "mcpServers": {
    "memory": {
      "command": "uvx",
      "args": [
        "--from",
        "git+https://github.com/selamy-labs/memory-mcp@v0.1.0#egg=memory-mcp[mcp]",
        "memory-mcp"
      ],
      "env": {
        "MEMORY_ROOT": "/path/to/your/memory"
      }
    }
  }
}
```

## Architecture

The store logic lives once in `memory_mcp.core.MemoryStore`; the MCP server in
`memory_mcp.mcp_server` is a thin wrapper that serialises structured results to
JSON and maps expected failures to `ToolError`. All file access goes through an
**injected storage** (`memory_mcp.storage`) and all timing through an injected
clock, so the full search / get / write / index path is exercised offline in
tests on an in-RAM root. The default `LocalStorage` and the stdlib document
parser keep the core package dependency-free; the `mcp` SDK is an optional
extra needed only to run the server.

## Development

```bash
python -m pip install -e ".[test]"
ruff format --check .
ruff check .
coverage run -m pytest
coverage report --fail-under=95
```

## License

MIT — see [LICENSE](LICENSE).
