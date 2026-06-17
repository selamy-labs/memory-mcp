"""MCP server exposing the markdown-memory store as typed tools.

This is an optional integration: install it with ``pip install memory-mcp[mcp]``.
The core package keeps its runtime dependencies minimal (stdlib only); the
``mcp`` SDK is required only to run this server.

Every tool is a thin wrapper over :class:`memory_mcp.core.MemoryStore`, so the
name validation, no-clobber / no-traversal write safety, and index
reconciliation live in exactly one place. Tools take structured inputs and
return JSON objects. Expected failures (bad name, missing memory, invalid type,
clobber attempt) surface as ``ToolError`` with a clean message.

There is **no delete tool** in this version: removing a memory is intentionally
not a one-call operation, to avoid accidental memory loss.

Configuration is resolved at call time from the environment: ``MEMORY_ROOT`` is
the path to the memory store's root directory (the directory holding the
per-memory ``.md`` files and ``MEMORY.md``). It defaults to
``~/.claude/projects/-home-dev/memory`` but must be overridable so the server is
not pinned to one box.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

try:
    from mcp.server.fastmcp import FastMCP
    from mcp.server.fastmcp.exceptions import ToolError
except ModuleNotFoundError as error:  # pragma: no cover - import guard
    raise SystemExit(
        "memory-mcp server requires the 'mcp' package. Install it with: pip install 'memory-mcp[mcp]'"
    ) from error

from memory_mcp.core import MemoryError, MemoryStore
from memory_mcp.storage import LocalStorage

DEFAULT_MEMORY_ROOT = "~/.claude/projects/-home-dev/memory"

INSTRUCTIONS = (
    "Typed access to the fleet markdown-memory store: one '.md' file per memory "
    "(YAML frontmatter name/description/metadata.type + markdown body) plus a "
    "MEMORY.md index. Reads: memory_search ranks memories by relevance over "
    "name+description+body; memory_get returns one full memory; memory_list "
    "enumerates them; memory_index returns the index pointer lines. Writes are "
    "careful: memory_write creates a new memory (refusing to overwrite an "
    "existing one) and adds its index pointer; memory_update edits an existing "
    "one. Names must be kebab-case slugs and are confined to the store root "
    "(no path traversal); the index is kept consistent (no duplicate pointer). "
    "There is no delete. The persist/recall methodology (when to save, how to "
    "dedupe, how to phrase) lives in the persist skill; this server is the call."
)

# One store per process. The root is resolved once at build time from the
# environment; no credentials are read or stored here.
_STORE: MemoryStore | None = None


def _memory_root() -> str:
    return os.environ.get("MEMORY_ROOT", DEFAULT_MEMORY_ROOT)


def _build_store() -> MemoryStore:
    """Construct the store from environment config. Separated so tests inject an
    in-RAM-backed store instead."""
    root = Path(_memory_root()).expanduser()
    return MemoryStore(LocalStorage(root))


def set_store(store: MemoryStore | None) -> None:
    """Install the store the tools use (tests inject an in-RAM-backed one)."""
    global _STORE
    _STORE = store


def _store() -> MemoryStore:
    global _STORE
    if _STORE is None:
        _STORE = _build_store()
    return _STORE


def _run(call: Any) -> dict[str, Any]:
    """Execute a store call, mapping expected failures to ``ToolError``."""
    try:
        return call()
    except MemoryError as error:
        raise ToolError(str(error)) from error


def memory_search(query: str, type: str | None = None, limit: int = 10) -> dict[str, Any]:
    """Search the memory store and return ranked hits.

    Matches ``query`` against each memory's name, description, and body (name
    matches rank highest, then description, then body; a whole-query phrase match
    is a strong bonus). Optionally filter by ``type`` (user/feedback/project/
    reference). Returns up to ``limit`` hits, each with name/description/type/
    path/score, plus the total number matched.
    """
    store = _store()
    return _run(lambda: store.search(query, type=type, limit=limit))


def memory_get(name: str) -> dict[str, Any]:
    """Return one memory's full frontmatter and body by its name (kebab slug)."""
    store = _store()
    return _run(lambda: store.get(name))


def memory_list(type: str | None = None) -> dict[str, Any]:
    """List every memory, optionally filtered by ``type``, sorted by name."""
    store = _store()
    return _run(lambda: store.list(type=type))


def memory_index() -> dict[str, Any]:
    """Return the MEMORY.md index pointer lines (the human-readable index)."""
    store = _store()
    return _run(store.index)


def memory_write(
    name: str,
    description: str,
    type: str,
    body: str,
    links: list[str] | None = None,
) -> dict[str, Any]:
    """Create a new memory and add its pointer to the index.

    ``name`` must be a kebab-case slug and is confined to the store root.
    ``type`` is one of user/feedback/project/reference and ``description`` is a
    non-empty single line. Refuses to overwrite an existing memory (use
    ``memory_update``). Optional ``links`` are appended to the body as
    ``[[name]]`` wiki-links. Returns the written path and indexed status.
    """
    store = _store()
    return _run(lambda: store.write(name, description, type, body, links=links))


def memory_update(
    name: str,
    description: str | None = None,
    type: str | None = None,
    body: str | None = None,
    links: list[str] | None = None,
) -> dict[str, Any]:
    """Edit an existing memory, preserving any field left unspecified.

    Only the provided fields change; the rest (including unknown metadata keys)
    are preserved. Reconciles the single index pointer in place. Raises if the
    memory does not exist (use ``memory_write`` to create it).
    """
    store = _store()
    return _run(lambda: store.update(name, description=description, type=type, body=body, links=links))


TOOLS = (
    memory_search,
    memory_get,
    memory_list,
    memory_index,
    memory_write,
    memory_update,
)


def build_server() -> FastMCP:
    """Build the memory-mcp server with every memory tool registered."""
    server = FastMCP("memory-mcp", instructions=INSTRUCTIONS)
    for tool in TOOLS:
        server.add_tool(tool)
    return server


def main() -> None:
    """Run the memory-mcp server over stdio."""
    build_server().run()


if __name__ == "__main__":  # pragma: no cover
    main()
