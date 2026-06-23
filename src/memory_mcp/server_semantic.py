"""Networked MCP server over the shared *semantic* memory (pgvector backend).

This is the in-cluster face of the shared fleet memory: a long-running service
that every agent and the orchestrator reach over the network (ClusterIP +
NetworkPolicy; the orchestrator via port-forward). It exposes the three
design-named tools -- ``add_memory`` / ``search_memory`` / ``get_memory`` -- each
a thin wrapper over :class:`~memory_mcp.semantic.SemanticMemory`, so the scope
namespacing, blended ranking, and recency live in exactly one place.

It is distinct from :mod:`memory_mcp.mcp_server` (the local, stdio,
markdown-file read/write server used via ``uvx`` against a local ``MEMORY_ROOT``)
because this one is *stateful and shared*: it talks to Postgres + pgvector and
serves many clients at once.

Configuration is resolved at start time from the environment; **no credentials
are baked into the image** -- the Postgres DSN is read from a value the
deployment mounts from a Secret (ExternalSecrets ← Google Secret Manager):

* ``MEMORY_BACKEND`` -- ``pgvector`` (default in-cluster) or ``memory`` (in-RAM,
  for a smoke test without a database).
* ``MEMORY_PG_DSN`` -- the Postgres DSN (``postgresql://user:pass@host:5432/db``);
  required when the backend is ``pgvector``.
* ``MEMORY_EMBEDDER`` -- ``hashing`` (default; zero-cost, no API key, fully
  self-hosted) or ``openai`` (uses ``MEMORY_EMBEDDING_API_KEY`` /
  ``MEMORY_EMBEDDING_BASE_URL`` / ``MEMORY_EMBEDDING_MODEL`` /
  ``MEMORY_EMBEDDING_DIM`` at call time -- never image-baked).
* ``MCP_TRANSPORT`` -- ``streamable-http`` (default for the service) or ``stdio``.
* ``MCP_HOST`` / ``MCP_PORT`` -- bind address for the HTTP transport
  (default ``0.0.0.0:8080``).

``MEMORY_ENSURE_SCHEMA=1`` makes the server create the pgvector schema on start
(idempotent) so a fresh database is usable without a separate migration step.
"""

from __future__ import annotations

import os
from typing import Any

try:
    from mcp.server.fastmcp import FastMCP
    from mcp.server.fastmcp.exceptions import ToolError
except ModuleNotFoundError as error:  # pragma: no cover - import guard
    raise SystemExit(
        "memory-mcp server requires the 'mcp' package. Install it with: pip install 'memory-mcp[mcp]'"
    ) from error

from memory_mcp.embeddings import EMBEDDING_DIM, Embedder, HashingEmbedder
from memory_mcp.semantic import SemanticMemory, SemanticMemoryError
from memory_mcp.vector_store import FLEET_SCOPE, InMemoryVectorStore, VectorStore

INSTRUCTIONS = (
    "Shared fleet semantic memory. Every agent and the orchestrator read and "
    "write ONE store, namespaced by group_id (scope): the shared 'fleet' scope "
    "is common ground, plus per-domain scopes (e.g. trading/infra/matchpoint/"
    "career). add_memory indexes a memory into a scope; search_memory ranks by a "
    "blend of semantic similarity, recency, and keyword overlap across one or "
    "more scopes (always including 'fleet' unless told otherwise); get_memory "
    "returns one memory's verbatim record. Markdown-in-git is the source of "
    "truth and this store is a rebuildable index over it, so prefer writing the "
    "durable memory to git (the persist skill) and treat add_memory as the fast "
    "shared-recall path. Recency reflects each memory's own date, not when it "
    "was indexed."
)

_MEMORY: SemanticMemory | None = None


def _bool_env(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


def _build_embedder() -> Embedder:
    """Select the embedder from the environment (hashing by default)."""
    choice = os.environ.get("MEMORY_EMBEDDER", "hashing").strip().lower()
    if choice == "hashing":
        dim = int(os.environ.get("MEMORY_EMBEDDING_DIM", str(EMBEDDING_DIM)))
        return HashingEmbedder(dim=dim)
    if choice == "openai":
        from memory_mcp.embeddings import HttpTransport, OpenAIEmbedder

        api_key = os.environ.get("MEMORY_EMBEDDING_API_KEY", "")
        base_url = os.environ.get("MEMORY_EMBEDDING_BASE_URL", "https://api.openai.com/v1")
        model = os.environ.get("MEMORY_EMBEDDING_MODEL", "text-embedding-3-small")
        dim = int(os.environ.get("MEMORY_EMBEDDING_DIM", "1536"))
        transport: HttpTransport = _UrllibTransport()
        return OpenAIEmbedder(transport=transport, api_key=api_key, base_url=base_url, model=model, dim=dim)
    raise SystemExit(f"unknown MEMORY_EMBEDDER {choice!r}: expected 'hashing' or 'openai'")


def _build_store(dim: int) -> VectorStore:
    """Select the vector store from the environment (pgvector by default)."""
    choice = os.environ.get("MEMORY_BACKEND", "pgvector").strip().lower()
    if choice == "memory":
        return InMemoryVectorStore()
    if choice == "pgvector":
        store = _build_pg_store(dim)
        if _bool_env("MEMORY_ENSURE_SCHEMA"):
            store.ensure_schema()
        return store
    raise SystemExit(f"unknown MEMORY_BACKEND {choice!r}: expected 'pgvector' or 'memory'")


def _build_pg_store(dim: int):
    """Build the pgvector store, preferring discrete params over a URL DSN.

    Discrete ``MEMORY_PG_HOST/PORT/USER/PASSWORD/DB`` are preferred because a
    password with URL-unsafe characters (``/`` ``+`` ``=`` from a base64 secret)
    silently breaks a ``postgresql://...`` DSN -- it gets misparsed into the wrong
    host. Falling back to ``MEMORY_PG_DSN`` keeps older config working.
    """
    from memory_mcp.pgvector_store import build_pg_store, build_pg_store_from_params

    host = os.environ.get("MEMORY_PG_HOST", "")
    if host:
        return build_pg_store_from_params(
            host=host,
            port=int(os.environ.get("MEMORY_PG_PORT", "5432")),
            user=os.environ.get("MEMORY_PG_USER", "memory"),
            password=os.environ.get("MEMORY_PG_PASSWORD", ""),
            dbname=os.environ.get("MEMORY_PG_DB", "memory"),
            dim=dim,
        )
    dsn = os.environ.get("MEMORY_PG_DSN", "")
    if not dsn:
        raise SystemExit("MEMORY_BACKEND=pgvector requires MEMORY_PG_HOST (preferred) or MEMORY_PG_DSN")
    return build_pg_store(dsn, dim=dim)


def build_memory() -> SemanticMemory:
    """Build the SemanticMemory from environment config (embedder + store)."""
    embedder = _build_embedder()
    store = _build_store(embedder.dim)
    return SemanticMemory(embedder, store)


def set_memory(memory: SemanticMemory | None) -> None:
    """Install the SemanticMemory the tools use (tests inject an offline one)."""
    global _MEMORY
    _MEMORY = memory


def _memory() -> SemanticMemory:
    global _MEMORY
    if _MEMORY is None:
        _MEMORY = build_memory()
    return _MEMORY


def _run(call: Any) -> dict[str, Any]:
    try:
        return call()
    except SemanticMemoryError as error:
        raise ToolError(str(error)) from error


def add_memory(
    name: str,
    description: str,
    type: str,
    body: str,
    group_id: str = FLEET_SCOPE,
    updated_at: str | None = None,
) -> dict[str, Any]:
    """Index one memory into the shared store under a scope (``group_id``).

    ``type`` is the memory's kind (user/feedback/project/reference). ``group_id``
    is the scope (default the shared ``fleet`` scope). ``updated_at`` is an
    optional ISO date used as the recency anchor (default now). Idempotent on
    ``(group_id, name)`` -- re-adding replaces the record.
    """
    memory = _memory()
    return _run(lambda: memory.add_memory(name, description, type, body, group_id=group_id, updated_at=updated_at))


def search_memory(
    query: str,
    group_ids: list[str] | None = None,
    include_fleet: bool = True,
    type: str | None = None,
    limit: int = 10,
) -> dict[str, Any]:
    """Search the shared memory across one or more scopes, ranked by relevance.

    Blends semantic similarity, recency, and keyword overlap. Searches the shared
    ``fleet`` scope by default; pass ``group_ids`` to add domain scopes.
    ``include_fleet`` keeps the shared scope in the search. Optionally filter by
    ``type``. Returns up to ``limit`` ranked hits with their component scores.
    """
    memory = _memory()
    return _run(
        lambda: memory.search_memory(query, group_ids=group_ids, include_fleet=include_fleet, type=type, limit=limit)
    )


def get_memory(name: str, group_id: str = FLEET_SCOPE) -> dict[str, Any]:
    """Return one indexed memory's verbatim record by ``(group_id, name)``."""
    memory = _memory()
    return _run(lambda: memory.get_memory(name, group_id=group_id))


TOOLS = (add_memory, search_memory, get_memory)


def build_server() -> FastMCP:
    """Build the shared semantic memory MCP server with its three tools."""
    host = os.environ.get("MCP_HOST", "0.0.0.0")  # noqa: S104 - in-cluster service binds all interfaces by design
    port = int(os.environ.get("MCP_PORT", "8080"))
    server = FastMCP("memory-mcp-shared", instructions=INSTRUCTIONS, host=host, port=port)
    for tool in TOOLS:
        server.add_tool(tool)
    return server


def main() -> None:
    """Run the shared semantic memory server (streamable-http by default)."""
    transport = os.environ.get("MCP_TRANSPORT", "streamable-http").strip()
    build_server().run(transport=transport)  # type: ignore[arg-type]


class _UrllibTransport:  # pragma: no cover - exercised only against a real endpoint
    """Default HTTP transport for OpenAIEmbedder using the stdlib (no new deps)."""

    def post_json(self, url: str, *, headers: dict[str, str], json: dict[str, Any]) -> dict[str, Any]:
        import json as _json
        import urllib.request

        data = _json.dumps(json).encode("utf-8")
        request = urllib.request.Request(url, data=data, headers=headers, method="POST")  # noqa: S310 - https from config
        with urllib.request.urlopen(request, timeout=30) as response:  # noqa: S310 - https endpoint from config
            return _json.loads(response.read().decode("utf-8"))


if __name__ == "__main__":  # pragma: no cover
    main()
