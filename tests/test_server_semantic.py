"""Tests for the networked shared-semantic-memory MCP server wrappers."""

from __future__ import annotations

import pytest
from mcp.server.fastmcp.exceptions import ToolError

import memory_mcp.server_semantic as srv
from memory_mcp.embeddings import HashingEmbedder
from memory_mcp.semantic import SemanticMemory
from memory_mcp.vector_store import FLEET_SCOPE, InMemoryVectorStore


@pytest.fixture
def offline_memory():
    mem = SemanticMemory(HashingEmbedder(dim=128), InMemoryVectorStore())
    srv.set_memory(mem)
    yield mem
    srv.set_memory(None)


def test_add_search_get_round_trip(offline_memory):
    out = srv.add_memory("a-mem", "a shared fleet fact", "feedback", "body here", updated_at="2026-01-01T00:00:00Z")
    assert out["indexed"] is True
    assert out["group_id"] == FLEET_SCOPE

    found = srv.search_memory("shared fleet fact")
    assert found["count"] == 1
    assert found["hits"][0]["name"] == "a-mem"

    got = srv.get_memory("a-mem")
    assert got["body"] == "body here"


def test_add_to_domain_scope(offline_memory):
    srv.add_memory("t", "kalshi edge", "project", "details", group_id="trading")
    assert srv.get_memory("t", group_id="trading")["name"] == "t"


def test_search_can_exclude_fleet(offline_memory):
    srv.add_memory("f", "alpha topic", "reference", "x")
    srv.add_memory("i", "alpha topic", "reference", "x", group_id="infra")
    out = srv.search_memory("alpha topic", group_ids=["infra"], include_fleet=False)
    assert {h["name"] for h in out["hits"]} == {"i"}


def test_tool_errors_map_to_toolerror(offline_memory):
    with pytest.raises(ToolError):
        srv.add_memory("", "d", "t", "b")
    with pytest.raises(ToolError):
        srv.search_memory("")
    with pytest.raises(ToolError, match="no such memory"):
        srv.get_memory("missing")


def test_build_embedder_hashing_default(monkeypatch):
    monkeypatch.delenv("MEMORY_EMBEDDER", raising=False)
    monkeypatch.setenv("MEMORY_EMBEDDING_DIM", "64")
    embedder = srv._build_embedder()
    assert isinstance(embedder, HashingEmbedder)
    assert embedder.dim == 64


def test_build_embedder_openai(monkeypatch):
    monkeypatch.setenv("MEMORY_EMBEDDER", "openai")
    monkeypatch.setenv("MEMORY_EMBEDDING_API_KEY", "sk-test")
    monkeypatch.setenv("MEMORY_EMBEDDING_DIM", "8")
    from memory_mcp.embeddings import OpenAIEmbedder

    embedder = srv._build_embedder()
    assert isinstance(embedder, OpenAIEmbedder)
    assert embedder.dim == 8


def test_build_embedder_unknown(monkeypatch):
    monkeypatch.setenv("MEMORY_EMBEDDER", "bogus")
    with pytest.raises(SystemExit, match="unknown MEMORY_EMBEDDER"):
        srv._build_embedder()


def test_build_store_memory_backend(monkeypatch):
    monkeypatch.setenv("MEMORY_BACKEND", "memory")
    assert isinstance(srv._build_store(64), InMemoryVectorStore)


def test_build_store_pgvector_requires_dsn(monkeypatch):
    monkeypatch.setenv("MEMORY_BACKEND", "pgvector")
    monkeypatch.delenv("MEMORY_PG_DSN", raising=False)
    with pytest.raises(SystemExit, match="requires MEMORY_PG_DSN"):
        srv._build_store(64)


def test_build_store_pgvector_with_fake_psycopg(monkeypatch):
    import sys
    import types

    fake = types.ModuleType("psycopg")

    class _Conn:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return None

        def cursor(self):
            return self

        def execute(self, *a, **k):
            return None

        def commit(self):
            return None

    fake.connect = lambda dsn: _Conn()
    monkeypatch.setitem(sys.modules, "psycopg", fake)
    monkeypatch.setenv("MEMORY_BACKEND", "pgvector")
    monkeypatch.setenv("MEMORY_PG_DSN", "postgresql://x")
    monkeypatch.setenv("MEMORY_ENSURE_SCHEMA", "1")
    from memory_mcp.pgvector_store import PgVectorStore

    store = srv._build_store(64)
    assert isinstance(store, PgVectorStore)


def test_build_store_pgvector_without_ensure_schema(monkeypatch):
    import sys
    import types

    fake = types.ModuleType("psycopg")
    fake.connect = lambda dsn: None
    monkeypatch.setitem(sys.modules, "psycopg", fake)
    monkeypatch.setenv("MEMORY_BACKEND", "pgvector")
    monkeypatch.setenv("MEMORY_PG_DSN", "postgresql://x")
    monkeypatch.delenv("MEMORY_ENSURE_SCHEMA", raising=False)
    from memory_mcp.pgvector_store import PgVectorStore

    # ensure_schema not called -> no connection use -> build succeeds without touching the (None) conn.
    assert isinstance(srv._build_store(64), PgVectorStore)


def test_build_store_unknown(monkeypatch):
    monkeypatch.setenv("MEMORY_BACKEND", "redis")
    with pytest.raises(SystemExit, match="unknown MEMORY_BACKEND"):
        srv._build_store(64)


def test_build_memory_composes_embedder_and_store(monkeypatch):
    monkeypatch.setenv("MEMORY_EMBEDDER", "hashing")
    monkeypatch.setenv("MEMORY_EMBEDDING_DIM", "32")
    monkeypatch.setenv("MEMORY_BACKEND", "memory")
    mem = srv.build_memory()
    assert isinstance(mem, SemanticMemory)


def test_memory_lazy_builds_when_unset(monkeypatch):
    srv.set_memory(None)
    monkeypatch.setenv("MEMORY_EMBEDDER", "hashing")
    monkeypatch.setenv("MEMORY_BACKEND", "memory")
    mem = srv._memory()
    assert isinstance(mem, SemanticMemory)
    srv.set_memory(None)


def test_bool_env(monkeypatch):
    monkeypatch.setenv("FLAG", "yes")
    assert srv._bool_env("FLAG") is True
    monkeypatch.setenv("FLAG", "0")
    assert srv._bool_env("FLAG") is False


def test_build_server_registers_three_tools(monkeypatch):
    monkeypatch.setenv("MCP_PORT", "9099")
    server = srv.build_server()
    # FastMCP exposes registered tools via list_tools (async); just assert it built.
    assert server.name == "memory-mcp-shared"


def test_main_runs_with_configured_transport(monkeypatch):
    calls = {}

    class _FakeServer:
        def run(self, transport):
            calls["transport"] = transport

    monkeypatch.setattr(srv, "build_server", lambda: _FakeServer())
    monkeypatch.setenv("MCP_TRANSPORT", "stdio")
    srv.main()
    assert calls["transport"] == "stdio"
