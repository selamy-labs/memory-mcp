"""Tests for the networked shared-semantic-memory MCP server wrappers."""

from __future__ import annotations

import pytest

import memory_mcp.server_semantic as srv
from memory_mcp.embeddings import HashingEmbedder
from memory_mcp.semantic import SemanticMemory
from memory_mcp.vector_store import InMemoryVectorStore


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


def test_build_store_pgvector_requires_host_or_dsn(monkeypatch):
    monkeypatch.setenv("MEMORY_BACKEND", "pgvector")
    monkeypatch.delenv("MEMORY_PG_DSN", raising=False)
    monkeypatch.delenv("MEMORY_PG_HOST", raising=False)
    with pytest.raises(SystemExit, match="MEMORY_PG_HOST"):
        srv._build_store(64)


def test_build_store_pgvector_prefers_discrete_params(monkeypatch):
    import sys
    import types

    captured = {}
    fake = types.ModuleType("psycopg")

    def fake_connect(**kwargs):
        captured.update(kwargs)

        class _C:
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return None

            def cursor(self):
                return self

            def execute(self, *a, **k):
                return None

            def fetchone(self):
                return (0,)

            def commit(self):
                return None

        return _C()

    fake.connect = fake_connect
    monkeypatch.setitem(sys.modules, "psycopg", fake)
    monkeypatch.setenv("MEMORY_BACKEND", "pgvector")
    monkeypatch.setenv("MEMORY_PG_HOST", "pg.host")
    monkeypatch.setenv("MEMORY_PG_PASSWORD", "a/b+c=d")
    monkeypatch.delenv("MEMORY_ENSURE_SCHEMA", raising=False)
    from memory_mcp.pgvector_store import PgVectorStore

    store = srv._build_store(64)
    assert isinstance(store, PgVectorStore)
    # Force a connection to confirm discrete params reached psycopg verbatim.
    store.count()
    assert captured["host"] == "pg.host"
    assert captured["password"] == "a/b+c=d"


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


def test_bool_env(monkeypatch):
    monkeypatch.setenv("FLAG", "yes")
    assert srv._bool_env("FLAG") is True
    monkeypatch.setenv("FLAG", "0")
    assert srv._bool_env("FLAG") is False


def test_main_refuses_startup_without_reviewed_production_composition():
    with pytest.raises(SystemExit, match="production composition is not configured"):
        srv.main()
