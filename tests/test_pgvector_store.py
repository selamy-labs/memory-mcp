"""Tests for PgVectorStore using a fake connection (no libpq, no database).

The fake records every executed (sql, params) pair and returns canned rows, so
the SQL shape, parameter ordering, scope/type filtering, idempotent upsert, and
row coercion are all exercised offline.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from memory_mcp.pgvector_store import (
    PgVectorStore,
    _vector_literal,
    build_pg_store,
    build_pg_store_from_params,
)
from memory_mcp.vector_store import MemoryRecord

_NOW = datetime(2026, 6, 23, tzinfo=timezone.utc)


class FakeCursor:
    def __init__(self, conn: FakeConnection) -> None:
        self._conn = conn

    def __enter__(self) -> FakeCursor:
        return self

    def __exit__(self, *exc) -> None:
        return None

    def execute(self, sql, params=None) -> None:
        self._conn.executed.append((sql, tuple(params) if params is not None else ()))

    def fetchone(self):
        return self._conn.next_one

    def fetchall(self):
        return self._conn.next_all


class FakeConnection:
    def __init__(self) -> None:
        self.executed: list[tuple[str, tuple]] = []
        self.commits = 0
        self.next_one = None
        self.next_all: list[tuple] = []

    def __enter__(self) -> FakeConnection:
        return self

    def __exit__(self, *exc) -> None:
        return None

    def cursor(self) -> FakeCursor:
        return FakeCursor(self)

    def commit(self) -> None:
        self.commits += 1


def _store(conn: FakeConnection, *, dim: int = 4) -> PgVectorStore:
    return PgVectorStore(lambda: conn, dim=dim)


def _record(*, group_id="fleet", name="a", dim=4) -> MemoryRecord:
    return MemoryRecord(
        group_id=group_id,
        name=name,
        type="feedback",
        description="desc",
        body="body text here",
        embedding=[0.5] * dim,
        updated_at=_NOW,
    )


def test_vector_literal_renders_bracketed_csv():
    assert _vector_literal([0.1, 0.2]) == "[0.1,0.2]"


def test_invalid_table_and_dim_rejected():
    with pytest.raises(ValueError, match="dim must be >= 1"):
        PgVectorStore(lambda: None, dim=0)
    with pytest.raises(ValueError, match="invalid table name"):
        PgVectorStore(lambda: None, dim=4, table="bad table")


def test_ensure_schema_runs_extension_table_indexes_and_commits():
    conn = FakeConnection()
    _store(conn).ensure_schema()
    sql_blob = "\n".join(sql for sql, _ in conn.executed)
    assert "CREATE EXTENSION IF NOT EXISTS vector" in sql_blob
    assert "CREATE TABLE IF NOT EXISTS memories" in sql_blob
    assert "vector(4)" in sql_blob
    assert "USING hnsw (embedding vector_cosine_ops)" in sql_blob
    assert conn.commits == 1


def test_upsert_uses_on_conflict_and_passes_vector_and_tokens():
    conn = FakeConnection()
    _store(conn).upsert(_record())
    sql, params = conn.executed[0]
    assert "ON CONFLICT (group_id, name) DO UPDATE" in sql
    # tokens param (index 5) is the space-joined token string; embedding (6) is the literal.
    assert params[5] == "a desc body text here"
    assert params[6] == "[0.5,0.5,0.5,0.5]"
    assert conn.commits == 1


def test_upsert_rejects_dim_mismatch():
    conn = FakeConnection()
    with pytest.raises(ValueError, match="dim mismatch"):
        _store(conn, dim=3).upsert(_record(dim=4))


def test_get_returns_none_when_missing():
    conn = FakeConnection()
    conn.next_one = None
    assert _store(conn).get("fleet", "nope") is None


def test_get_coerces_row_to_record():
    conn = FakeConnection()
    conn.next_one = ("fleet", "a", "feedback", "desc", "body", "[0.1,0.2,0.3,0.4]", _NOW)
    rec = _store(conn).get("fleet", "a")
    assert rec.name == "a"
    assert rec.embedding == [0.1, 0.2, 0.3, 0.4]
    assert rec.updated_at == _NOW


def test_get_coerces_naive_datetime_and_iso_string_and_list_embedding():
    conn = FakeConnection()
    conn.next_one = ("fleet", "a", "feedback", "desc", "body", [0.1, 0.2, 0.3, 0.4], datetime(2026, 1, 1))
    rec = _store(conn).get("fleet", "a")
    assert rec.updated_at.tzinfo == timezone.utc
    assert rec.embedding == [0.1, 0.2, 0.3, 0.4]

    conn.next_one = ("fleet", "b", "feedback", "desc", "body", "[]", "2026-02-02T00:00:00Z")
    rec2 = _store(conn).get("fleet", "b")
    assert rec2.embedding == []
    assert rec2.updated_at == datetime(2026, 2, 2, tzinfo=timezone.utc)


def test_count_all_and_by_scope():
    conn = FakeConnection()
    conn.next_one = (7,)
    assert _store(conn).count() == 7
    assert "WHERE" not in conn.executed[-1][0]
    assert _store(conn).count("trading") == 7
    sql, params = conn.executed[-1]
    assert "WHERE group_id = %s" in sql and params == ("trading",)


def test_count_zero_when_no_row():
    conn = FakeConnection()
    conn.next_one = None
    assert _store(conn).count() == 0


def test_search_empty_scopes_short_circuits():
    conn = FakeConnection()
    assert _store(conn).search([], [0.1] * 4, "q") == []
    assert conn.executed == []


def test_search_builds_blended_sql_with_scopes_type_and_limit():
    conn = FakeConnection()
    conn.next_all = [
        ("fleet", "a", "feedback", "desc", "body", "[0.5,0.5,0.5,0.5]", _NOW, 0.9, 0.8, 1.0, 1.5),
    ]
    store = _store(conn)
    hits = store.search(["fleet", "infra"], [0.5] * 4, "body text", limit=5, type="feedback", now=_NOW)
    sql, params = conn.executed[0]
    assert "embedding <=>" in sql
    assert "group_id IN (%s, %s)" in sql
    assert "AND type = %s" in sql
    assert sql.strip().endswith("LIMIT %s")
    # scopes, type, and limit appear at the tail of the params in order.
    assert params[-4:] == ("fleet", "infra", "feedback", 5)
    assert len(hits) == 1
    assert hits[0].score == pytest.approx(1.5)
    assert hits[0].semantic == pytest.approx(0.9)


def test_search_drops_nonpositive_scores():
    conn = FakeConnection()
    conn.next_all = [
        ("fleet", "a", "feedback", "d", "b", "[0.5,0.5,0.5,0.5]", _NOW, 0.0, 0.0, 0.0, 0.0),
    ]
    assert _store(conn).search(["fleet"], [0.5] * 4, "no overlap", now=_NOW) == []


def test_search_without_query_tokens_uses_constant_keyword():
    conn = FakeConnection()
    conn.next_all = []
    _store(conn).search(["fleet"], [0.5] * 4, "!!!", now=_NOW)  # query tokenizes to nothing
    sql, _ = conn.executed[0]
    assert "0 AS keyword" in sql or "+ 0.5 * 0)" in sql  # keyword expr collapsed to constant 0


def test_build_pg_store_uses_injected_psycopg_connect(monkeypatch):
    import sys
    import types

    conn = FakeConnection()
    conn.next_one = (3,)
    fake_psycopg = types.ModuleType("psycopg")
    calls: list[str] = []

    def fake_connect(dsn):
        calls.append(dsn)
        return conn

    fake_psycopg.connect = fake_connect
    monkeypatch.setitem(sys.modules, "psycopg", fake_psycopg)

    store = build_pg_store("postgresql://u:p@host/db", dim=4, table="memories")
    # Exercise the connect() closure end-to-end via a real store call.
    assert store.count() == 3
    assert calls == ["postgresql://u:p@host/db"]


def test_build_pg_store_from_params_passes_special_char_password(monkeypatch):
    import sys
    import types

    captured = {}
    fake = types.ModuleType("psycopg")

    def fake_connect(**kwargs):
        captured.update(kwargs)
        conn = FakeConnection()
        conn.next_one = (1,)
        return conn

    fake.connect = fake_connect
    monkeypatch.setitem(sys.modules, "psycopg", fake)

    # A base64-style password with URL-unsafe chars must pass through verbatim.
    store = build_pg_store_from_params(
        host="memory-mcp-postgres.memory.svc",
        user="memory",
        password="aB/cd+ef=gh",
        dbname="memory",
        port=5432,
        dim=4,
    )
    assert store.count() == 1
    assert captured == {
        "host": "memory-mcp-postgres.memory.svc",
        "port": 5432,
        "user": "memory",
        "password": "aB/cd+ef=gh",  # verbatim — never URL-encoded/misparsed
        "dbname": "memory",
    }


def test_build_pg_store_from_params_without_psycopg_raises(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "psycopg":
            raise ModuleNotFoundError("no psycopg")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(SystemExit, match="requires the 'psycopg' package"):
        build_pg_store_from_params(host="h", user="u", password="p", dbname="d", dim=4)


def test_build_pg_store_without_psycopg_raises_systemexit(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "psycopg":
            raise ModuleNotFoundError("no psycopg")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(SystemExit, match="requires the 'psycopg' package"):
        build_pg_store("postgresql://x", dim=4)
