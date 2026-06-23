"""Postgres + pgvector backend for the shared semantic memory.

This is the production :class:`~memory_mcp.vector_store.VectorStore`: a single
table of memory records keyed by ``(group_id, name)``, with the embedding stored
in a ``vector`` column and ranking done in SQL. It mirrors the in-memory store's
blended ranking -- semantic (pgvector cosine distance), recency (exponential
decay on age), keyword (token overlap) -- so swapping backends does not change
behaviour, only durability and scale.

Design constraints honoured:

* **Injected connection factory.** The store is built with a callable returning
  a DBAPI/psycopg connection. The real factory (``psycopg.connect`` against the
  in-cluster Postgres, credentials resolved at runtime) lives in
  :func:`build_pg_store`; unit tests pass a fake connection that records SQL and
  returns canned rows, so the SQL shape is exercised with **no libpq and no
  database**. ``psycopg`` is an optional ``[pg]`` extra, imported only inside the
  production factory -- never at module import, exactly like the ``mcp`` SDK.
* **Scope namespacing** via the ``group_id`` column; a search is filtered to the
  requested scopes with a parameterised ``IN`` list (no string interpolation of
  identifiers/values -- injection-safe).
* **Idempotent upsert** via ``INSERT ... ON CONFLICT (group_id, name) DO UPDATE``,
  so a full re-index from git converges.
* **Rebuildable.** ``ensure_schema`` is idempotent (``CREATE ... IF NOT
  EXISTS``); dropping the table and re-indexing rebuilds the store from git.

The vector dimension is taken from the embedder at build time so the column type
and the inserted vectors never drift.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any

from memory_mcp.embeddings import tokenize
from memory_mcp.vector_store import (
    DEFAULT_RECENCY_HALFLIFE_DAYS,
    MemoryRecord,
    ScoredRecord,
)

# A DBAPI-ish connection: ``cursor()`` context manager + ``commit()``. Kept as a
# loose callable type so a psycopg connection or a test fake both satisfy it.
ConnectionFactory = Callable[[], Any]

DEFAULT_TABLE = "memories"


def _vector_literal(vector: list[float]) -> str:
    """Render a vector as a pgvector text literal, e.g. ``[0.1,0.2]``.

    pgvector accepts its text input form for both storage and the ``<=>`` /
    ``<#>`` operators; rendering here keeps the value a bound parameter (passed as
    a string) rather than interpolated SQL.
    """
    return "[" + ",".join(repr(float(component)) for component in vector) + "]"


class PgVectorStore:
    """Durable vector store on Postgres + pgvector with blended SQL ranking."""

    def __init__(
        self,
        connect: ConnectionFactory,
        *,
        dim: int,
        table: str = DEFAULT_TABLE,
        semantic_weight: float = 1.0,
        recency_weight: float = 0.3,
        keyword_weight: float = 0.5,
        recency_halflife_days: float = DEFAULT_RECENCY_HALFLIFE_DAYS,
    ) -> None:
        if dim < 1:
            raise ValueError("dim must be >= 1")
        if not table.isidentifier():
            # The table name is an SQL identifier we interpolate; constrain it to
            # a safe identifier so it can never carry injection.
            raise ValueError(f"invalid table name: {table!r}")
        self._connect = connect
        self._dim = dim
        self._table = table
        self._semantic_weight = semantic_weight
        self._recency_weight = recency_weight
        self._keyword_weight = keyword_weight
        self._halflife = recency_halflife_days

    # -- schema ----------------------------------------------------------------

    def ensure_schema(self) -> None:
        """Create the pgvector extension, table, and indexes if absent (idempotent)."""
        statements = [
            "CREATE EXTENSION IF NOT EXISTS vector",
            (
                f"CREATE TABLE IF NOT EXISTS {self._table} ("
                "  group_id    text        NOT NULL,"
                "  name        text        NOT NULL,"
                "  type        text        NOT NULL,"
                "  description text        NOT NULL,"
                "  body        text        NOT NULL,"
                "  tokens      text        NOT NULL,"
                f"  embedding   vector({self._dim}) NOT NULL,"
                "  updated_at  timestamptz NOT NULL,"
                "  PRIMARY KEY (group_id, name)"
                ")"
            ),
            f"CREATE INDEX IF NOT EXISTS {self._table}_group_idx ON {self._table} (group_id)",
            (
                f"CREATE INDEX IF NOT EXISTS {self._table}_embedding_idx "
                f"ON {self._table} USING hnsw (embedding vector_cosine_ops)"
            ),
        ]
        with self._connect() as conn:
            with conn.cursor() as cur:
                for statement in statements:
                    cur.execute(statement)
            conn.commit()

    # -- writes ----------------------------------------------------------------

    def upsert(self, record: MemoryRecord) -> None:
        """Insert or replace a record by ``(group_id, name)`` (idempotent)."""
        if len(record.embedding) != self._dim:
            raise ValueError(f"embedding dim mismatch: got {len(record.embedding)}, expected {self._dim}")
        tokens = " ".join(tokenize(f"{record.name} {record.description} {record.body}"))
        sql = (
            f"INSERT INTO {self._table} "
            "(group_id, name, type, description, body, tokens, embedding, updated_at) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s) "
            "ON CONFLICT (group_id, name) DO UPDATE SET "
            "type = EXCLUDED.type, description = EXCLUDED.description, body = EXCLUDED.body, "
            "tokens = EXCLUDED.tokens, embedding = EXCLUDED.embedding, updated_at = EXCLUDED.updated_at"
        )
        params = (
            record.group_id,
            record.name,
            record.type,
            record.description,
            record.body,
            tokens,
            _vector_literal(record.embedding),
            record.updated_at.astimezone(timezone.utc),
        )
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
            conn.commit()

    # -- reads -----------------------------------------------------------------

    def get(self, group_id: str, name: str) -> MemoryRecord | None:
        sql = (
            f"SELECT group_id, name, type, description, body, embedding, updated_at "
            f"FROM {self._table} WHERE group_id = %s AND name = %s"
        )
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(sql, (group_id, name))
            row = cur.fetchone()
        if row is None:
            return None
        return self._row_to_record(row)

    def count(self, group_id: str | None = None) -> int:
        if group_id is None:
            sql = f"SELECT COUNT(*) FROM {self._table}"
            params: tuple[Any, ...] = ()
        else:
            sql = f"SELECT COUNT(*) FROM {self._table} WHERE group_id = %s"
            params = (group_id,)
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(sql, params)
            row = cur.fetchone()
        return int(row[0]) if row else 0

    def search(
        self,
        group_ids: list[str],
        query_embedding: list[float],
        query_text: str,
        *,
        limit: int = 10,
        type: str | None = None,
        now: datetime | None = None,
    ) -> list[ScoredRecord]:
        """Blended semantic + recency + keyword ranking, computed in SQL.

        ``1 - (embedding <=> query)`` is cosine similarity (pgvector's ``<=>`` is
        cosine *distance*); recency is ``0.5 ^ (age_days / halflife)`` via
        ``EXTRACT(EPOCH ...)``; keyword overlap is the parameterised fraction of
        query tokens present in the stored ``tokens`` column. The scopes and the
        optional type filter are parameterised; only the (validated) table name
        and weight constants are interpolated.
        """
        if not group_ids:
            return []
        when = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
        query_tokens = sorted(set(tokenize(query_text)))
        vec = _vector_literal(query_embedding)

        scope_placeholders = ", ".join(["%s"] * len(group_ids))
        keyword_expr, keyword_params = self._keyword_sql(query_tokens)

        # ts in seconds for the recency decay; ln(0.5)/halflife_seconds as rate.
        halflife_seconds = self._halflife * 86_400.0
        recency_expr = (
            f"EXP( (LEAST(0, EXTRACT(EPOCH FROM (updated_at - %s)))) * {math.log(2.0) / halflife_seconds!r} )"
        )
        semantic_expr = "(1 - (embedding <=> %s))"
        score_expr = (
            f"({self._semantic_weight!r} * {semantic_expr} "
            f"+ {self._recency_weight!r} * {recency_expr} "
            f"+ {self._keyword_weight!r} * {keyword_expr})"
        )

        sql_parts = [
            "SELECT group_id, name, type, description, body, embedding, updated_at,",
            f"  {semantic_expr} AS semantic,",
            f"  {recency_expr} AS recency,",
            f"  {keyword_expr} AS keyword,",
            f"  {score_expr} AS score",
            f"FROM {self._table}",
            f"WHERE group_id IN ({scope_placeholders})",
        ]
        # Parameter order must match the placeholders as they appear in the SQL.
        params: list[Any] = [vec, when]  # semantic_expr (SELECT), recency_expr (SELECT)
        params += keyword_params  # keyword_expr (SELECT)
        params += [vec, when]  # semantic + recency inside score_expr
        params += keyword_params  # keyword inside score_expr
        params += list(group_ids)  # WHERE scope IN
        if type is not None:
            sql_parts.append("AND type = %s")
            params.append(type)
        sql_parts.append("ORDER BY score DESC, updated_at DESC, name ASC")
        sql_parts.append("LIMIT %s")
        params.append(int(limit))
        sql = "\n".join(sql_parts)

        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(sql, tuple(params))
            rows = cur.fetchall()

        hits: list[ScoredRecord] = []
        for row in rows:
            record = self._row_to_record(row[:7])
            semantic, recency, keyword, score = (float(row[7]), float(row[8]), float(row[9]), float(row[10]))
            if score <= 0.0:
                continue
            hits.append(ScoredRecord(record=record, score=score, semantic=semantic, recency=recency, keyword=keyword))
        return hits

    # -- internals -------------------------------------------------------------

    def _keyword_sql(self, query_tokens: list[str]) -> tuple[str, list[Any]]:
        """SQL expression (and params) for the keyword-overlap signal in [0, 1].

        Counts how many of the query tokens appear as whole words in the stored
        space-joined ``tokens`` column, divided by the number of query tokens. A
        query with no tokens contributes a constant 0.
        """
        if not query_tokens:
            return "0", []
        # One LIKE term per query token, matched against the space-delimited
        # ``tokens`` column padded with spaces so a pattern '% token %' matches a
        # whole word only. Patterns are bound parameters (injection-safe).
        terms = ["(CASE WHEN (' ' || tokens || ' ') LIKE %s THEN 1 ELSE 0 END)" for _ in query_tokens]
        params: list[Any] = [f"% {token} %" for token in query_tokens]
        expr = f"(({' + '.join(terms)})::float / {len(query_tokens)!r})"
        return expr, params

    def _row_to_record(self, row: tuple[Any, ...]) -> MemoryRecord:
        group_id, name, mem_type, description, body, embedding, updated_at = row
        return MemoryRecord(
            group_id=group_id,
            name=name,
            type=mem_type,
            description=description,
            body=self._coerce_text(body),
            embedding=self._coerce_embedding(embedding),
            updated_at=self._coerce_dt(updated_at),
        )

    @staticmethod
    def _coerce_text(value: Any) -> str:
        return value if isinstance(value, str) else str(value)

    @staticmethod
    def _coerce_embedding(value: Any) -> list[float]:
        if isinstance(value, str):
            inner = value.strip().lstrip("[").rstrip("]")
            return [float(part) for part in inner.split(",")] if inner else []
        return [float(component) for component in value]

    @staticmethod
    def _coerce_dt(value: Any) -> datetime:
        if isinstance(value, datetime):
            return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
        from memory_mcp.vector_store import _parse_iso

        return _parse_iso(str(value))


def build_pg_store(dsn: str, *, dim: int, table: str = DEFAULT_TABLE) -> PgVectorStore:
    """Build a :class:`PgVectorStore` against a real Postgres via psycopg.

    ``psycopg`` is an optional ``[pg]`` extra imported only here, so the package
    has no hard libpq dependency and unit tests never need a database. The DSN
    (host/db/user/password) is resolved by the caller from the environment / a
    mounted secret at runtime -- never baked into the image.
    """
    try:
        import psycopg
    except ModuleNotFoundError as error:  # pragma: no cover - import guard
        raise SystemExit(
            "PgVectorStore requires the 'psycopg' package. Install it with: pip install 'memory-mcp[pg]'"
        ) from error

    def connect() -> Any:
        return psycopg.connect(dsn)

    return PgVectorStore(connect, dim=dim, table=table)
