"""Vector storage for the shared semantic memory.

A *record* is one indexed memory: its scope (``group_id``), name, type, a short
description, the raw markdown body kept verbatim for exact retrieval, an
embedding vector, and a recency timestamp seeded from the memory's frontmatter
date (NOT the index time -- old memories must not all look brand-new). Search
blends three signals so recall is good without a learned reranker:

* **semantic** -- cosine similarity between the query vector and each record's
  vector (the records are stored L2-normalised, so this is a dot product);
* **recency** -- an exponential decay on age, so when two memories are about
  equally relevant the fresher one wins;
* **keyword** -- token overlap between the query and the record's text, a small
  bonus that rescues exact-term matches a hashing embedder might under-weight.

Two implementations:

* :class:`InMemoryVectorStore` -- a list of records with the ranking done in
  Python. No database, no network: the full add/search/get path runs offline in
  tests and is also a legitimate tiny single-process backend.
* :class:`PgVectorStore` -- the production backend on Postgres + the ``pgvector``
  extension. It is constructed with an injected connection factory so the SQL is
  exercised against a real Postgres in integration tests while unit tests stay
  on the in-memory store. The semantic ordering uses pgvector's ``<=>`` cosine
  distance operator; recency + keyword blending happens in SQL.

Scope (``group_id``) namespaces every record: a search in scope ``trading`` never
sees ``matchpoint`` memories, while the shared ``fleet`` scope is the common
ground all agents read and write. The blend weights are constructor knobs so the
ranking is tunable without code changes (feedback-runtime-knobs-no-restart).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Protocol

from memory_mcp.embeddings import tokenize

# The shared scope every agent and the orchestrator read/write by default.
FLEET_SCOPE = "fleet"

# Half-life (in days) of the recency signal: a memory this old contributes half
# the recency score of a brand-new one. ~120d keeps a quarter's worth of context
# "fresh" while still letting newer facts win ties.
DEFAULT_RECENCY_HALFLIFE_DAYS = 120.0


def _parse_iso(value: str) -> datetime:
    """Parse an ISO-8601 timestamp (tolerating a trailing ``Z``) as UTC-aware."""
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


@dataclass(frozen=True)
class MemoryRecord:
    """One indexed memory in the vector store.

    ``updated_at`` is the recency anchor, seeded from the memory's frontmatter
    date at index time so historical memories keep their true age. ``body`` is
    the verbatim markdown chunk, retained so retrieval can return the exact
    source text rather than a lossy reconstruction.
    """

    group_id: str
    name: str
    type: str
    description: str
    body: str
    embedding: list[float]
    updated_at: datetime

    def to_view(self) -> dict[str, Any]:
        return {
            "group_id": self.group_id,
            "name": self.name,
            "type": self.type,
            "description": self.description,
            "body": self.body,
            "updated_at": self.updated_at.astimezone(timezone.utc).isoformat().replace("+00:00", "Z"),
        }


@dataclass(frozen=True)
class ScoredRecord:
    """A search hit with its blended score and the component signals."""

    record: MemoryRecord
    score: float
    semantic: float
    recency: float
    keyword: float

    def to_view(self) -> dict[str, Any]:
        view = self.record.to_view()
        view.update(
            {
                "score": round(self.score, 6),
                "semantic": round(self.semantic, 6),
                "recency": round(self.recency, 6),
                "keyword": round(self.keyword, 6),
            }
        )
        return view


def _cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity of two vectors, clamped to ``[0, 1]`` for ranking.

    Records and queries are stored normalised, so this is a dot product; the
    clamp drops the (rare, hashing-trick) negative lobe so the semantic signal
    composes cleanly with the non-negative recency and keyword signals.
    """
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    return max(0.0, min(1.0, dot))


def _recency_score(updated_at: datetime, now: datetime, halflife_days: float) -> float:
    """Exponential-decay recency in ``[0, 1]``: 1.0 now, 0.5 at one half-life."""
    age_days = max(0.0, (now - updated_at).total_seconds() / 86_400.0)
    return math.pow(0.5, age_days / halflife_days)


def _keyword_score(query_tokens: set[str], record: MemoryRecord) -> float:
    """Fraction of query tokens present in the record's text, in ``[0, 1]``."""
    if not query_tokens:
        return 0.0
    haystack = set(tokenize(f"{record.name} {record.description} {record.body}"))
    return len(query_tokens & haystack) / len(query_tokens)


class VectorStore(Protocol):
    """Scope-namespaced storage + blended search over memory records."""

    def upsert(self, record: MemoryRecord) -> None: ...

    def get(self, group_id: str, name: str) -> MemoryRecord | None: ...

    def search(
        self,
        group_ids: list[str],
        query_embedding: list[float],
        query_text: str,
        *,
        limit: int = 10,
        type: str | None = None,
        now: datetime | None = None,
    ) -> list[ScoredRecord]: ...

    def count(self, group_id: str | None = None) -> int: ...


class InMemoryVectorStore:
    """A list-backed vector store: ranking in Python, no DB, fully offline.

    ``(group_id, name)`` is the identity, so re-indexing a memory replaces its
    record (idempotent index). Blend weights are constructor knobs so ranking is
    tunable without code changes.
    """

    def __init__(
        self,
        *,
        semantic_weight: float = 1.0,
        recency_weight: float = 0.3,
        keyword_weight: float = 0.5,
        recency_halflife_days: float = DEFAULT_RECENCY_HALFLIFE_DAYS,
    ) -> None:
        self._records: dict[tuple[str, str], MemoryRecord] = {}
        self._semantic_weight = semantic_weight
        self._recency_weight = recency_weight
        self._keyword_weight = keyword_weight
        self._halflife = recency_halflife_days

    def upsert(self, record: MemoryRecord) -> None:
        self._records[(record.group_id, record.name)] = record

    def get(self, group_id: str, name: str) -> MemoryRecord | None:
        return self._records.get((group_id, name))

    def count(self, group_id: str | None = None) -> int:
        if group_id is None:
            return len(self._records)
        return sum(1 for key in self._records if key[0] == group_id)

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
        when = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
        query_tokens = set(tokenize(query_text))
        scope = set(group_ids)

        scored: list[ScoredRecord] = []
        for record in self._records.values():
            if record.group_id not in scope:
                continue
            if type is not None and record.type != type:
                continue
            semantic = _cosine(query_embedding, record.embedding)
            recency = _recency_score(record.updated_at, when, self._halflife)
            keyword = _keyword_score(query_tokens, record)
            score = self._semantic_weight * semantic + self._recency_weight * recency + self._keyword_weight * keyword
            if score <= 0.0:
                continue
            scored.append(ScoredRecord(record=record, score=score, semantic=semantic, recency=recency, keyword=keyword))

        scored.sort(key=lambda hit: (-hit.score, -hit.record.updated_at.timestamp(), hit.record.name))
        return scored[:limit]
