"""Tests for the in-memory vector store ranking and scope namespacing."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from memory_mcp.embeddings import HashingEmbedder
from memory_mcp.vector_store import (
    FLEET_SCOPE,
    InMemoryVectorStore,
    MemoryRecord,
    _cosine,
    _keyword_score,
    _parse_iso,
    _recency_score,
)

_EMBEDDER = HashingEmbedder(dim=256)
_NOW = datetime(2026, 6, 23, tzinfo=timezone.utc)


def _record(name: str, text: str, *, group_id: str = FLEET_SCOPE, age_days: float = 0.0, type: str = "project"):
    return MemoryRecord(
        group_id=group_id,
        name=name,
        type=type,
        description=text,
        body=text,
        embedding=_EMBEDDER.embed(f"{name} {text}"),
        updated_at=_NOW - timedelta(days=age_days),
    )


def test_parse_iso_handles_z_and_naive():
    assert _parse_iso("2026-06-23T00:00:00Z") == datetime(2026, 6, 23, tzinfo=timezone.utc)
    assert _parse_iso("2026-06-23").tzinfo == timezone.utc


def test_cosine_zero_for_empty_or_mismatched():
    assert _cosine([], [1.0]) == 0.0
    assert _cosine([1.0, 0.0], [1.0]) == 0.0
    assert _cosine([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)


def test_keyword_score_zero_for_empty_query():
    assert _keyword_score(set(), _record("a", "anything at all")) == 0.0


def test_recency_score_halves_at_one_halflife():
    older = _NOW - timedelta(days=120)
    assert _recency_score(_NOW, _NOW, 120.0) == pytest.approx(1.0)
    assert _recency_score(older, _NOW, 120.0) == pytest.approx(0.5)


def test_upsert_is_idempotent_on_group_and_name():
    store = InMemoryVectorStore()
    store.upsert(_record("a", "first"))
    store.upsert(_record("a", "second"))
    assert store.count() == 1
    assert store.get(FLEET_SCOPE, "a").description == "second"


def test_get_returns_none_for_missing():
    store = InMemoryVectorStore()
    assert store.get(FLEET_SCOPE, "nope") is None


def test_count_by_scope():
    store = InMemoryVectorStore()
    store.upsert(_record("a", "x", group_id="fleet"))
    store.upsert(_record("b", "y", group_id="trading"))
    assert store.count() == 2
    assert store.count("trading") == 1


def test_search_respects_scope_isolation():
    store = InMemoryVectorStore()
    store.upsert(_record("fleet-mem", "shared knowledge", group_id="fleet"))
    store.upsert(_record("trade-mem", "shared knowledge", group_id="trading"))
    hits = store.search(["trading"], _EMBEDDER.embed("shared knowledge"), "shared knowledge", now=_NOW)
    names = {hit.record.name for hit in hits}
    assert names == {"trade-mem"}


def test_search_filters_by_type():
    store = InMemoryVectorStore()
    store.upsert(_record("p", "alpha beta", type="project"))
    store.upsert(_record("f", "alpha beta", type="feedback"))
    hits = store.search([FLEET_SCOPE], _EMBEDDER.embed("alpha beta"), "alpha beta", type="feedback", now=_NOW)
    assert [hit.record.name for hit in hits] == ["f"]


def test_search_recency_breaks_a_semantic_tie():
    store = InMemoryVectorStore(semantic_weight=1.0, recency_weight=0.5, keyword_weight=0.0)
    store.upsert(_record("fresh", "identical content here", age_days=0))
    store.upsert(_record("stale", "identical content here", age_days=300))
    hits = store.search([FLEET_SCOPE], _EMBEDDER.embed("identical content here"), "identical content here", now=_NOW)
    assert next(hit.record.name for hit in hits) == "fresh"


def test_search_keyword_signal_rescues_exact_terms():
    store = InMemoryVectorStore(semantic_weight=0.0, recency_weight=0.0, keyword_weight=1.0)
    store.upsert(_record("has-term", "the falkordb graph database"))
    store.upsert(_record("no-term", "completely different words"))
    hits = store.search([FLEET_SCOPE], _EMBEDDER.embed("falkordb"), "falkordb", now=_NOW)
    assert hits[0].record.name == "has-term"
    assert hits[0].keyword == pytest.approx(1.0)


def test_search_drops_zero_score_records_and_limits():
    store = InMemoryVectorStore(semantic_weight=0.0, recency_weight=0.0, keyword_weight=1.0)
    store.upsert(_record("match", "tunnel boring machine"))
    store.upsert(_record("miss", "unrelated text"))
    hits = store.search([FLEET_SCOPE], _EMBEDDER.embed("tunnel"), "tunnel", limit=5, now=_NOW)
    assert [hit.record.name for hit in hits] == ["match"]


def test_search_defaults_now_to_wall_clock():
    store = InMemoryVectorStore()
    store.upsert(_record("a", "wall clock default path"))
    hits = store.search([FLEET_SCOPE], _EMBEDDER.embed("wall clock"), "wall clock")
    assert hits and hits[0].record.name == "a"


def test_scored_record_view_has_components():
    store = InMemoryVectorStore()
    store.upsert(_record("a", "viewable scored record"))
    hit = store.search([FLEET_SCOPE], _EMBEDDER.embed("viewable scored"), "viewable scored", now=_NOW)[0]
    view = hit.to_view()
    assert set(view) >= {"name", "group_id", "score", "semantic", "recency", "keyword", "updated_at"}
    assert view["updated_at"].endswith("Z")
