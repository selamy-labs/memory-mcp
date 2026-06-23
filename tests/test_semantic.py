"""Tests for the SemanticMemory orchestrator (add / search / get + scoping)."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from memory_mcp.embeddings import HashingEmbedder
from memory_mcp.semantic import SemanticMemory, SemanticMemoryError, _coerce_updated_at
from memory_mcp.vector_store import FLEET_SCOPE, InMemoryVectorStore

_NOW = datetime(2026, 6, 23, tzinfo=timezone.utc)


def _memory() -> SemanticMemory:
    return SemanticMemory(HashingEmbedder(dim=256), InMemoryVectorStore())


def test_add_then_get_round_trips_body_verbatim():
    mem = _memory()
    mem.add_memory("prefer-wif", "Prefer WIF over SA keys", "feedback", "Use keyless OIDC.\n\nRelated: [[x]]")
    got = mem.get_memory("prefer-wif")
    assert got["name"] == "prefer-wif"
    assert got["type"] == "feedback"
    assert got["body"] == "Use keyless OIDC.\n\nRelated: [[x]]"
    assert got["group_id"] == FLEET_SCOPE


def test_add_is_idempotent_and_reports_recency():
    mem = _memory()
    result = mem.add_memory("a", "desc", "project", "body", updated_at="2026-01-01T00:00:00Z")
    assert result["indexed"] is True
    assert result["updated_at"] == "2026-01-01T00:00:00Z"
    mem.add_memory("a", "desc2", "project", "body2", updated_at="2026-02-01T00:00:00Z")
    assert mem.get_memory("a")["description"] == "desc2"


def test_add_to_domain_scope_is_isolated_from_fleet():
    mem = _memory()
    mem.add_memory("trade-secret", "kalshi edge", "project", "details", group_id="trading")
    assert mem.get_memory("trade-secret", group_id="trading")["name"] == "trade-secret"
    with pytest.raises(SemanticMemoryError, match="no such memory"):
        mem.get_memory("trade-secret")  # default fleet scope


def test_search_default_scope_is_fleet():
    mem = _memory()
    mem.add_memory("fleet-fact", "shared fleet knowledge", "reference", "everyone reads this")
    out = mem.search_memory("shared fleet knowledge", now=_NOW)
    assert out["scopes"] == [FLEET_SCOPE]
    assert out["count"] == 1
    assert out["hits"][0]["name"] == "fleet-fact"


def test_search_includes_fleet_alongside_domain_scope():
    mem = _memory()
    mem.add_memory("fleet-fact", "credential rotation policy", "feedback", "rotate keys")
    mem.add_memory("infra-fact", "credential rotation policy", "feedback", "rotate keys", group_id="infra")
    out = mem.search_memory("credential rotation", group_ids=["infra"], now=_NOW)
    assert set(out["scopes"]) == {"infra", FLEET_SCOPE}
    assert {hit["name"] for hit in out["hits"]} == {"fleet-fact", "infra-fact"}


def test_search_can_exclude_fleet():
    mem = _memory()
    mem.add_memory("fleet-fact", "alpha topic", "reference", "x")
    mem.add_memory("infra-fact", "alpha topic", "reference", "x", group_id="infra")
    out = mem.search_memory("alpha topic", group_ids=["infra"], include_fleet=False, now=_NOW)
    assert out["scopes"] == ["infra"]
    assert {hit["name"] for hit in out["hits"]} == {"infra-fact"}


def test_search_type_filter():
    mem = _memory()
    mem.add_memory("a", "gamma delta", "project", "body")
    mem.add_memory("b", "gamma delta", "feedback", "body")
    out = mem.search_memory("gamma delta", type="feedback", now=_NOW)
    assert [hit["name"] for hit in out["hits"]] == ["b"]


def test_search_with_no_scope_raises():
    mem = _memory()
    with pytest.raises(SemanticMemoryError, match="no scope to search"):
        mem.search_memory("anything", include_fleet=False, now=_NOW)


@pytest.mark.parametrize(
    "bad_call",
    [
        lambda m: m.add_memory("", "d", "t", "b"),
        lambda m: m.add_memory("n", "", "t", "b"),
        lambda m: m.add_memory("n", "d", "", "b"),
        lambda m: m.add_memory("n", "d", "t", "b", group_id="  "),
        lambda m: m.add_memory("n", "d", "t", "b", group_id="has space"),
        lambda m: m.add_memory("n", "d", "t", "b", group_id="x" * 65),
        lambda m: m.search_memory(""),
        lambda m: m.search_memory("q", limit=0),
        lambda m: m.search_memory("q", limit="nope"),
        lambda m: m.get_memory(""),
    ],
)
def test_validation_errors(bad_call):
    with pytest.raises(SemanticMemoryError):
        bad_call(_memory())


def test_search_limit_is_capped():
    mem = _memory()
    out = mem.search_memory("q", limit=10_000, now=_NOW)
    assert out["count"] == 0  # empty store, but no error -> limit coerced/capped


def test_coerce_updated_at_variants():
    assert _coerce_updated_at("2026-01-01T00:00:00Z").year == 2026
    naive = datetime(2026, 3, 1)
    assert _coerce_updated_at(naive).tzinfo == timezone.utc
    assert _coerce_updated_at(None).tzinfo == timezone.utc
    with pytest.raises(SemanticMemoryError, match="invalid updated_at"):
        _coerce_updated_at("not-a-date")
