"""Tests for the embedding providers (hashing + OpenAI-compatible adapter)."""

from __future__ import annotations

import math

import pytest

from memory_mcp.embeddings import (
    EMBEDDING_DIM,
    HashingEmbedder,
    OpenAIEmbedder,
    tokenize,
)


def _norm(vector: list[float]) -> float:
    return math.sqrt(sum(component * component for component in vector))


def test_tokenize_lowercases_and_splits_on_non_word():
    assert tokenize("Hello, WORLD-123!") == ["hello", "world", "123"]


def test_hashing_embedder_is_deterministic_and_unit_length():
    embedder = HashingEmbedder()
    first = embedder.embed("workload identity federation keyless")
    second = embedder.embed("workload identity federation keyless")
    assert first == second
    assert embedder.dim == EMBEDDING_DIM
    assert len(first) == EMBEDDING_DIM
    assert _norm(first) == pytest.approx(1.0, abs=1e-9)


def test_hashing_embedder_similar_text_is_closer_than_unrelated():
    embedder = HashingEmbedder(dim=256)
    base = embedder.embed("nash trading credentials secret manager")
    similar = embedder.embed("nash trading credentials backed up to secret manager")
    unrelated = embedder.embed("matchpoint wizard onboarding lovable port")

    def dot(a: list[float], b: list[float]) -> float:
        return sum(x * y for x, y in zip(a, b, strict=True))

    assert dot(base, similar) > dot(base, unrelated)


def test_hashing_embedder_empty_text_is_zero_vector():
    embedder = HashingEmbedder(dim=16)
    vector = embedder.embed("")
    assert vector == [0.0] * 16


def test_hashing_embedder_rejects_bad_dim():
    with pytest.raises(ValueError, match="dim must be >= 1"):
        HashingEmbedder(dim=0)


class _StubTransport:
    """Records the request and returns a canned embeddings response."""

    def __init__(self, embedding: list[float]) -> None:
        self._embedding = embedding
        self.calls: list[dict] = []

    def post_json(self, url, *, headers, json):
        self.calls.append({"url": url, "headers": headers, "json": json})
        return {"data": [{"embedding": self._embedding}]}


def test_openai_embedder_calls_endpoint_and_normalises():
    transport = _StubTransport([3.0, 4.0])
    embedder = OpenAIEmbedder(transport=transport, api_key="sk-test", dim=2, base_url="https://gw/v1/")
    vector = embedder.embed("hello")

    assert embedder.dim == 2
    assert vector == pytest.approx([0.6, 0.8])  # 3-4-5 triangle, normalised
    call = transport.calls[0]
    assert call["url"] == "https://gw/v1/embeddings"
    assert call["headers"]["Authorization"] == "Bearer sk-test"
    assert call["json"]["input"] == "hello"


def test_openai_embedder_no_auth_header_when_key_empty():
    # A self-hosted in-cluster model (TEI) needs no auth: no Authorization header.
    transport = _StubTransport([1.0, 0.0])
    embedder = OpenAIEmbedder(transport=transport, dim=2, base_url="http://tei.memory.svc/v1")
    embedder.embed("hi")
    assert "Authorization" not in transport.calls[0]["headers"]


def test_openai_embedder_sends_auth_header_when_key_present():
    transport = _StubTransport([1.0, 0.0])
    OpenAIEmbedder(transport=transport, api_key="sk-x", dim=2).embed("hi")
    assert transport.calls[0]["headers"]["Authorization"] == "Bearer sk-x"


def test_openai_embedder_rejects_dim_mismatch():
    embedder = OpenAIEmbedder(transport=_StubTransport([1.0, 2.0, 3.0]), api_key="k", dim=2)
    with pytest.raises(ValueError, match="dim mismatch"):
        embedder.embed("x")


def test_openai_embedder_rejects_bad_response_shape():
    class _Bad:
        def post_json(self, url, *, headers, json):
            return {"nope": True}

    embedder = OpenAIEmbedder(transport=_Bad(), api_key="k", dim=2)
    with pytest.raises(ValueError, match="unexpected embeddings response"):
        embedder.embed("x")
