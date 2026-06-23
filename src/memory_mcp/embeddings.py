"""Embedding providers for the semantic memory layer.

The semantic store turns a memory's text into a fixed-length vector and ranks by
cosine similarity. *Which* model produces that vector is an injected detail so
the whole semantic path runs offline in tests and so the real provider (and its
credentials) is resolved only at runtime, never image-baked.

Two implementations ship here:

* :class:`HashingEmbedder` -- a deterministic, dependency-free embedder that maps
  text to a vector with the hashing trick (feature hashing over word tokens). It
  needs no network and no API key, so it is the default for tests and for a
  cheap, fully self-hosted Phase-1 deployment. Same text -> same vector; similar
  text (shared tokens) -> nearby vectors. It is **not** as good as a learned
  model, but it makes the semantic *plumbing* real and testable end to end.
* :class:`OpenAIEmbedder` -- a thin adapter over an injected HTTP transport that
  calls an OpenAI-compatible ``/embeddings`` endpoint (OpenAI, or any
  compatible gateway such as the fleet's OpenRouter facade). The API key and
  base URL are read at call time from injected config, never stored in the
  package. The transport is injected so this adapter is unit-tested with a stub
  (no real network).

Both return plain ``list[float]`` vectors; the vector store owns persistence and
similarity. ``EMBEDDING_DIM`` is the default dimension for the hashing embedder
and the schema; a real model overrides it via :attr:`Embedder.dim`.
"""

from __future__ import annotations

import hashlib
import math
import re
from typing import Any, Protocol

# Default vector dimension for the hashing embedder and the pgvector column.
# Chosen as a power of two large enough to keep token collisions low while small
# enough to stay cheap to store and compare. A real model (e.g. OpenAI
# text-embedding-3-small at 1536) sets its own ``dim``; the schema reads it from
# the embedder so the two never drift.
EMBEDDING_DIM = 512

_WORD_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Lowercase word/number tokens used by the keyword and hashing paths."""
    return _WORD_RE.findall(text.lower())


class Embedder(Protocol):
    """Turns text into a fixed-length dense vector.

    Implementations must be deterministic for a given input within a process so
    that re-indexing the same markdown yields a stable vector (idempotent index).
    """

    @property
    def dim(self) -> int: ...

    def embed(self, text: str) -> list[float]: ...


def _normalise(vector: list[float]) -> list[float]:
    """Scale a vector to unit length so cosine similarity is a plain dot product.

    A zero vector (empty text) is returned unchanged; the vector store treats a
    zero query as "no semantic signal" and falls back to recency/keyword.
    """
    norm = math.sqrt(sum(component * component for component in vector))
    if norm == 0.0:
        return vector
    return [component / norm for component in vector]


class HashingEmbedder:
    """Deterministic, dependency-free embedder via the hashing trick.

    Each token is hashed to a bucket in ``[0, dim)`` with a sign, and its count
    accumulates into that bucket; the vector is then L2-normalised. No model, no
    network, no credentials -- ideal for offline tests and a zero-cost default.
    Shared vocabulary between two texts pulls their vectors together, which is
    enough to make semantic-style ranking meaningfully better than substring
    matching for the recall use case.
    """

    def __init__(self, dim: int = EMBEDDING_DIM) -> None:
        if dim < 1:
            raise ValueError("embedding dim must be >= 1")
        self._dim = dim

    @property
    def dim(self) -> int:
        return self._dim

    def embed(self, text: str) -> list[float]:
        vector = [0.0] * self._dim
        for token in tokenize(text):
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            bucket = int.from_bytes(digest[:4], "big") % self._dim
            sign = 1.0 if digest[4] & 1 else -1.0
            vector[bucket] += sign
        return _normalise(vector)


class HttpTransport(Protocol):
    """Minimal POST transport injected into :class:`OpenAIEmbedder`.

    Returns the decoded JSON body. Injecting this keeps the embedder offline in
    tests (a stub returns canned vectors) and decouples it from any specific HTTP
    library.
    """

    def post_json(self, url: str, *, headers: dict[str, str], json: dict[str, Any]) -> dict[str, Any]: ...


class OpenAIEmbedder:
    """Adapter over an OpenAI-compatible ``/embeddings`` endpoint.

    The API key, base URL, model, and dimension are injected (resolved at runtime
    from the environment / a secret, never baked in). The transport is injected
    so the request shape is unit-tested without a real network call.
    """

    def __init__(
        self,
        *,
        transport: HttpTransport,
        api_key: str,
        model: str = "text-embedding-3-small",
        base_url: str = "https://api.openai.com/v1",
        dim: int = 1536,
    ) -> None:
        if not api_key:
            raise ValueError("OpenAIEmbedder requires an api_key")
        self._transport = transport
        self._api_key = api_key
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._dim = dim

    @property
    def dim(self) -> int:
        return self._dim

    def embed(self, text: str) -> list[float]:
        body = self._transport.post_json(
            f"{self._base_url}/embeddings",
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            json={"model": self._model, "input": text},
        )
        try:
            vector = [float(component) for component in body["data"][0]["embedding"]]
        except (KeyError, IndexError, TypeError, ValueError) as error:
            raise ValueError(f"unexpected embeddings response shape: {error}") from error
        if len(vector) != self._dim:
            raise ValueError(f"embedding dim mismatch: got {len(vector)}, expected {self._dim}")
        return _normalise(vector)
