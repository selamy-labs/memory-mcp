"""The shared semantic memory: add / search / get over a scoped vector store.

This is the Phase-1 shared-fleet-memory core (design of record:
``~/.local/share/life/architecture/shared-fleet-memory-graphiti.md`` -- the cheap
pgvector phase). It composes an injected :class:`~memory_mcp.embeddings.Embedder`
with an injected :class:`~memory_mcp.vector_store.VectorStore`, so the whole path
runs offline in tests (hashing embedder + in-RAM store) and in production
(real embedder + pgvector) with no code change.

Markdown-in-git stays the source of truth; this index is rebuildable from it.
Accordingly:

* ``add_memory`` indexes a memory into a scope. It is what the markdown indexer
  calls per file and what an agent calls opportunistically. It is idempotent on
  ``(group_id, name)`` -- re-adding replaces the record, so a full re-index from
  git converges rather than duplicating.
* ``search_memory`` blends semantic + recency + keyword signals within one or
  more scopes (always including the shared ``fleet`` scope unless asked
  otherwise), so an agent sees both the common ground and its domain.
* ``get_memory`` returns one record's verbatim body by ``(scope, name)``.

Recency is seeded from the memory's own date (``updated_at``), not the index
time, so importing the existing corpus does not make every old memory look new.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from memory_mcp.embeddings import Embedder
from memory_mcp.vector_store import FLEET_SCOPE, MemoryRecord, VectorStore

# A scope id is a short namespace label: the shared "fleet" plus per-domain
# scopes like "trading" / "infra" / "matchpoint" / "career". Keep it a simple
# slug so it maps cleanly to a group_id column and a directory name.
_MAX_SCOPE_LEN = 64


class SemanticMemoryError(Exception):
    """An add/search/get request failed for an expected, user-facing reason."""


def _validate_scope(group_id: str) -> str:
    cleaned = group_id.strip()
    if not cleaned:
        raise SemanticMemoryError("group_id (scope) must not be empty")
    if len(cleaned) > _MAX_SCOPE_LEN:
        raise SemanticMemoryError(f"group_id too long: {len(cleaned)} > {_MAX_SCOPE_LEN}")
    if any(ch.isspace() for ch in cleaned):
        raise SemanticMemoryError("group_id must not contain whitespace")
    return cleaned


def _coerce_updated_at(value: datetime | str | None) -> datetime:
    """Resolve a recency anchor to a UTC-aware datetime, defaulting to now.

    Accepts a datetime or an ISO string (a memory frontmatter date); a naive
    datetime is assumed UTC. ``None`` means "no date known" -> now, which is only
    correct for a freshly written memory, not a historical import (the indexer
    always passes the frontmatter date).
    """
    if value is None:
        return datetime.now(timezone.utc)
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    from memory_mcp.vector_store import _parse_iso  # local import: shared parser, avoid cycle at module top

    try:
        return _parse_iso(value)
    except ValueError as error:
        raise SemanticMemoryError(f"invalid updated_at {value!r}: {error}") from error


class SemanticMemory:
    """Add / search / get over a scope-namespaced semantic store."""

    def __init__(self, embedder: Embedder, store: VectorStore) -> None:
        self._embedder = embedder
        self._store = store

    def add_memory(
        self,
        name: str,
        description: str,
        type: str,
        body: str,
        *,
        group_id: str = FLEET_SCOPE,
        updated_at: datetime | str | None = None,
    ) -> dict[str, Any]:
        """Index one memory into ``group_id``; idempotent on ``(group_id, name)``.

        The embedding is computed over the memory's full text (name +
        description + body) so search matches against everything an author wrote,
        not just a title. Returns the indexed identity and its recency anchor.
        """
        scope = _validate_scope(group_id)
        clean_name = name.strip()
        if not clean_name:
            raise SemanticMemoryError("name must not be empty")
        clean_description = description.strip()
        if not clean_description:
            raise SemanticMemoryError("description must not be empty")
        clean_type = type.strip()
        if not clean_type:
            raise SemanticMemoryError("type must not be empty")
        when = _coerce_updated_at(updated_at)

        embedding = self._embedder.embed(f"{clean_name}\n{clean_description}\n{body}")
        record = MemoryRecord(
            group_id=scope,
            name=clean_name,
            type=clean_type,
            description=clean_description,
            body=body,
            embedding=embedding,
            updated_at=when,
        )
        self._store.upsert(record)
        return {
            "group_id": scope,
            "name": clean_name,
            "type": clean_type,
            "updated_at": when.astimezone(timezone.utc).isoformat().replace("+00:00", "Z"),
            "indexed": True,
        }

    def search_memory(
        self,
        query: str,
        *,
        group_ids: list[str] | None = None,
        include_fleet: bool = True,
        type: str | None = None,
        limit: int = 10,
        now: datetime | None = None,
    ) -> dict[str, Any]:
        """Blended semantic + recency + keyword search across one or more scopes.

        By default searches the shared ``fleet`` scope. Pass ``group_ids`` to add
        domain scopes; ``include_fleet`` keeps the shared scope in the search so
        an agent always sees common ground alongside its domain. Returns ranked
        hits, each carrying the blended score and its component signals.
        """
        clean_query = query.strip()
        if not clean_query:
            raise SemanticMemoryError("query must not be empty")
        limit = self._coerce_limit(limit)

        scopes: list[str] = []
        for raw in group_ids or []:
            scopes.append(_validate_scope(raw))
        if include_fleet and FLEET_SCOPE not in scopes:
            scopes.append(FLEET_SCOPE)
        if not scopes:
            raise SemanticMemoryError("no scope to search: pass group_ids or keep include_fleet=True")

        query_embedding = self._embedder.embed(clean_query)
        hits = self._store.search(
            scopes,
            query_embedding,
            clean_query,
            limit=limit,
            type=type,
            now=now,
        )
        return {
            "query": clean_query,
            "scopes": scopes,
            "count": len(hits),
            "hits": [hit.to_view() for hit in hits],
        }

    def get_memory(self, name: str, *, group_id: str = FLEET_SCOPE) -> dict[str, Any]:
        """Return one indexed memory's verbatim record by ``(group_id, name)``."""
        scope = _validate_scope(group_id)
        clean_name = name.strip()
        if not clean_name:
            raise SemanticMemoryError("name must not be empty")
        record = self._store.get(scope, clean_name)
        if record is None:
            raise SemanticMemoryError(f"no such memory {clean_name!r} in scope {scope!r}")
        return record.to_view()

    @staticmethod
    def _coerce_limit(limit: int) -> int:
        try:
            value = int(limit)
        except (TypeError, ValueError) as error:
            raise SemanticMemoryError(f"invalid limit {limit!r}: must be an integer") from error
        if value < 1:
            raise SemanticMemoryError("limit must be >= 1")
        return min(value, 100)
