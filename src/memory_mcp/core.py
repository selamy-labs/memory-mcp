"""The memory store core: read and careful write over the markdown-memory store.

This module holds the store logic exactly once. The MCP server in
:mod:`memory_mcp.mcp_server` is a thin wrapper that serialises these structured
results to JSON; nothing here imports the MCP SDK.

The capability is narrow on purpose: read (search/get/list/index) and *careful*
write (create/update one memory and keep the ``MEMORY.md`` index consistent).
The persist/recall **methodology** -- when to save, what to dedupe, how to phrase
a memory -- stays a skill; this server is only the read/write *call*.

Safety model for writes
------------------------
* **Validated names.** A name must be a kebab-case slug; it is resolved to a
  single ``<name>.md`` inside the root (see :func:`memory_mcp.storage.safe_relpath`).
  A name that escapes the root is rejected before any byte is written.
* **No clobber across identities.** ``memory_write`` refuses to overwrite an
  existing file (use ``memory_update`` to change one). Both write the same file
  the name maps to and never touch any other memory's file.
* **Index stays consistent.** Every write add/replaces exactly one line in
  ``MEMORY.md`` for that memory -- never duplicating an existing pointer.
* **Idempotent.** Writing the same content twice yields the same file and the
  same single index line.

All file access goes through the injected :class:`memory_mcp.storage.Storage`
and timing through the injected clock, so the full path runs offline in tests on
a temp/in-RAM root.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from memory_mcp.document import VALID_TYPES, DocumentError, Memory, parse, serialise, to_view
from memory_mcp.storage import Clock, Storage, StorageError, SystemClock, safe_relpath

INDEX_FILENAME = "MEMORY.md"

# A memory name is a kebab-case slug: lowercase, digit, dash; no leading/trailing
# dash. This is stricter than the path-safety check and matches how the store
# names files, so a created memory always has a clean, linkable slug.
_NAME_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")

# Matches an existing index pointer line for a given file so updates replace it
# in place rather than appending a duplicate. The link target is the filename.
_DESCRIPTION_MAX = 2_000

# Word characters used to tokenise a query for ranked search.
_WORD_RE = re.compile(r"[a-z0-9]+")


class MemoryError(Exception):
    """A memory request failed for an expected, user-facing reason.

    The MCP layer maps this to a ``ToolError`` so clients get a clean message
    instead of a stack trace.
    """


def _validate_name(name: str) -> str:
    cleaned = name.strip()
    if not cleaned:
        raise MemoryError("name must not be empty")
    if not _NAME_RE.match(cleaned):
        raise MemoryError(
            f"invalid name {name!r}: must be a kebab-case slug "
            "(lowercase letters, digits, single dashes; e.g. 'my-memory-name')"
        )
    return cleaned


def _validate_type(mem_type: str) -> str:
    cleaned = mem_type.strip()
    if cleaned not in VALID_TYPES:
        raise MemoryError(f"invalid type {mem_type!r}: must be one of {', '.join(VALID_TYPES)}")
    return cleaned


def _validate_description(description: str) -> str:
    cleaned = description.strip()
    if not cleaned:
        raise MemoryError("description must not be empty")
    if "\n" in cleaned:
        raise MemoryError("description must be a single line")
    if len(cleaned) > _DESCRIPTION_MAX:
        raise MemoryError(f"description too long: {len(cleaned)} > {_DESCRIPTION_MAX} characters")
    return cleaned


def _index_line(name: str, description: str) -> str:
    """One MEMORY.md pointer line: ``- [name.md](name.md) — <description>``.

    The link target and text are the filename, matching the real store's index
    convention, so the pointer is stable and de-duplicatable by filename.
    """
    return f"- [{name}.md]({name}.md) — {description}"


def _link_body(body: str, links: list[str] | None) -> str:
    """Append a ``Related:`` line of ``[[name]]`` links if any are supplied.

    Links are a convenience for the writer; they are rendered into the body as
    the store's wiki-link convention and not stored separately.
    """
    if not links:
        return body
    refs = " ".join(f"[[{link}]]" for link in links)
    trimmed = body.rstrip("\n")
    separator = "\n\n" if trimmed else ""
    return f"{trimmed}{separator}Related: {refs}"


@dataclass(frozen=True)
class SearchHit:
    """One ranked search result: where it is and why it matched."""

    name: str
    description: str
    type: str
    path: str
    score: int

    def to_view(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "type": self.type,
            "path": self.path,
            "score": self.score,
        }


class MemoryStore:
    """Read and careful-write access to a markdown-memory root directory.

    Construct with an injected :class:`Storage` (a temp/in-RAM root in tests, a
    real directory in production). All reads list and parse the root's ``.md``
    files; all writes validate, write a single file, and reconcile the index.
    """

    def __init__(self, storage: Storage, *, clock: Clock | None = None) -> None:
        self._storage = storage
        self._clock = clock or SystemClock()

    # -- reads -----------------------------------------------------------------

    def search(self, query: str, *, type: str | None = None, limit: int = 10) -> dict[str, Any]:
        """Rank memories against ``query`` over name + description + body.

        Scoring rewards the rarer, more specific signals: a query term in the
        ``name`` outweighs one in the ``description``, which outweighs the body;
        a phrase match on the whole query is a strong bonus. Optionally filter by
        ``type``. Returns the top ``limit`` hits, highest score first.
        """
        cleaned_query = query.strip()
        if not cleaned_query:
            raise MemoryError("query must not be empty")
        type_filter = _validate_type(type) if type is not None else None
        limit = self._coerce_limit(limit)

        terms = set(_WORD_RE.findall(cleaned_query.lower()))
        phrase = cleaned_query.lower()

        hits: list[SearchHit] = []
        for filename in self._storage.list_md():
            if filename == INDEX_FILENAME:
                continue
            memory = self._try_parse(filename)
            if memory is None:
                continue
            if type_filter is not None and memory.type != type_filter:
                continue
            score = self._score(memory, terms, phrase)
            if score <= 0:
                continue
            hits.append(
                SearchHit(
                    name=memory.name,
                    description=memory.description,
                    type=memory.type,
                    path=filename,
                    score=score,
                )
            )

        hits.sort(key=lambda hit: (-hit.score, hit.name))
        return {
            "query": cleaned_query,
            "count": min(len(hits), limit),
            "total_matched": len(hits),
            "hits": [hit.to_view() for hit in hits[:limit]],
        }

    def get(self, name: str) -> dict[str, Any]:
        """Return the full frontmatter + body view of one memory by name."""
        name = _validate_name(name)
        relpath = safe_relpath(name)
        if not self._storage.exists(relpath):
            raise MemoryError(f"no such memory: {name!r}")
        memory = parse(self._storage.read(relpath))
        return to_view(memory, name=name, path=relpath)

    def list(self, *, type: str | None = None) -> dict[str, Any]:
        """List every memory (optionally filtered by ``type``), name-sorted."""
        type_filter = _validate_type(type) if type is not None else None
        items: list[dict[str, Any]] = []
        for filename in self._storage.list_md():
            if filename == INDEX_FILENAME:
                continue
            memory = self._try_parse(filename)
            if memory is None:
                continue
            if type_filter is not None and memory.type != type_filter:
                continue
            items.append(
                {
                    "name": memory.name,
                    "description": memory.description,
                    "type": memory.type,
                    "path": filename,
                }
            )
        items.sort(key=lambda item: item["name"])
        return {"count": len(items), "memories": items}

    def index(self) -> dict[str, Any]:
        """Return the raw ``MEMORY.md`` pointer lines (the human-readable index)."""
        if not self._storage.exists(INDEX_FILENAME):
            return {"count": 0, "lines": [], "raw": ""}
        raw = self._storage.read(INDEX_FILENAME)
        lines = [line for line in raw.splitlines() if line.strip().startswith("- [")]
        return {"count": len(lines), "lines": lines, "raw": raw}

    # -- writes ----------------------------------------------------------------

    def write(
        self,
        name: str,
        description: str,
        type: str,
        body: str,
        *,
        links: list[str] | None = None,
    ) -> dict[str, Any]:
        """Create a new memory and add its pointer to the index.

        Refuses to overwrite an existing memory (use :meth:`update`). Validates
        the name/description/type, writes ``<root>/<name>.md``, and inserts a
        single ``MEMORY.md`` pointer line. Idempotent: writing the same content
        for an existing name via :meth:`update` is a no-op-equivalent rewrite.
        """
        name = _validate_name(name)
        relpath = safe_relpath(name)
        if self._storage.exists(relpath):
            raise MemoryError(f"memory {name!r} already exists; use memory_update to change it")
        return self._persist(name, relpath, description, type, body, links)

    def update(
        self,
        name: str,
        *,
        description: str | None = None,
        type: str | None = None,
        body: str | None = None,
        links: list[str] | None = None,
    ) -> dict[str, Any]:
        """Edit an existing memory, preserving fields left unspecified.

        Reads the current memory, overlays only the provided fields (and any
        supplied ``links`` appended to the new body), rewrites the file, and
        reconciles the single index pointer in place. Raises if the memory does
        not exist.
        """
        name = _validate_name(name)
        relpath = safe_relpath(name)
        if not self._storage.exists(relpath):
            raise MemoryError(f"no such memory: {name!r}; use memory_write to create it")
        current = parse(self._storage.read(relpath))
        new_description = description if description is not None else current.description
        new_type = type if type is not None else current.type
        new_body = body if body is not None else current.body
        return self._persist(
            name,
            relpath,
            new_description,
            new_type,
            new_body,
            links,
            extra_metadata=current.extra_metadata,
        )

    # -- internals -------------------------------------------------------------

    def _persist(
        self,
        name: str,
        relpath: str,
        description: str,
        mem_type: str,
        body: str,
        links: list[str] | None,
        *,
        extra_metadata: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        description = _validate_description(description)
        mem_type = _validate_type(mem_type)
        body_text = _link_body(body or "", links)
        memory = Memory(
            name=name,
            description=description,
            type=mem_type,
            body=body_text,
            extra_metadata=dict(extra_metadata or {}),
        )
        self._storage.write(relpath, serialise(memory))
        self._upsert_index_line(name, description)
        return {
            "name": name,
            "path": relpath,
            "type": mem_type,
            "description": description,
            "indexed": True,
        }

    def _upsert_index_line(self, name: str, description: str) -> None:
        """Add or replace the one ``MEMORY.md`` pointer for ``name`` in place.

        Existing pointers for the same filename are replaced (never duplicated);
        a brand-new memory's pointer is appended. Non-pointer lines (headings,
        prose) are preserved untouched.
        """
        new_line = _index_line(name, description)
        link_target = f"]({name}.md)"

        existing = self._storage.read(INDEX_FILENAME) if self._storage.exists(INDEX_FILENAME) else ""
        lines = existing.splitlines()

        replaced = False
        out: list[str] = []
        for line in lines:
            if line.strip().startswith("- [") and link_target in line:
                if not replaced:
                    out.append(new_line)
                    replaced = True
                # Drop any duplicate pointer to the same file.
                continue
            out.append(line)

        if not replaced:
            out.append(new_line)

        rendered = "\n".join(out).rstrip("\n") + "\n"
        self._storage.write(INDEX_FILENAME, rendered)

    def _try_parse(self, filename: str) -> Memory | None:
        """Parse a store file, skipping anything that is not a valid memory."""
        try:
            return parse(self._storage.read(filename))
        except (DocumentError, StorageError):
            return None

    @staticmethod
    def _score(memory: Memory, terms: set[str], phrase: str) -> int:
        name = memory.name.lower()
        description = memory.description.lower()
        body = memory.body.lower()
        score = 0
        for term in terms:
            if term in name:
                score += 10
            if term in description:
                score += 4
            if term in body:
                score += 1
        # Whole-phrase bonuses reward a tight, specific match.
        if phrase in name:
            score += 20
        if phrase in description:
            score += 8
        return score

    @staticmethod
    def _coerce_limit(limit: int) -> int:
        try:
            value = int(limit)
        except (TypeError, ValueError) as error:
            raise MemoryError(f"invalid limit {limit!r}: must be an integer") from error
        if value < 1:
            raise MemoryError("limit must be >= 1")
        return min(value, 100)
