"""Rebuildable markdown indexer for the shared semantic memory.

Markdown-in-git is the source of truth; the semantic store is a **rebuildable
index** over it (design of record: ``shared-fleet-memory-graphiti.md``). This
module walks the fleet's markdown memory sources, turns each file into a
:class:`~memory_mcp.semantic.SemanticMemory` ``add_memory`` call, and is:

* **idempotent** -- re-running it converges (``add_memory`` upserts on
  ``(group_id, name)``), so a full re-walk after the index PVC dies rebuilds the
  store from git rather than duplicating;
* **recency-honest** -- each record's recency is seeded from the file's own date
  (git last-commit date, else a date in the filename, else file mtime), never
  the index run time, so importing the existing corpus does not make every old
  memory look brand-new;
* **selective** -- it indexes frontmatter-bearing memory files and a configured
  set of roots, not every stray ``.txt``/``.json`` lying next to them;
* **offline-testable** -- all filesystem, git, and clock access goes through
  injected ports, so the full walk runs on an in-RAM source tree in tests.

A *source* pairs a root directory with the scope (``group_id``) its files index
into: the orchestrator's memory dir and per-agent ``/opt/data/memories`` map to
the shared ``fleet`` scope; a domain corpus (e.g. life ``architecture/``) can map
to a domain scope. The indexer never reads credentials and never writes
markdown -- it only reads files and writes into the (injected) semantic store.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol

from memory_mcp.document import DocumentError, parse
from memory_mcp.semantic import SemanticMemory
from memory_mcp.vector_store import FLEET_SCOPE

# The human-readable index file is not itself a memory; skip it on every walk.
INDEX_FILENAME = "MEMORY.md"

# A date embedded in a filename or heading, e.g. ``...-2026-03-29`` -- used as a
# recency fallback when git/mtime is unavailable.
_DATE_RE = re.compile(r"(\d{4})-(\d{2})-(\d{2})")

# Default memory type for a markdown file that has no frontmatter ``type``.
_DEFAULT_TYPE = "reference"

# Derive a kebab-ish name from a filename stem when frontmatter has no ``name``.
_SLUG_RE = re.compile(r"[^a-z0-9]+")


class SourceReader(Protocol):
    """Read-only access to a markdown source tree + each file's recency date.

    ``list_markdown`` yields repo-relative ``.md`` paths under the root;
    ``read`` returns a file's text; ``modified_at`` returns the file's authored
    date (a git last-commit date in production, a seeded value in tests). All
    paths are confined to the configured root by the implementation.
    """

    def list_markdown(self) -> list[str]: ...

    def read(self, relpath: str) -> str: ...

    def modified_at(self, relpath: str) -> datetime | None: ...


@dataclass(frozen=True)
class MemorySource:
    """One markdown root and the scope its files index into."""

    reader: SourceReader
    group_id: str = FLEET_SCOPE


@dataclass
class IndexReport:
    """What one indexing run did, for observable, assertable output."""

    indexed: int = 0
    skipped: int = 0
    errors: list[str] = field(default_factory=list)

    def merge(self, other: IndexReport) -> None:
        self.indexed += other.indexed
        self.skipped += other.skipped
        self.errors.extend(other.errors)

    def to_view(self) -> dict[str, object]:
        return {"indexed": self.indexed, "skipped": self.skipped, "errors": list(self.errors)}


def _slugify(stem: str) -> str:
    slug = _SLUG_RE.sub("-", stem.lower()).strip("-")
    return slug or "memory"


def _date_from_name(relpath: str) -> datetime | None:
    match = _DATE_RE.search(relpath)
    if not match:
        return None
    try:
        return datetime(int(match.group(1)), int(match.group(2)), int(match.group(3)), tzinfo=timezone.utc)
    except ValueError:
        return None


def _first_meaningful_line(body: str) -> str:
    """A short description from the first non-blank, non-heading-marker line."""
    for raw in body.splitlines():
        line = raw.strip().lstrip("#").strip()
        if line:
            return line[:280]
    return ""


# A top-level ``date:`` (or ``updated:`` / ``created:``) line inside the leading
# YAML frontmatter fence. This is the author's explicit recency for a document.
_FRONTMATTER_DATE_RE = re.compile(
    r"^(?:date|updated|created):\s*[\"']?(\d{4}-\d{2}-\d{2})",
    re.MULTILINE,
)


def _date_from_frontmatter(text: str) -> datetime | None:
    """Extract an explicit ``date:``/``updated:``/``created:`` from frontmatter.

    Only looks inside the leading ``---`` fence so a date mentioned in the body
    is never mistaken for the document's recency. Returns ``None`` when there is
    no fenced frontmatter or no recognised date key. This is the most
    authoritative recency signal — an author set it deliberately — so it ranks
    above a filename date or a (possibly shallow-clone) git commit date.
    """
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return None
    end = next((i for i in range(1, len(lines)) if lines[i].strip() == "---"), None)
    if end is None:
        return None
    front = "\n".join(lines[1:end])
    match = _FRONTMATTER_DATE_RE.search(front)
    if not match:
        return None
    try:
        year, month, day = (int(part) for part in match.group(1).split("-"))
        return datetime(year, month, day, tzinfo=timezone.utc)
    except ValueError:
        return None


@dataclass(frozen=True)
class _Candidate:
    """A parsed markdown file ready to index: contract fields + recency."""

    name: str
    description: str
    type: str
    body: str
    updated_at: datetime


def _build_candidate(relpath: str, text: str, modified_at: datetime | None) -> _Candidate | None:
    """Turn a markdown file into an indexable candidate.

    A file with valid memory frontmatter uses its ``name``/``description``/
    ``type`` directly. A frontmatter-less markdown file is still indexed (the
    fleet has plenty of un-frontmattered notes) by synthesising a name from the
    filename, a description from its first line, and the default type. Recency is
    an explicit frontmatter ``date:`` if present, else a date in the filename,
    else the file's git/mtime date; a file with no usable date at all is reported
    as skipped (recency-honesty over silent "now").
    """
    name: str
    description: str
    mem_type: str
    body: str
    try:
        memory = parse(text)
        name = memory.name
        description = memory.description
        mem_type = memory.type
        body = memory.body
    except DocumentError:
        stem = Path(relpath).stem
        name = _slugify(stem)
        description = _first_meaningful_line(text) or stem
        mem_type = _DEFAULT_TYPE
        body = text

    # Recency precedence, most authoritative first: an explicit frontmatter date
    # (the author set it) > a date in the filename > the file's git commit / mtime
    # date. This keeps recency honest even when the source is a shallow clone
    # (where every file shares the single cloned commit's date).
    updated_at = _date_from_frontmatter(text) or _date_from_name(relpath) or modified_at
    if updated_at is None:
        return None
    return _Candidate(name=name, description=description, type=mem_type, body=body, updated_at=updated_at)


class MarkdownIndexer:
    """Walk markdown sources and (re)index them into the semantic store."""

    def __init__(self, memory: SemanticMemory) -> None:
        self._memory = memory

    def index_source(self, source: MemorySource) -> IndexReport:
        """Index every markdown file under one source into its scope."""
        report = IndexReport()
        for relpath in source.reader.list_markdown():
            if Path(relpath).name == INDEX_FILENAME:
                report.skipped += 1
                continue
            try:
                text = source.reader.read(relpath)
            except OSError as error:  # pragma: no cover - defensive; reader confines paths
                report.errors.append(f"{relpath}: read failed: {error}")
                continue

            candidate = _build_candidate(relpath, text, source.reader.modified_at(relpath))
            if candidate is None:
                report.skipped += 1
                continue
            try:
                self._memory.add_memory(
                    candidate.name,
                    candidate.description,
                    candidate.type,
                    candidate.body,
                    group_id=source.group_id,
                    updated_at=candidate.updated_at,
                )
                report.indexed += 1
            except Exception as error:  # record and continue; one bad file must not abort a rebuild
                report.errors.append(f"{relpath}: {error}")
        return report

    def index_all(self, sources: list[MemorySource]) -> IndexReport:
        """Index every source; a failure in one file never aborts the rebuild."""
        total = IndexReport()
        for source in sources:
            total.merge(self.index_source(source))
        return total


class LocalSourceReader:
    """Production reader over a real directory; recency from git, else mtime.

    ``modified_at`` shells out to ``git log -1`` for the file's last-commit date
    (deterministic and rebuildable from history); when the tree is not a git
    checkout it falls back to the filesystem mtime. The git call is injected as
    ``run_git`` so it is stubbed in tests and never required at import time.
    """

    def __init__(
        self,
        root: str | Path,
        *,
        run_git=None,
    ) -> None:
        self._root = Path(root).expanduser()
        self._run_git = run_git

    def _resolve(self, relpath: str) -> Path:
        target = (self._root / relpath).resolve()
        root = self._root.resolve()
        if root != target and root not in target.parents:
            raise OSError(f"refusing path outside source root: {relpath!r}")
        return target

    def list_markdown(self) -> list[str]:
        if not self._root.is_dir():
            return []
        root = self._root.resolve()
        return sorted(str(p.resolve().relative_to(root)) for p in self._root.rglob("*.md") if p.is_file())

    def read(self, relpath: str) -> str:
        return self._resolve(relpath).read_text(encoding="utf-8")

    def modified_at(self, relpath: str) -> datetime | None:
        git_date = self._git_date(relpath)
        if git_date is not None:
            return git_date
        try:
            ts = self._resolve(relpath).stat().st_mtime
        except OSError:
            return None
        return datetime.fromtimestamp(ts, tz=timezone.utc)

    def _git_date(self, relpath: str) -> datetime | None:
        if self._run_git is None:
            return None
        out = self._run_git(["log", "-1", "--format=%cI", "--", relpath], cwd=str(self._root))
        if not out:
            return None
        try:
            from memory_mcp.vector_store import _parse_iso

            return _parse_iso(out.strip())
        except ValueError:
            return None
