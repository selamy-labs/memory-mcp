"""Tests for the rebuildable markdown indexer (offline, in-RAM sources)."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from memory_mcp.embeddings import HashingEmbedder
from memory_mcp.indexer import (
    MarkdownIndexer,
    MemorySource,
    _build_candidate,
    _date_from_frontmatter,
    _date_from_name,
    _first_meaningful_line,
    _slugify,
)
from memory_mcp.semantic import SemanticMemory
from memory_mcp.vector_store import FLEET_SCOPE, InMemoryVectorStore

_FRONTMATTER = """---
name: prefer-wif
description: Prefer WIF over service-account keys
metadata:
  node_type: memory
  type: feedback
---

Use keyless OIDC for GCP auth.
"""

_PLAIN = "# Some Loose Note\n\nThis note has no frontmatter at all.\n"


class FakeReader:
    """In-RAM source: relpath -> (text, modified_at)."""

    def __init__(self, files: dict[str, tuple[str, datetime | None]]) -> None:
        self._files = files

    def list_markdown(self) -> list[str]:
        return sorted(self._files)

    def read(self, relpath: str) -> str:
        return self._files[relpath][0]

    def modified_at(self, relpath: str) -> datetime | None:
        return self._files[relpath][1]


def _memory() -> SemanticMemory:
    return SemanticMemory(HashingEmbedder(dim=256), InMemoryVectorStore())


def test_slugify_and_helpers():
    assert _slugify("Axel Coaching 2026") == "axel-coaching-2026"
    assert _slugify("!!!") == "memory"
    assert _first_meaningful_line("\n\n## Heading\nbody") == "Heading"
    assert _first_meaningful_line("\n\n") == ""


def test_date_from_name():
    assert _date_from_name("axel-coaching-session-2026-03-29.md") == datetime(2026, 3, 29, tzinfo=timezone.utc)
    assert _date_from_name("no-date-here.md") is None
    assert _date_from_name("bad-2026-13-40.md") is None  # invalid month/day -> None


# A session-journal-style file: frontmatter with a `date:` but no memory `name`/
# `type`, so it synthesises but its explicit date must win for recency.
_JOURNAL = """---
date: 2026-05-10
project: dev
headline: "A day in dev"
---

# 2026-05-10 — dev

Some substantive turns.
"""


def test_date_from_frontmatter():
    assert _date_from_frontmatter(_JOURNAL) == datetime(2026, 5, 10, tzinfo=timezone.utc)
    assert _date_from_frontmatter("---\nupdated: 2026-04-04\n---\nbody") == datetime(2026, 4, 4, tzinfo=timezone.utc)
    assert _date_from_frontmatter("no frontmatter here") is None
    assert _date_from_frontmatter("---\nproject: x\n---\nbody") is None  # no date key
    assert _date_from_frontmatter("---\nunclosed fence\nbody") is None
    assert _date_from_frontmatter("---\ndate: 2026-13-40\n---\n") is None  # invalid -> None
    # A date in the BODY (not frontmatter) must not be picked up.
    assert _date_from_frontmatter("---\nproject: x\n---\ndate: 2026-01-01") is None


def test_build_candidate_frontmatter_date_beats_clone_commit_date():
    # Shallow clone gives every file the same (today-ish) commit date; the
    # explicit frontmatter date must win so journal recency stays honest.
    clone_date = datetime(2026, 6, 23, tzinfo=timezone.utc)
    cand = _build_candidate("journal/2026-05-10--dev.md", _JOURNAL, clone_date)
    assert cand.updated_at == datetime(2026, 5, 10, tzinfo=timezone.utc)
    assert cand.type == "reference"  # synthesised (no memory frontmatter)


def test_build_candidate_uses_frontmatter():
    when = datetime(2026, 1, 1, tzinfo=timezone.utc)
    cand = _build_candidate("prefer-wif.md", _FRONTMATTER, when)
    assert cand.name == "prefer-wif"
    assert cand.type == "feedback"
    # _FRONTMATTER has no date: field, so the git/mtime date is used.
    assert cand.updated_at == when


def test_build_candidate_synthesises_for_plain_markdown():
    when = datetime(2026, 2, 2, tzinfo=timezone.utc)
    cand = _build_candidate("notes/Some Loose Note.md", _PLAIN, when)
    assert cand.name == "some-loose-note"
    assert cand.type == "reference"
    assert cand.description == "Some Loose Note"


def test_build_candidate_recency_falls_back_to_filename_date():
    cand = _build_candidate("session-2026-03-29.md", _PLAIN, None)
    assert cand.updated_at == datetime(2026, 3, 29, tzinfo=timezone.utc)


def test_build_candidate_skips_when_no_date_available():
    assert _build_candidate("undated.md", _PLAIN, None) is None


def test_index_source_indexes_and_seeds_recency():
    mem = _memory()
    reader = FakeReader(
        {
            "prefer-wif.md": (_FRONTMATTER, datetime(2026, 1, 1, tzinfo=timezone.utc)),
            "loose.md": (_PLAIN, datetime(2026, 2, 2, tzinfo=timezone.utc)),
        }
    )
    report = MarkdownIndexer(mem).index_source(MemorySource(reader=reader))
    assert report.indexed == 2
    got = mem.get_memory("prefer-wif")
    assert got["updated_at"] == "2026-01-01T00:00:00Z"  # frontmatter file's authored date, not "now"


def test_index_skips_memory_index_file_and_undated():
    mem = _memory()
    reader = FakeReader(
        {
            "MEMORY.md": ("- [x.md](x.md) — index pointer\n", datetime(2026, 1, 1, tzinfo=timezone.utc)),
            "undated.md": (_PLAIN, None),
            "good.md": (_FRONTMATTER, datetime(2026, 1, 1, tzinfo=timezone.utc)),
        }
    )
    report = MarkdownIndexer(mem).index_source(MemorySource(reader=reader))
    assert report.indexed == 1
    assert report.skipped == 2


def test_index_is_idempotent_on_rerun():
    mem = _memory()
    reader = FakeReader({"prefer-wif.md": (_FRONTMATTER, datetime(2026, 1, 1, tzinfo=timezone.utc))})
    indexer = MarkdownIndexer(mem)
    indexer.index_source(MemorySource(reader=reader))
    indexer.index_source(MemorySource(reader=reader))
    assert mem.search_memory("keyless OIDC", now=datetime(2026, 6, 1, tzinfo=timezone.utc))["count"] == 1


def test_index_records_per_file_error_without_aborting():
    mem = _memory()
    # A frontmatter doc with an empty description triggers add_memory validation.
    bad = "---\nname: bad\ndescription: \nmetadata:\n  type: feedback\n---\nbody\n"
    reader = FakeReader(
        {
            "bad.md": (bad, datetime(2026, 1, 1, tzinfo=timezone.utc)),
            "good.md": (_FRONTMATTER, datetime(2026, 1, 1, tzinfo=timezone.utc)),
        }
    )
    report = MarkdownIndexer(mem).index_source(MemorySource(reader=reader))
    # 'bad' fails to PARSE (empty description) -> synthesised as plain markdown -> still indexes.
    assert report.indexed == 2
    assert report.errors == []


def test_index_all_merges_reports_across_sources_and_scopes():
    mem = _memory()
    fleet = FakeReader({"a.md": (_FRONTMATTER, datetime(2026, 1, 1, tzinfo=timezone.utc))})
    infra = FakeReader({"some-loose-note.md": (_PLAIN, datetime(2026, 2, 2, tzinfo=timezone.utc))})
    report = MarkdownIndexer(mem).index_all(
        [
            MemorySource(reader=fleet, group_id=FLEET_SCOPE),
            MemorySource(reader=infra, group_id="infra"),
        ]
    )
    assert report.indexed == 2
    assert mem.get_memory("prefer-wif", group_id=FLEET_SCOPE)["name"] == "prefer-wif"
    assert mem.get_memory("some-loose-note", group_id="infra")["name"] == "some-loose-note"


def test_report_view_shape():
    mem = _memory()
    reader = FakeReader({"a.md": (_FRONTMATTER, datetime(2026, 1, 1, tzinfo=timezone.utc))})
    view = MarkdownIndexer(mem).index_all([MemorySource(reader=reader)]).to_view()
    assert view == {"indexed": 1, "skipped": 0, "errors": []}


def test_add_memory_error_is_recorded(monkeypatch):
    mem = _memory()

    def boom(*args, **kwargs):
        raise RuntimeError("store down")

    monkeypatch.setattr(mem, "add_memory", boom)
    reader = FakeReader({"a.md": (_FRONTMATTER, datetime(2026, 1, 1, tzinfo=timezone.utc))})
    report = MarkdownIndexer(mem).index_source(MemorySource(reader=reader))
    assert report.indexed == 0
    assert report.errors and "store down" in report.errors[0]


def test_local_source_reader_round_trip(tmp_path):
    from memory_mcp.indexer import LocalSourceReader

    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "note-2026-05-05.md").write_text(_PLAIN, encoding="utf-8")
    reader = LocalSourceReader(tmp_path)
    listed = reader.list_markdown()
    assert listed == ["sub/note-2026-05-05.md"]
    assert reader.read(listed[0]) == _PLAIN
    # No git runner injected -> mtime fallback yields a real datetime.
    assert isinstance(reader.modified_at(listed[0]), datetime)


def test_local_source_reader_empty_root(tmp_path):
    from memory_mcp.indexer import LocalSourceReader

    missing = LocalSourceReader(tmp_path / "nope")
    assert missing.list_markdown() == []


def test_local_source_reader_rejects_path_escape(tmp_path):
    from memory_mcp.indexer import LocalSourceReader

    reader = LocalSourceReader(tmp_path)
    with pytest.raises(OSError, match="outside source root"):
        reader.read("../escape.md")


def test_local_source_reader_git_date(tmp_path):
    from memory_mcp.indexer import LocalSourceReader

    calls: list[list[str]] = []

    def fake_git(argv, *, cwd):
        calls.append(list(argv))
        return "2026-04-04T12:00:00Z\n"

    (tmp_path / "x.md").write_text(_PLAIN, encoding="utf-8")
    reader = LocalSourceReader(tmp_path, run_git=fake_git)
    when = reader.modified_at("x.md")
    assert when == datetime(2026, 4, 4, 12, 0, tzinfo=timezone.utc)
    assert calls and calls[0][0] == "log"


def test_local_source_reader_git_empty_falls_back_to_mtime(tmp_path):
    from memory_mcp.indexer import LocalSourceReader

    (tmp_path / "x.md").write_text(_PLAIN, encoding="utf-8")
    reader = LocalSourceReader(tmp_path, run_git=lambda argv, *, cwd: "")
    assert isinstance(reader.modified_at("x.md"), datetime)


def test_local_source_reader_git_bad_date_falls_back(tmp_path):
    from memory_mcp.indexer import LocalSourceReader

    (tmp_path / "x.md").write_text(_PLAIN, encoding="utf-8")
    reader = LocalSourceReader(tmp_path, run_git=lambda argv, *, cwd: "not-a-date\n")
    assert isinstance(reader.modified_at("x.md"), datetime)


def test_local_source_reader_missing_file_mtime_is_none(tmp_path):
    from memory_mcp.indexer import LocalSourceReader

    reader = LocalSourceReader(tmp_path)
    assert reader.modified_at("ghost.md") is None
