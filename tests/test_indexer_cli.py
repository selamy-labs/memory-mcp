"""Tests for the indexer CLI (``python -m memory_mcp``)."""

from __future__ import annotations

import json
from datetime import datetime, timezone

from memory_mcp.__main__ import build_sources, main, parse_sources_spec
from memory_mcp.embeddings import HashingEmbedder
from memory_mcp.indexer import MemorySource
from memory_mcp.semantic import SemanticMemory
from memory_mcp.vector_store import FLEET_SCOPE, InMemoryVectorStore

_FRONTMATTER = """---
name: a-mem
description: a description
metadata:
  type: feedback
---
body text
"""


def test_parse_sources_spec_variants():
    pairs = parse_sources_spec("/a, /b:infra , , /c:")
    assert pairs == [("/a", FLEET_SCOPE), ("/b", "infra"), ("/c", FLEET_SCOPE)]
    assert parse_sources_spec("") == []
    # An entry that is only a scope (empty path) is dropped, not indexed.
    assert parse_sources_spec(":infra") == []


def test_build_sources_constructs_local_readers(tmp_path):
    sources = build_sources(f"{tmp_path}:infra")
    assert len(sources) == 1 and isinstance(sources[0], MemorySource)
    assert sources[0].group_id == "infra"


def test_main_no_sources_returns_2(capsys):
    assert main(argv=[], memory=None) == 2 or main(argv=[]) == 2
    err = capsys.readouterr().err
    assert "no sources" in err


def test_main_indexes_dir_and_prints_report(tmp_path, capsys):
    (tmp_path / "a.md").write_text(_FRONTMATTER, encoding="utf-8")
    mem = SemanticMemory(HashingEmbedder(dim=128), InMemoryVectorStore())
    code = main(argv=[str(tmp_path)], memory=mem)
    out = json.loads(capsys.readouterr().out)
    assert code == 0
    assert out["indexed"] == 1
    assert mem.get_memory("a-mem")["name"] == "a-mem"


def test_main_reads_spec_from_env(tmp_path, monkeypatch, capsys):
    (tmp_path / "a.md").write_text(_FRONTMATTER, encoding="utf-8")
    monkeypatch.setenv("MEMORY_SOURCES", f"{tmp_path}:fleet")
    mem = SemanticMemory(HashingEmbedder(dim=128), InMemoryVectorStore())
    code = main(argv=[], memory=mem)
    out = json.loads(capsys.readouterr().out)
    assert code == 0 and out["indexed"] == 1


def test_main_default_builder_runs_without_injected_memory(tmp_path, capsys):
    (tmp_path / "a.md").write_text(_FRONTMATTER, encoding="utf-8")
    code = main(argv=[str(tmp_path)])
    out = json.loads(capsys.readouterr().out)
    assert code == 0 and out["indexed"] == 1


def test_main_returns_1_when_a_file_errors(tmp_path, capsys, monkeypatch):
    (tmp_path / "a.md").write_text(_FRONTMATTER, encoding="utf-8")
    mem = SemanticMemory(HashingEmbedder(dim=128), InMemoryVectorStore())
    monkeypatch.setattr(mem, "add_memory", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    code = main(argv=[str(tmp_path)], memory=mem)
    out = json.loads(capsys.readouterr().out)
    assert code == 1 and out["errors"]


def test_run_git_returns_stdout_on_success(tmp_path):
    from memory_mcp.__main__ import _run_git

    out = _run_git(["rev-parse", "--show-toplevel"], cwd=str(tmp_path))
    assert isinstance(out, str)  # '' when tmp_path is not a repo, real path otherwise


def test_run_git_handles_missing_binary(monkeypatch, tmp_path):
    import memory_mcp.__main__ as cli

    def boom(*a, **k):
        raise FileNotFoundError("git not found")

    monkeypatch.setattr(cli.subprocess, "run", boom)
    assert cli._run_git(["log"], cwd=str(tmp_path)) == ""


def test_build_components_default(tmp_path):
    from memory_mcp.__main__ import build_components

    mem = build_components()
    mem.add_memory("x", "d", "feedback", "b", updated_at=datetime(2026, 1, 1, tzinfo=timezone.utc))
    assert mem.get_memory("x")["name"] == "x"
