"""CLI for the markdown indexer: ``python -m memory_mcp``.

Rebuilds the semantic index from markdown-in-git. Designed to run as a one-shot
job (a Kubernetes CronJob in production, or by hand for a local rebuild). Config
is read from the environment at start time:

* ``MEMORY_SOURCES`` -- a comma-separated list of ``path[:group_id]`` entries.
  ``path`` is a markdown root; the optional ``:group_id`` scopes that root
  (default ``fleet``). Example::

      MEMORY_SOURCES="/opt/data/memories,/home/dev/.claude/.../memory:fleet,/life/architecture:infra"

* embeddings + vector store selection is delegated to :func:`build_components`,
  which the deployment wires (hashing embedder + pgvector in production); by
  default this CLI uses the offline hashing embedder + an in-memory store so it
  runs anywhere, printing a JSON report. The deployment overrides the store via
  the same builder so a real rebuild writes to Postgres.

No credentials are read here; a real embedder/store that needs a key resolves it
at construction from its own injected config.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from collections.abc import Sequence

from memory_mcp.embeddings import Embedder, HashingEmbedder
from memory_mcp.indexer import LocalSourceReader, MarkdownIndexer, MemorySource
from memory_mcp.semantic import SemanticMemory
from memory_mcp.vector_store import FLEET_SCOPE, InMemoryVectorStore, VectorStore


def parse_sources_spec(spec: str) -> list[tuple[str, str]]:
    """Parse ``MEMORY_SOURCES`` into ``(path, group_id)`` pairs.

    Each comma-separated entry is ``path`` or ``path:group_id``. A trailing
    ``:group_id`` is split on the LAST colon so Windows-style or URL-ish paths
    are not mangled; an empty entry is ignored.
    """
    pairs: list[tuple[str, str]] = []
    for raw in spec.split(","):
        entry = raw.strip()
        if not entry:
            continue
        if ":" in entry:
            path, group_id = entry.rsplit(":", 1)
            path = path.strip()
            group_id = group_id.strip() or FLEET_SCOPE
        else:
            path, group_id = entry, FLEET_SCOPE
        if path:
            pairs.append((path, group_id))
    return pairs


def _run_git(argv: Sequence[str], *, cwd: str) -> str:
    """Run a fixed ``git`` query for a file's commit date; '' on any failure."""
    try:
        result = subprocess.run(  # noqa: S603 - fixed git query, no shell
            ["git", *argv],  # noqa: S607 - git resolved from PATH by design (portable across images)
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return result.stdout if result.returncode == 0 else ""


def build_components(
    embedder: Embedder | None = None,
    store: VectorStore | None = None,
) -> SemanticMemory:
    """Construct the SemanticMemory the indexer writes into.

    When ``MEMORY_BACKEND`` is set (the deployment / reindex CronJob), the same
    env-driven builders the server uses select a durable embedder + pgvector
    store, so a rebuild actually writes to Postgres. With no backend configured,
    it defaults to the offline hashing embedder + in-memory store so the CLI runs
    anywhere. Explicit ``embedder``/``store`` arguments (tests) win over both.
    """
    if embedder is None and store is None and os.environ.get("MEMORY_BACKEND"):
        from memory_mcp.server_semantic import build_memory

        return build_memory()
    return SemanticMemory(embedder or HashingEmbedder(), store or InMemoryVectorStore())


def build_sources(spec: str) -> list[MemorySource]:
    """Build :class:`MemorySource` objects from a ``MEMORY_SOURCES`` spec."""
    return [
        MemorySource(reader=LocalSourceReader(path, run_git=_run_git), group_id=group_id)
        for path, group_id in parse_sources_spec(spec)
    ]


def main(argv: Sequence[str] | None = None, *, memory: SemanticMemory | None = None) -> int:
    """Index every configured source and print a JSON report. Returns exit code."""
    args = list(sys.argv[1:] if argv is None else argv)
    spec = os.environ.get("MEMORY_SOURCES", "")
    if args:
        spec = ",".join(args)
    if not spec.strip():
        print("no sources: set MEMORY_SOURCES or pass paths as arguments", file=sys.stderr)
        return 2

    sources = build_sources(spec)
    indexer = MarkdownIndexer(memory or build_components())
    report = indexer.index_all(sources)
    print(json.dumps(report.to_view()))
    return 0 if not report.errors else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
