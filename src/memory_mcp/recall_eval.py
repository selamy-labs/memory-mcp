"""Offline fleet-memory recall fixture evaluator.

The evaluator intentionally uses the same SemanticMemory path as the shared
MCP service, but with the deterministic hashing embedder and in-memory vector
store. This gives CI a source-controlled recall/isolation contract without live
MCP or database access.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from memory_mcp.embeddings import HashingEmbedder
from memory_mcp.semantic import SemanticMemory
from memory_mcp.vector_store import InMemoryVectorStore

DEFAULT_NOW = datetime(2026, 7, 13, tzinfo=timezone.utc)


@dataclass(frozen=True)
class CaseResult:
    id: str
    expected: list[str]
    forbidden: list[str]
    hits: list[str]
    positive_passed: bool
    negative_passed: bool

    @property
    def passed(self) -> bool:
        return self.positive_passed and self.negative_passed

    def to_view(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "expected": self.expected,
            "forbidden": self.forbidden,
            "hits": self.hits,
            "positive_passed": self.positive_passed,
            "negative_passed": self.negative_passed,
            "passed": self.passed,
        }


def evaluate_fixture(fixture: dict[str, Any], *, now: datetime = DEFAULT_NOW) -> dict[str, Any]:
    """Evaluate one labeled recall fixture and return a machine-readable summary."""
    memory = SemanticMemory(HashingEmbedder(dim=256), InMemoryVectorStore())
    for record in fixture["records"]:
        memory.add_memory(
            record["name"],
            record["description"],
            record["type"],
            record["body"],
            group_id=record.get("group_id", "fleet"),
            updated_at=record.get("updated_at"),
        )

    results = [_evaluate_case(memory, case, now=now) for case in fixture["cases"]]
    positive_recall_passed = all(result.positive_passed for result in results)
    negative_isolation_passed = all(result.negative_passed for result in results)
    return {
        "fixture": fixture["name"],
        "case_count": len(results),
        "positive_recall_passed": positive_recall_passed,
        "negative_isolation_passed": negative_isolation_passed,
        "overall_verdict": "pass" if positive_recall_passed and negative_isolation_passed else "fail",
        "cases": [result.to_view() for result in results],
    }


def load_fixture(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate a labeled fleet-memory recall fixture.")
    parser.add_argument("fixture", type=Path, help="Path to a fleet-memory recall fixture JSON file.")
    args = parser.parse_args(argv)

    summary = evaluate_fixture(load_fixture(args.fixture))
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["overall_verdict"] == "pass" else 1


def _evaluate_case(memory: SemanticMemory, case: dict[str, Any], *, now: datetime) -> CaseResult:
    out = memory.search_memory(
        case["query"],
        group_ids=case.get("group_ids") or None,
        include_fleet=case.get("include_fleet", True),
        limit=case.get("limit", 5),
        now=now,
    )
    hits = [hit["name"] for hit in out["hits"]]
    expected = list(case.get("expected", []))
    forbidden = list(case.get("forbidden", []))
    return CaseResult(
        id=case["id"],
        expected=expected,
        forbidden=forbidden,
        hits=hits,
        positive_passed=all(name in hits for name in expected),
        negative_passed=not any(name in hits for name in forbidden),
    )


if __name__ == "__main__":
    raise SystemExit(main())
