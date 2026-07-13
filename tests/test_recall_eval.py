from __future__ import annotations

from pathlib import Path

from memory_mcp.recall_eval import evaluate_fixture, load_fixture, main

FIXTURE = Path("fixtures/fleet_memory_recall_fixture.json")


def test_fleet_memory_recall_fixture_passes_positive_and_negative_cases() -> None:
    summary = evaluate_fixture(load_fixture(FIXTURE))

    assert summary["case_count"] == 4
    assert summary["positive_recall_passed"] is True
    assert summary["negative_isolation_passed"] is True
    assert summary["overall_verdict"] == "pass"


def test_recall_eval_cli_emits_success_summary(capsys) -> None:
    assert main([str(FIXTURE)]) == 0

    captured = capsys.readouterr()
    assert '"overall_verdict": "pass"' in captured.out
    assert '"negative_isolation_passed": true' in captured.out
