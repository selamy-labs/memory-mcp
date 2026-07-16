from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest
import yaml

WORKFLOW_PATH = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "build-image.yml"
BOOL_TAG = "tag:yaml.org,2002:bool"
SAFE_WORKFLOW = """
name: Build Image
on:
  pull_request:
  push:
  workflow_dispatch:
permissions:
  contents: read
jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: docker/setup-buildx-action@v3
      - uses: docker/build-push-action@v6
        with:
          context: .
          push: false
"""

FORBIDDEN_MUTATIONS = [
    pytest.param(
        SAFE_WORKFLOW.replace(
            "      - uses: docker/setup-buildx-action@v3\n",
            "      - uses: docker/setup-buildx-action@v3\n      - uses: docker/login-action@v3\n",
        ),
        "authenticates to a container registry",
        id="registry-login-action",
    ),
]


class GitHubWorkflowLoader(yaml.SafeLoader):
    """Parse GitHub's YAML 1.2-style booleans without coercing the `on` key."""


GitHubWorkflowLoader.yaml_implicit_resolvers = {
    first: [(tag, regexp) for tag, regexp in resolvers if tag != BOOL_TAG]
    for first, resolvers in yaml.SafeLoader.yaml_implicit_resolvers.items()
}
GitHubWorkflowLoader.add_implicit_resolver(
    BOOL_TAG,
    re.compile(r"^(?:true|false)$", re.IGNORECASE),
    list("tTfF"),
)


def load_workflow(text: str) -> dict[str, Any]:
    workflow = yaml.load(text, Loader=GitHubWorkflowLoader)
    assert isinstance(workflow, dict)
    assert "on" in workflow
    assert True not in workflow
    return workflow


def policy_violations(workflow: dict[str, Any]) -> list[str]:
    violations: list[str] = []
    jobs = workflow.get("jobs", {})
    permission_blocks = [("workflow", workflow.get("permissions", {}))]
    permission_blocks.extend(
        (f"job {name}", job.get("permissions", {}))
        for name, job in jobs.items()
        if isinstance(job, dict)
    )

    for location, permissions in permission_blocks:
        if isinstance(permissions, dict) and permissions.get("packages") == "write":
            violations.append(f"{location} requests packages: write")

    for job in jobs.values():
        if not isinstance(job, dict):
            continue
        for step in job.get("steps", []):
            if isinstance(step, dict) and str(step.get("uses", "")).lower().startswith("docker/login-action@"):
                violations.append("authenticates to a container registry")

    return violations


def test_image_workflow_is_build_only() -> None:
    workflow = load_workflow(WORKFLOW_PATH.read_text(encoding="utf-8"))

    assert policy_violations(workflow) == []


@pytest.mark.parametrize(("fixture", "expected_violation"), FORBIDDEN_MUTATIONS)
def test_forbidden_image_workflow_mutations_are_rejected(fixture: str, expected_violation: str) -> None:
    workflow = load_workflow(fixture)

    assert expected_violation in policy_violations(workflow)
