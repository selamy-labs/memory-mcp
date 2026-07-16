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
        SAFE_WORKFLOW.replace("  contents: read\n", "  contents: read\n  packages: write\n"),
        "workflow requests packages: write",
        id="packages-write",
    ),
    pytest.param(
        SAFE_WORKFLOW.replace(
            "      - uses: docker/setup-buildx-action@v3\n",
            "      - uses: docker/setup-buildx-action@v3\n      - uses: docker/login-action@v3\n",
        ),
        "authenticates to a container registry",
        id="registry-login-action",
    ),
    pytest.param(
        SAFE_WORKFLOW.replace("          push: false", "          push: true"),
        "push is not literal false",
        id="literal-push-true",
    ),
    pytest.param(
        SAFE_WORKFLOW.replace(
            "          push: false", "          push: ${{ github.event_name == 'pull_request' }}"
        ),
        "push is not literal false",
        id="publish-on-pull-request",
    ),
    pytest.param(
        SAFE_WORKFLOW.replace("          push: false", "          push: ${{ github.event_name == 'push' }}"),
        "push is not literal false",
        id="publish-on-push",
    ),
    pytest.param(
        SAFE_WORKFLOW.replace("  workflow_dispatch:\n", "  schedule:\n    - cron: '0 0 * * *'\n  workflow_dispatch:\n").replace(
            "          push: false", "          push: ${{ github.event_name == 'schedule' }}"
        ),
        "push is not literal false",
        id="publish-on-schedule",
    ),
    pytest.param(
        SAFE_WORKFLOW.replace(
            "          push: false", "          push: ${{ github.event_name == 'workflow_dispatch' }}"
        ),
        "push is not literal false",
        id="publish-on-manual-dispatch",
    ),
    pytest.param(
        SAFE_WORKFLOW.replace("          push: false", "          push: ${{ vars.PUBLISH_IMAGE == 'true' }}"),
        "push is not literal false",
        id="expression-can-enable-push",
    ),
    pytest.param(
        SAFE_WORKFLOW.replace("          context: .\n", "          context: .\n          tags: example/image:latest\n"),
        "configures image publication tags",
        id="latest-tag",
    ),
    pytest.param(
        SAFE_WORKFLOW.replace("          context: .\n", "          context: .\n          tags: example/image:edge\n"),
        "configures image publication tags",
        id="other-mutable-tag",
    ),
    pytest.param(
        SAFE_WORKFLOW.replace(
            "  build:\n    runs-on: ubuntu-latest\n    steps:",
            "  publish:\n    uses: ./.github/workflows/publish-image.yml\n  build:\n    runs-on: ubuntu-latest\n    steps:",
        ),
        "calls a reusable workflow",
        id="reusable-publication-workflow",
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
        if "uses" in job:
            violations.append("calls a reusable workflow")
        for step in job.get("steps", []):
            if not isinstance(step, dict):
                continue
            if str(step.get("uses", "")).lower().startswith("docker/login-action@"):
                violations.append("authenticates to a container registry")
            inputs = step.get("with", {})
            if isinstance(inputs, dict) and "push" in inputs and inputs["push"] is not False:
                violations.append("push is not literal false")
            if isinstance(inputs, dict) and "tags" in inputs:
                violations.append("configures image publication tags")

    return violations


def test_image_workflow_is_build_only() -> None:
    workflow = load_workflow(WORKFLOW_PATH.read_text(encoding="utf-8"))

    assert policy_violations(workflow) == []


@pytest.mark.parametrize(("fixture", "expected_violation"), FORBIDDEN_MUTATIONS)
def test_forbidden_image_workflow_mutations_are_rejected(fixture: str, expected_violation: str) -> None:
    workflow = load_workflow(fixture)

    assert expected_violation in policy_violations(workflow)
