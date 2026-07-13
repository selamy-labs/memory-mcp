from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _render(*args: str) -> str:
    helm = shutil.which("helm")
    assert helm is not None, "helm must be installed for chart render tests"
    result = subprocess.run(  # noqa: S603 - controlled test command.
        [helm, "template", "memory-mcp", str(ROOT), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


def test_reindex_sync_hook_is_opt_in() -> None:
    rendered = _render("--set", "reindex.enabled=true")

    assert "kind: Job" not in rendered
    assert "memory-mcp-reindex-sync" not in rendered


def test_reindex_sync_hook_renders_git_sources_and_argocd_lifecycle() -> None:
    rendered = _render(
        "--set",
        "reindex.enabled=true",
        "--set",
        "reindex.runOnSync.enabled=true",
        "--set",
        "reindex.cloneTokenSecret.enabled=true",
        "--set",
        "reindex.gitSources[0].name=fleet-memory-operational",
        "--set",
        "reindex.gitSources[0].repo=selamy-labs/fleet-memory-operational",
        "--set",
        "reindex.gitSources[0].group_id=fleet",
        "--set",
        "reindex.gitSources[0].subpath=memories",
        "--set",
        "reindex.gitSources[1].name=fleet-memory-operational-sable",
        "--set",
        "reindex.gitSources[1].repo=selamy-labs/fleet-memory-operational",
        "--set",
        "reindex.gitSources[1].group_id=sable",
        "--set",
        "reindex.gitSources[1].subpath=domains/sable/memories",
    )

    assert "kind: Job" in rendered
    assert "name: memory-mcp-reindex-sync" in rendered
    assert "argocd.argoproj.io/hook: PostSync" in rendered
    assert "argocd.argoproj.io/hook-delete-policy: BeforeHookCreation,HookSucceeded" in rendered
    assert "selamy-labs/fleet-memory-operational" in rendered
    assert "name: GIT_CLONE_TOKEN" in rendered
    assert "name: memory-mcp-clone-token" in rendered
    assert "key: token" in rendered
    assert (
        'value: "/sources/fleet-memory-operational/memories:fleet,'
        "/sources/fleet-memory-operational-sable/domains/sable/memories:sable"
        '"' in rendered
    )
