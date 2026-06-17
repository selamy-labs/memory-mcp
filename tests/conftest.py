"""Shared offline test doubles: an in-RAM storage and a deterministic clock.

Nothing in the test suite touches the real disk by default. :class:`MemoryStorage`
is a dict of ``filename -> content`` that honours the same root-confinement
contract as :class:`memory_mcp.storage.LocalStorage` (callers pass relative
filenames already validated by ``safe_relpath``). :class:`FakeClock` yields
deterministic UTC-style timestamps. A small set of test memories is provided so
search/list/index assertions have a stable corpus.
"""

from __future__ import annotations

from memory_mcp.document import Memory, serialise
from memory_mcp.storage import StorageError


class MemoryStorage:
    """An in-RAM flat store: ``filename -> content``, no disk, no traversal.

    Rejects any relpath containing a separator so a buggy caller cannot pretend
    to escape the (virtual) root, mirroring LocalStorage's confinement.
    """

    def __init__(self) -> None:
        self.files: dict[str, str] = {}

    def _check(self, relpath: str) -> str:
        if "/" in relpath or "\\" in relpath or relpath in ("", ".", ".."):
            raise StorageError(f"refusing path outside store root: {relpath!r}")
        return relpath

    def read(self, relpath: str) -> str:
        key = self._check(relpath)
        if key not in self.files:
            raise StorageError(f"no such memory file: {relpath}")
        return self.files[key]

    def write(self, relpath: str, content: str) -> None:
        self.files[self._check(relpath)] = content

    def exists(self, relpath: str) -> bool:
        return self._check(relpath) in self.files

    def list_md(self) -> list[str]:
        return sorted(name for name in self.files if name.endswith(".md"))

    # -- test helpers ----------------------------------------------------------

    def seed(self, memory: Memory) -> None:
        """Write a memory directly (bypassing core validation) for read tests."""
        self.files[f"{memory.name}.md"] = serialise(memory)


class FakeClock:
    """A deterministic clock yielding a fixed-format ISO timestamp."""

    def __init__(self) -> None:
        self._seq = 0

    def now_iso(self) -> str:
        self._seq += 1
        return f"2026-06-17T00:00:{self._seq:02d}Z"


def sample_memories() -> list[Memory]:
    """A small, stable corpus used across read tests.

    Includes the extra ``node_type`` metadata key the real store carries, so
    round-trip tests prove unknown keys are preserved.
    """
    return [
        Memory(
            name="laneq-lane-routing",
            description="laneq next filters by lane and bounces on the lowest-id P0",
            type="project",
            body="The laneq queue routes work by lane. Plain next returns the lowest-id P0.",
            extra_metadata={"node_type": "memory"},
        ),
        Memory(
            name="feedback-prefer-wif",
            description="Prefer Workload Identity Federation over service-account keys",
            type="feedback",
            body="Use WIF (keyless OIDC) instead of downloaded SA JSON keys for GCP auth.",
            extra_metadata={"node_type": "memory"},
        ),
        Memory(
            name="nash-credentials-custody",
            description="Nash trading credentials are backed up to Google Secret Manager",
            type="project",
            body="Nash gas wallet and exchange creds live on the PVC and are mirrored to GSM.",
        ),
    ]
