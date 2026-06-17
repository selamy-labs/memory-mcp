"""Filesystem and clock abstractions so the memory core stays offline-testable.

The memory store is just markdown files in a root directory, but the core never
touches :mod:`os` or :class:`pathlib.Path` directly. It depends only on the
:class:`Storage` protocol below -- read a file, write a file, list the ``.md``
files, check existence. Production code injects :class:`LocalStorage`, which is
the one place real disk I/O happens; tests inject :class:`MemoryStorage`, an
in-RAM dict, so the full search/get/write/index path runs with no disk.

One safety property lives here and is enforced for every write: a memory
``name`` is resolved to a path **inside the root only**. A name that escapes the
root (``..``, an absolute path, a separator) is rejected before any byte is
written, so a write can never clobber a file outside the configured store.
"""

from __future__ import annotations

from pathlib import Path, PurePosixPath
from typing import Protocol


class StorageError(Exception):
    """A storage operation failed (missing file, escape attempt, I/O error)."""


def safe_relpath(name: str) -> str:
    """Resolve a memory ``name`` to a single ``<name>.md`` filename inside root.

    The name is a flat kebab-case slug -- never a path. This rejects anything
    that could escape the root (separators, ``..`` components, absolute paths,
    drive/anchor parts), so a resolved write target is always a direct child of
    the store root. Returns the relative filename (``"<name>.md"``).
    """
    cleaned = name.strip()
    if not cleaned:
        raise StorageError("memory name must not be empty")
    if "/" in cleaned or "\\" in cleaned or "\x00" in cleaned:
        raise StorageError(f"invalid memory name {name!r}: must not contain path separators")
    # With separators already rejected, the only remaining escapes are the '.'
    # and '..' relative components; everything else is a single safe filename.
    candidate = PurePosixPath(cleaned)
    if candidate.parts != (cleaned,) or cleaned in ("..", "."):
        raise StorageError(f"invalid memory name {name!r}: must be a single path component")
    return f"{cleaned}.md"


class Storage(Protocol):
    """A flat file store rooted at one directory.

    All paths are relative filenames within the root. Implementations must never
    resolve a path outside the root; callers pass names already validated by
    :func:`safe_relpath`. Reading a missing file raises :class:`StorageError`.
    """

    def read(self, relpath: str) -> str: ...

    def write(self, relpath: str, content: str) -> None: ...

    def exists(self, relpath: str) -> bool: ...

    def list_md(self) -> list[str]: ...


class LocalStorage:
    """Production storage: real files under a configured root directory.

    The root is created on first write if it does not yet exist. Every path is
    re-checked against the resolved root so even a bug in a caller cannot write
    outside the store.
    """

    def __init__(self, root: str | Path) -> None:
        self._root = Path(root).expanduser()

    @property
    def root(self) -> Path:
        return self._root

    def _resolve(self, relpath: str) -> Path:
        target = (self._root / relpath).resolve()
        root = self._root.resolve()
        if root != target and root not in target.parents:
            raise StorageError(f"refusing path outside store root: {relpath!r}")
        return target

    def read(self, relpath: str) -> str:
        target = self._resolve(relpath)
        try:
            return target.read_text(encoding="utf-8")
        except FileNotFoundError as error:
            raise StorageError(f"no such memory file: {relpath}") from error
        except OSError as error:
            raise StorageError(f"failed to read {relpath}: {error}") from error

    def write(self, relpath: str, content: str) -> None:
        target = self._resolve(relpath)
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
        except OSError as error:
            raise StorageError(f"failed to write {relpath}: {error}") from error

    def exists(self, relpath: str) -> bool:
        return self._resolve(relpath).is_file()

    def list_md(self) -> list[str]:
        root = self._root
        if not root.is_dir():
            return []
        return sorted(p.name for p in root.iterdir() if p.is_file() and p.suffix == ".md")


class Clock(Protocol):
    """A wall clock, injected so timestamps in audit output are testable."""

    def now_iso(self) -> str: ...


class SystemClock:
    """The real clock: UTC ISO timestamps with a trailing ``Z``."""

    def now_iso(self) -> str:
        from datetime import datetime, timezone

        return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
