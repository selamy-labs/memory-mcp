"""Tests for path-safety and the real LocalStorage adapter, fully offline.

LocalStorage is exercised against a pytest ``tmp_path`` -- a real but temporary
directory -- so the production disk path gets coverage without touching any
configured store. The path-traversal contract is tested directly on
``safe_relpath`` and again through LocalStorage's own resolution guard.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from memory_mcp.storage import LocalStorage, StorageError, SystemClock, safe_relpath

# -- safe_relpath: the traversal gate ------------------------------------------


def test_safe_relpath_accepts_plain_slug() -> None:
    assert safe_relpath("my-memory") == "my-memory.md"


def test_safe_relpath_rejects_empty() -> None:
    with pytest.raises(StorageError, match="must not be empty"):
        safe_relpath("   ")


@pytest.mark.parametrize(
    "bad",
    [
        "../escape",
        "../../etc/passwd",
        "sub/dir",
        "back\\slash",
        "/absolute",
        "..",
        ".",
        "with\x00null",
    ],
)
def test_safe_relpath_rejects_traversal(bad: str) -> None:
    with pytest.raises(StorageError):
        safe_relpath(bad)


# -- LocalStorage round-trip on a temp dir -------------------------------------


def test_local_storage_write_read_exists(tmp_path: Path) -> None:
    storage = LocalStorage(tmp_path)
    assert not storage.exists("a.md")
    storage.write("a.md", "hello")
    assert storage.exists("a.md")
    assert storage.read("a.md") == "hello"
    assert storage.root == tmp_path


def test_local_storage_creates_root_on_first_write(tmp_path: Path) -> None:
    root = tmp_path / "does-not-exist-yet"
    storage = LocalStorage(root)
    assert storage.list_md() == []  # missing dir lists empty, not error
    storage.write("m.md", "x")
    assert (root / "m.md").read_text(encoding="utf-8") == "x"


def test_local_storage_list_md_sorted_and_md_only(tmp_path: Path) -> None:
    storage = LocalStorage(tmp_path)
    storage.write("b.md", "1")
    storage.write("a.md", "2")
    (tmp_path / "not-markdown.txt").write_text("ignore", encoding="utf-8")
    assert storage.list_md() == ["a.md", "b.md"]


def test_local_storage_read_missing_raises(tmp_path: Path) -> None:
    storage = LocalStorage(tmp_path)
    with pytest.raises(StorageError, match="no such memory file"):
        storage.read("nope.md")


def test_local_storage_rejects_path_outside_root(tmp_path: Path) -> None:
    storage = LocalStorage(tmp_path)
    with pytest.raises(StorageError, match="outside store root"):
        storage.read("../escape.md")


def test_local_storage_read_os_error_surfaces(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    storage = LocalStorage(tmp_path)
    storage.write("m.md", "x")

    def _boom(*_a: object, **_k: object) -> None:
        raise OSError("disk gone")

    monkeypatch.setattr(Path, "read_text", _boom)
    with pytest.raises(StorageError, match="failed to read"):
        storage.read("m.md")


def test_local_storage_write_os_error_surfaces(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    storage = LocalStorage(tmp_path)

    def _boom(*_a: object, **_k: object) -> None:
        raise OSError("read-only filesystem")

    monkeypatch.setattr(Path, "write_text", _boom)
    with pytest.raises(StorageError, match="failed to write"):
        storage.write("m.md", "x")


def test_system_clock_now_iso_has_z() -> None:
    assert SystemClock().now_iso().endswith("Z")
