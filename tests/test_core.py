"""Tests for the memory store core, fully offline on an in-RAM root.

A :class:`MemoryStorage` stands in for the disk and a :class:`FakeClock` makes
timing deterministic, so search ranking, the get/list/index round-trips, and
the careful-write contract (validation, no-clobber, no-traversal, index
consistency, idempotence) all run with no filesystem.
"""

from __future__ import annotations

import pytest

from memory_mcp.core import INDEX_FILENAME, MemoryError, MemoryStore
from memory_mcp.storage import StorageError
from tests.conftest import FakeClock, MemoryStorage, sample_memories


def _store(seeded: bool = True) -> tuple[MemoryStore, MemoryStorage]:
    storage = MemoryStorage()
    store = MemoryStore(storage, clock=FakeClock())
    if seeded:
        for memory in sample_memories():
            storage.seed(memory)
    return store, storage


# -- search ranking ------------------------------------------------------------


def test_search_ranks_name_match_above_body_match() -> None:
    store, _ = _store()
    # "laneq" is in one memory's name; "creds" only in another's body.
    out = store.search("laneq")
    assert out["hits"][0]["name"] == "laneq-lane-routing"
    assert out["total_matched"] >= 1


def test_search_phrase_bonus_promotes_exact_description() -> None:
    store, _ = _store()
    out = store.search("Workload Identity Federation")
    assert out["hits"][0]["name"] == "feedback-prefer-wif"


def test_search_filters_by_type() -> None:
    store, _ = _store()
    out = store.search("the", type="feedback")
    names = {hit["name"] for hit in out["hits"]}
    assert names <= {"feedback-prefer-wif"}
    for hit in out["hits"]:
        assert hit["type"] == "feedback"


def test_search_respects_limit() -> None:
    store, _ = _store()
    out = store.search("the", limit=1)
    assert out["count"] <= 1
    assert len(out["hits"]) <= 1


def test_search_no_match_is_empty() -> None:
    store, _ = _store()
    out = store.search("zzzznomatchterm")
    assert out["count"] == 0
    assert out["hits"] == []


def test_search_empty_query_rejected() -> None:
    store, _ = _store()
    with pytest.raises(MemoryError, match="query must not be empty"):
        store.search("   ")


def test_search_bad_type_rejected() -> None:
    store, _ = _store()
    with pytest.raises(MemoryError, match="invalid type"):
        store.search("x", type="bogus")


def test_search_bad_limit_rejected() -> None:
    store, _ = _store()
    with pytest.raises(MemoryError, match="limit must be"):
        store.search("x", limit=0)
    with pytest.raises(MemoryError, match="invalid limit"):
        store.search("x", limit="lots")  # type: ignore[arg-type]


def test_search_skips_unparseable_files_and_index() -> None:
    store, storage = _store()
    storage.files["broken.md"] = "not a memory at all"
    storage.files[INDEX_FILENAME] = "- [laneq-lane-routing.md](laneq-lane-routing.md) — x"
    out = store.search("lane")
    paths = {hit["path"] for hit in out["hits"]}
    assert "broken.md" not in paths
    assert INDEX_FILENAME not in paths


# -- get / list / index --------------------------------------------------------


def test_get_returns_full_view() -> None:
    store, _ = _store()
    view = store.get("feedback-prefer-wif")
    assert view["name"] == "feedback-prefer-wif"
    assert view["type"] == "feedback"
    assert view["path"] == "feedback-prefer-wif.md"
    assert "WIF" in view["body"]
    assert view["metadata"]["node_type"] == "memory"


def test_get_missing_rejected() -> None:
    store, _ = _store()
    with pytest.raises(MemoryError, match="no such memory"):
        store.get("does-not-exist")


def test_get_bad_name_rejected() -> None:
    store, _ = _store()
    with pytest.raises(MemoryError, match="invalid name"):
        store.get("Not A Slug")


def test_empty_name_rejected() -> None:
    store, _ = _store()
    with pytest.raises(MemoryError, match="name must not be empty"):
        store.get("   ")


def test_too_long_description_rejected() -> None:
    store, _ = _store(seeded=False)
    with pytest.raises(MemoryError, match="too long"):
        store.write("m", "x" * 2001, "user", "b")


def test_list_all_and_by_type() -> None:
    store, _ = _store()
    all_out = store.list()
    assert all_out["count"] == 3
    names = [item["name"] for item in all_out["memories"]]
    assert names == sorted(names)  # name-sorted

    projects = store.list(type="project")
    assert {item["name"] for item in projects["memories"]} == {
        "laneq-lane-routing",
        "nash-credentials-custody",
    }


def test_list_skips_index_and_unparseable() -> None:
    store, storage = _store()
    storage.files[INDEX_FILENAME] = "- [x.md](x.md) — y"
    storage.files["junk.md"] = "no frontmatter"
    out = store.list()
    assert out["count"] == 3  # only the three real memories


def test_index_returns_pointer_lines() -> None:
    store, storage = _store(seeded=False)
    storage.files[INDEX_FILENAME] = "# Index\n\n- [a.md](a.md) — first\n- [b.md](b.md) — second\nprose line\n"
    out = store.index()
    assert out["count"] == 2
    assert out["lines"] == ["- [a.md](a.md) — first", "- [b.md](b.md) — second"]
    assert "# Index" in out["raw"]


def test_index_empty_when_no_index_file() -> None:
    store, _ = _store(seeded=False)
    out = store.index()
    assert out == {"count": 0, "lines": [], "raw": ""}


# -- write: creation + index ---------------------------------------------------


def test_write_creates_file_and_index_pointer() -> None:
    store, storage = _store(seeded=False)
    result = store.write(
        "new-memory",
        "a one line description",
        "project",
        "The body of the memory.",
    )
    assert result["path"] == "new-memory.md"
    assert result["indexed"] is True
    assert storage.exists("new-memory.md")

    # File round-trips through get.
    view = store.get("new-memory")
    assert view["description"] == "a one line description"
    assert view["type"] == "project"
    assert "The body" in view["body"]

    # Exactly one index pointer was added.
    index = store.index()
    assert index["count"] == 1
    assert index["lines"][0] == "- [new-memory.md](new-memory.md) — a one line description"


def test_write_with_links_appends_wiki_links_to_body() -> None:
    store, _ = _store(seeded=False)
    store.write("m", "desc", "user", "core body", links=["other-a", "other-b"])
    body = store.get("m")["body"]
    assert "core body" in body
    assert "[[other-a]]" in body
    assert "[[other-b]]" in body


def test_write_refuses_to_clobber_existing() -> None:
    store, _ = _store()
    with pytest.raises(MemoryError, match="already exists"):
        store.write("feedback-prefer-wif", "new desc", "feedback", "new body")


def test_write_rejects_traversal_name() -> None:
    store, _ = _store(seeded=False)
    # An invalid slug is caught by name validation before storage even sees it.
    with pytest.raises(MemoryError, match="invalid name"):
        store.write("../escape", "d", "user", "b")


def test_write_rejects_invalid_type() -> None:
    store, _ = _store(seeded=False)
    with pytest.raises(MemoryError, match="invalid type"):
        store.write("m", "d", "notatype", "b")


def test_write_rejects_empty_description() -> None:
    store, _ = _store(seeded=False)
    with pytest.raises(MemoryError, match="description must not be empty"):
        store.write("m", "   ", "user", "b")


def test_write_rejects_multiline_description() -> None:
    store, _ = _store(seeded=False)
    with pytest.raises(MemoryError, match="single line"):
        store.write("m", "line one\nline two", "user", "b")


# -- write: idempotence + no duplicate index line ------------------------------


def test_index_not_duplicated_on_repeated_persist() -> None:
    store, _ = _store(seeded=False)
    store.write("m", "first desc", "user", "body one")
    # Update twice; the pointer must be replaced in place, never duplicated.
    store.update("m", description="second desc")
    store.update("m", description="third desc")
    index = store.index()
    pointers = [line for line in index["lines"] if "(m.md)" in line]
    assert len(pointers) == 1
    assert pointers[0] == "- [m.md](m.md) — third desc"


def test_rewriting_same_content_is_idempotent() -> None:
    store, storage = _store(seeded=False)
    store.write("m", "desc", "user", "body")
    first_file = storage.files["m.md"]
    first_index = storage.files[INDEX_FILENAME]
    store.update("m", body="body")  # same body, same desc/type
    assert storage.files["m.md"] == first_file
    assert storage.files[INDEX_FILENAME] == first_index


def test_existing_duplicate_pointers_collapse_to_one_on_update() -> None:
    store, storage = _store(seeded=False)
    store.write("m", "desc", "user", "body")
    # Simulate a pre-existing duplicate pointer in the index.
    storage.files[INDEX_FILENAME] += "\n- [m.md](m.md) — stale duplicate\n"
    store.update("m", description="canonical")
    pointers = [line for line in store.index()["lines"] if "(m.md)" in line]
    assert pointers == ["- [m.md](m.md) — canonical"]


# -- update --------------------------------------------------------------------


def test_update_preserves_unspecified_fields_and_extra_metadata() -> None:
    store, _ = _store()
    # laneq-lane-routing is a project memory with node_type metadata.
    store.update("laneq-lane-routing", description="new description only")
    view = store.get("laneq-lane-routing")
    assert view["description"] == "new description only"
    assert view["type"] == "project"  # preserved
    assert view["metadata"]["node_type"] == "memory"  # preserved
    assert "routes work by lane" in view["body"]  # body preserved


def test_update_can_change_type_and_body() -> None:
    store, _ = _store()
    store.update("nash-credentials-custody", type="reference", body="new body text")
    view = store.get("nash-credentials-custody")
    assert view["type"] == "reference"
    assert view["body"] == "new body text"


def test_update_missing_memory_rejected() -> None:
    store, _ = _store(seeded=False)
    with pytest.raises(MemoryError, match="no such memory"):
        store.update("ghost", description="x")


def test_update_validates_new_type() -> None:
    store, _ = _store()
    with pytest.raises(MemoryError, match="invalid type"):
        store.update("nash-credentials-custody", type="bogus")


# -- in-RAM storage confinement (defence in depth) -----------------------------


def test_memory_storage_rejects_separator_paths() -> None:
    storage = MemoryStorage()
    with pytest.raises(StorageError):
        storage.read("a/b.md")
