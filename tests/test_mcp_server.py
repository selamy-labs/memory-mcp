"""Tests for the thin MCP wrapper, fully offline.

The server module is a thin adapter over :class:`memory_mcp.core.MemoryStore`:
it resolves config, maps :class:`MemoryError` to ``ToolError``, and registers
the tools. Tests inject an in-RAM-backed store via ``set_store`` so nothing
touches the configured disk, and assert the wrapper's mapping and registration.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.exceptions import ToolError

from memory_mcp import mcp_server
from memory_mcp.core import MemoryStore
from tests.conftest import FakeClock, MemoryStorage, sample_memories


@pytest.fixture(autouse=True)
def _reset_store() -> None:
    mcp_server.set_store(None)
    yield
    mcp_server.set_store(None)


def _install_fake() -> MemoryStorage:
    storage = MemoryStorage()
    for memory in sample_memories():
        storage.seed(memory)
    mcp_server.set_store(MemoryStore(storage, clock=FakeClock()))
    return storage


def test_read_tools_round_trip() -> None:
    _install_fake()
    search = mcp_server.memory_search("laneq")
    assert search["hits"][0]["name"] == "laneq-lane-routing"

    got = mcp_server.memory_get("feedback-prefer-wif")
    assert got["type"] == "feedback"

    listing = mcp_server.memory_list(type="project")
    assert {m["name"] for m in listing["memories"]} == {
        "laneq-lane-routing",
        "nash-credentials-custody",
    }


def test_write_then_get_and_index_via_tools() -> None:
    _install_fake()
    out = mcp_server.memory_write("tool-made", "a description", "user", "body", links=["laneq-lane-routing"])
    assert out["indexed"] is True
    body = mcp_server.memory_get("tool-made")["body"]
    assert "[[laneq-lane-routing]]" in body
    index = mcp_server.memory_index()
    assert any("(tool-made.md)" in line for line in index["lines"])


def test_update_via_tool() -> None:
    _install_fake()
    mcp_server.memory_update("nash-credentials-custody", description="changed")
    assert mcp_server.memory_get("nash-credentials-custody")["description"] == "changed"


def test_clobber_maps_to_tool_error() -> None:
    _install_fake()
    with pytest.raises(ToolError, match="already exists"):
        mcp_server.memory_write("feedback-prefer-wif", "d", "feedback", "b")


def test_missing_get_maps_to_tool_error() -> None:
    _install_fake()
    with pytest.raises(ToolError, match="no such memory"):
        mcp_server.memory_get("ghost")


def test_traversal_name_maps_to_tool_error() -> None:
    _install_fake()
    with pytest.raises(ToolError, match="invalid name"):
        mcp_server.memory_write("../escape", "d", "user", "b")


def test_build_server_registers_all_tools() -> None:
    _install_fake()
    server = mcp_server.build_server()
    assert server.name == "memory-mcp"
    names = {tool.name for tool in server._tool_manager.list_tools()}
    assert {
        "memory_search",
        "memory_get",
        "memory_list",
        "memory_index",
        "memory_write",
        "memory_update",
    } <= names


def test_store_is_built_from_env_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # No store pre-installed -> the server builds one from MEMORY_ROOT and writes
    # a real file into the temp root, proving the env wiring end to end.
    monkeypatch.setenv("MEMORY_ROOT", str(tmp_path))
    mcp_server.set_store(None)
    mcp_server.memory_write("env-made", "from env root", "reference", "body")
    assert (tmp_path / "env-made.md").is_file()
    assert (tmp_path / "MEMORY.md").is_file()


def test_default_root_when_env_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MEMORY_ROOT", raising=False)
    assert mcp_server._memory_root() == mcp_server.DEFAULT_MEMORY_ROOT


def test_main_runs_the_built_server(monkeypatch: pytest.MonkeyPatch) -> None:
    ran: list[bool] = []
    monkeypatch.setattr(FastMCP, "run", lambda self, *a, **k: ran.append(True))
    mcp_server.main()
    assert ran == [True]
