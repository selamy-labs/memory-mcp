"""Tests for the stdlib markdown-memory document parser/serialiser, fully offline."""

from __future__ import annotations

import pytest

from memory_mcp.document import DocumentError, Memory, parse, serialise, to_view

REAL_SHAPE = """---
name: feedback-prefer-wif
description: "Prefer passwordless/tokenless auth — WIF over service-account keys"
metadata:
  node_type: memory
  type: feedback
  originSessionId: 1ff00fae-a1c3-4134-a508-9a8ac5c9fda1
---

Prefer **passwordless / tokenless auth.** Use WIF instead of SA JSON keys.

Related: [[feedback-prefer-opentofu]]
"""


def test_parse_real_store_shape() -> None:
    memory = parse(REAL_SHAPE)
    assert memory.name == "feedback-prefer-wif"
    assert memory.description.startswith("Prefer passwordless")
    assert memory.type == "feedback"
    # Unknown metadata keys are preserved, 'type' is lifted out.
    assert memory.extra_metadata == {
        "node_type": "memory",
        "originSessionId": "1ff00fae-a1c3-4134-a508-9a8ac5c9fda1",
    }
    assert "passwordless" in memory.body
    assert "[[feedback-prefer-opentofu]]" in memory.body


def test_round_trip_preserves_contract_and_extra_metadata() -> None:
    original = parse(REAL_SHAPE)
    text = serialise(original)
    reparsed = parse(text)
    assert reparsed.name == original.name
    assert reparsed.description == original.description
    assert reparsed.type == original.type
    assert reparsed.extra_metadata == original.extra_metadata
    assert reparsed.body == original.body


def test_serialise_emits_type_first_then_extra_metadata() -> None:
    memory = Memory(
        name="x-mem",
        description="a description",
        type="project",
        body="body text",
        extra_metadata={"node_type": "memory"},
    )
    text = serialise(memory)
    assert "metadata:\n  type: project\n  node_type: memory\n" in text
    assert text.endswith("body text\n")


def test_serialise_handles_empty_body() -> None:
    memory = Memory(name="x", description="d", type="user", body="")
    text = serialise(memory)
    # Closed frontmatter then a single trailing blank line; no body content.
    assert text == "---\nname: x\ndescription: d\nmetadata:\n  type: user\n---\n\n"
    assert "body" not in text


def test_haystack_is_lowercased_name_description_body() -> None:
    memory = Memory(name="My-Name", description="Desc TEXT", type="user", body="Body WORDS")
    hay = memory.haystack()
    assert "my-name" in hay
    assert "desc text" in hay
    assert "body words" in hay
    assert "My-Name" not in hay


def test_blank_lines_in_frontmatter_are_skipped() -> None:
    text = "---\nname: a\n\ndescription: d\n\nmetadata:\n  type: user\n---\nbody\n"
    memory = parse(text)
    assert memory.name == "a"
    assert memory.description == "d"
    assert memory.type == "user"


def test_unquoted_description_is_parsed() -> None:
    text = "---\nname: a\ndescription: plain text here\nmetadata:\n  type: user\n---\nbody\n"
    memory = parse(text)
    assert memory.description == "plain text here"


def test_missing_opening_fence_rejected() -> None:
    with pytest.raises(DocumentError, match="start with a '---'"):
        parse("name: a\ndescription: d\n")


def test_unclosed_frontmatter_rejected() -> None:
    with pytest.raises(DocumentError, match="not closed"):
        parse("---\nname: a\ndescription: d\n")


def test_missing_name_rejected() -> None:
    with pytest.raises(DocumentError, match="missing 'name'"):
        parse("---\ndescription: d\nmetadata:\n  type: user\n---\nb\n")


def test_missing_description_rejected() -> None:
    with pytest.raises(DocumentError, match="missing 'description'"):
        parse("---\nname: a\nmetadata:\n  type: user\n---\nb\n")


def test_missing_type_rejected() -> None:
    with pytest.raises(DocumentError, match="missing 'type'"):
        parse("---\nname: a\ndescription: d\nmetadata:\n  node_type: memory\n---\nb\n")


def test_unparseable_line_rejected() -> None:
    with pytest.raises(DocumentError, match="unparseable"):
        parse("---\nname a\ndescription: d\nmetadata:\n  type: user\n---\nb\n")


def test_inline_metadata_mapping_rejected() -> None:
    with pytest.raises(DocumentError, match="inline 'metadata:'"):
        parse("---\nname: a\ndescription: d\nmetadata: {type: user}\n---\nb\n")


def test_unexpected_indented_line_rejected() -> None:
    with pytest.raises(DocumentError, match="unexpected indented"):
        parse("---\nname: a\n  stray: value\ndescription: d\nmetadata:\n  type: user\n---\nb\n")


def test_to_view_shape() -> None:
    memory = parse(REAL_SHAPE)
    view = to_view(memory, name="feedback-prefer-wif", path="feedback-prefer-wif.md")
    assert view["name"] == "feedback-prefer-wif"
    assert view["type"] == "feedback"
    assert view["path"] == "feedback-prefer-wif.md"
    assert view["metadata"]["node_type"] == "memory"
    assert "body" in view
