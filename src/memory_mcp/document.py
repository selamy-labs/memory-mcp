"""Parse and serialise one markdown-memory document.

The store's on-disk format is a YAML frontmatter block delimited by ``---``
lines, followed by a markdown body. The frontmatter the fleet uses is small and
regular::

    ---
    name: some-slug
    description: one-line summary used for recall relevance
    metadata:
      node_type: memory
      type: feedback
      originSessionId: ...
    ---
    <body...>

This module parses *exactly that shape* with the standard library only -- no
PyYAML dependency, matching the zero-runtime-dependency design of the sibling
MCP servers. It is deliberately narrow: top-level ``key: value`` scalars and a
single nested ``metadata:`` block of ``key: value`` scalars. It is **not** a
general YAML parser; a document whose frontmatter uses unsupported YAML
(sequences, multi-line scalars) is reported as an error rather than silently
mis-read.

On write the serialiser preserves any unknown ``metadata`` keys it was given
(e.g. ``node_type``, ``originSessionId``) so round-tripping an existing memory
never drops data it did not understand.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

FRONTMATTER_FENCE = "---"

# The memory contract: these metadata.type values are the recall vocabulary.
VALID_TYPES = ("user", "feedback", "project", "reference")


class DocumentError(Exception):
    """A memory document could not be parsed or built from given fields."""


@dataclass(frozen=True)
class Memory:
    """One parsed memory: its contract fields, extra metadata, and body.

    ``name``/``description``/``type`` are the contract the recall layer depends
    on. ``extra_metadata`` carries any other ``metadata`` keys found on disk
    (``node_type``, ``originSessionId``, ...) so they survive a rewrite. ``body``
    is the markdown after the frontmatter, verbatim and unstripped of meaning.
    """

    name: str
    description: str
    type: str
    body: str
    extra_metadata: dict[str, str] = field(default_factory=dict)

    def haystack(self) -> str:
        """The lowercased text searched by :func:`memory_mcp.core` ranking."""
        return f"{self.name}\n{self.description}\n{self.body}".lower()


def _split_frontmatter(text: str) -> tuple[list[str], str]:
    """Split a document into its frontmatter lines and the body string.

    The document must start with a ``---`` fence and contain a closing ``---``.
    Returns ``(frontmatter_lines, body)``; raises :class:`DocumentError` when the
    fences are missing.
    """
    lines = text.splitlines()
    if not lines or lines[0].strip() != FRONTMATTER_FENCE:
        raise DocumentError("document does not start with a '---' frontmatter fence")
    for index in range(1, len(lines)):
        if lines[index].strip() == FRONTMATTER_FENCE:
            front = lines[1:index]
            body = "\n".join(lines[index + 1 :])
            # Preserve a single leading blank line convention without forcing it.
            return front, body.lstrip("\n")
    raise DocumentError("frontmatter is not closed by a second '---' fence")


def _parse_scalar(raw: str) -> str:
    """Strip matching surrounding quotes from a scalar value, else return as-is."""
    value = raw.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
        return value[1:-1]
    return value


def parse(text: str) -> Memory:
    """Parse a full memory document string into a :class:`Memory`.

    Tolerates the real store's extra ``metadata`` keys and quoted or unquoted
    scalar values. Raises :class:`DocumentError` for a malformed document or a
    missing/invalid contract field.
    """
    front, body = _split_frontmatter(text)

    top: dict[str, str] = {}
    metadata: dict[str, str] = {}
    in_metadata = False

    for line in front:
        if not line.strip():
            continue
        indented = line[0] in (" ", "\t")
        key_part, sep, value_part = line.partition(":")
        if not sep:
            raise DocumentError(f"unparseable frontmatter line: {line!r}")
        key = key_part.strip()
        value = _parse_scalar(value_part)

        if not indented:
            if key == "metadata":
                in_metadata = True
                if value:  # inline mapping not supported; flag rather than misread
                    raise DocumentError("inline 'metadata:' mapping is not supported")
                continue
            in_metadata = False
            top[key] = value
        else:
            if not in_metadata:
                raise DocumentError(f"unexpected indented frontmatter line: {line!r}")
            metadata[key] = value

    name = top.get("name", "").strip()
    description = top.get("description", "").strip()
    mem_type = metadata.get("type", "").strip()
    if not name:
        raise DocumentError("frontmatter is missing 'name'")
    if not description:
        raise DocumentError("frontmatter is missing 'description'")
    if not mem_type:
        raise DocumentError("frontmatter metadata is missing 'type'")

    extra = {k: v for k, v in metadata.items() if k != "type"}
    return Memory(name=name, description=description, type=mem_type, body=body, extra_metadata=extra)


def serialise(memory: Memory) -> str:
    """Render a :class:`Memory` back to the on-disk document format.

    Emits ``name``/``description`` at the top level and ``type`` plus any
    preserved ``extra_metadata`` under ``metadata``. The body is written after a
    blank line, with a single trailing newline.
    """
    out: list[str] = [FRONTMATTER_FENCE]
    out.append(f"name: {memory.name}")
    out.append(f"description: {memory.description}")
    out.append("metadata:")
    out.append(f"  type: {memory.type}")
    for key, value in memory.extra_metadata.items():
        out.append(f"  {key}: {value}")
    out.append(FRONTMATTER_FENCE)
    out.append("")
    body = memory.body.rstrip("\n")
    if body:
        out.append(body)
    text = "\n".join(out)
    return text + "\n"


def to_view(memory: Memory, *, name: str, path: str) -> dict[str, Any]:
    """A structured, JSON-serialisable view of a memory for tool output."""
    return {
        "name": name,
        "description": memory.description,
        "type": memory.type,
        "path": path,
        "metadata": dict(memory.extra_metadata),
        "body": memory.body,
    }
