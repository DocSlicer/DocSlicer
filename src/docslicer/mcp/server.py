# SPDX-FileCopyrightText: 2026 Market Framer Inc.
# SPDX-License-Identifier: AGPL-3.0-only
"""docslicer MCP server — two tools.

    parse(source)                  -> doc_id + hierarchy.to_outline()
    read(doc_id, headings=[...])   -> chunks_under() for each, as text

Run with::

    docslicer-mcp                 # stdio, for local clients
    docslicer-mcp --transport http --port 8000

Environment:
    DOCSLICER_MCP_CACHE       Where parsed results are persisted.
    DOCSLICER_MCP_ROOT        Restrict file sources to this directory tree.
    DOCSLICER_MCP_ALLOW_URLS  Set to 0 to reject http(s) sources.
"""

from __future__ import annotations

import argparse
import contextlib
import functools
import os
import sys

import anyio

from .. import __version__, parse_document
from .._result import Chunk, HierarchyNode
from ._store import DocumentStore, make_doc_id, resolve_source

# The SDK renamed FastMCP to MCPServer in mcp 2.0; support both generations.
try:
    from mcp.server import MCPServer as _Server  # mcp >= 2.0

    _MCP_V2 = True
except ImportError:  # pragma: no cover - depends on installed SDK
    try:
        from mcp.server.fastmcp import FastMCP as _Server  # mcp 1.x

        _MCP_V2 = False
    except ImportError as exc:
        raise ImportError(
            "The MCP server requires the 'mcp' package. Install it with:\n"
            "    pip install 'docslicer[mcp]'"
        ) from exc


INSTRUCTIONS = """\
Read documents in two steps.

1. `parse` the document. You get its full heading outline.
2. `read` the headings you need. Normally pass each heading's text exactly as
   it appears in the outline. Pass all of them in one call.

   Indentation in the outline shows which headings sit under which. If a
   heading appears in more than one place, prefix a parent from above it,
   separated by ">", to say which one you mean:

       "Koselugo"                    ambiguous — appears twice
       "R&D progress > Koselugo"     the one under R&D progress

   Any ancestor will do; the full chain is not needed.

If what you read answers the question, stop. If it does not, read more
headings — the outline is still valid.
"""


mcp = _Server("docslicer", instructions=INSTRUCTIONS)

_store = DocumentStore()

# What merge_small_chunks joins absorbed headings with (step_08_chunk_builder).
_MERGED_SEP = " | "

# Separates a heading from its ancestors, both in requests and in path lines.
_PATH_SEP = ">"


def _server_options() -> dict:
    """Chunking options for every parse on this server. Not model-facing."""
    return {
        "max_chunk_size": 3200,
        "optimal_chunk_size": 1500,
        "min_chunk_size": 700,
        "table_representation": "markdown",
        "include_headers_footers": False,
        "include_comments": False,
    }


async def _to_thread(func, *args, **kwargs):
    return await anyio.to_thread.run_sync(functools.partial(func, *args, **kwargs))


@contextlib.contextmanager
def _stdout_guard():
    """Route stray library prints to stderr.

    Under the stdio transport, stdout *is* the JSON-RPC channel — a single
    stray print from deep in the pipeline would desynchronise the client.
    """
    with contextlib.redirect_stdout(sys.stderr):
        yield


# ===========================================================================
# Tools
# ===========================================================================


@mcp.tool()
async def parse(
    source: str,
    password: str | None = None,
    refresh: bool = False,
) -> dict:
    """Parse a document and return its heading outline. Call this first.

    Local path or http(s) URL; format detected automatically. Returns the
    outline, not the document text.

    Args:
        source: File path or URL to parse.
        password: Password for an encrypted document.
        refresh: Re-parse even if an identical parse is already cached.
    """
    resolved = resolve_source(source)
    options = _server_options()
    doc_id = make_doc_id(resolved, options)

    cached = None if refresh else _store.find(resolved, options)
    if cached:
        result = _store.get(cached)
    else:
        with _stdout_guard():
            result = await _to_thread(
                parse_document,
                resolved,
                password=password,
                **options,
            )
        _store.put(doc_id, result, resolved, options)

    meta = result.metadata
    return {
        "doc_id": doc_id,
        "title": meta.title,
        "pages": meta.page_count,
        "outline": result.hierarchy.to_outline(),
        "headings": len(result.hierarchy.flatten()),
        "tokens": meta.token_count,
    }


@mcp.tool()
async def read(doc_id: str, headings: list[str]) -> dict:
    """Return the text under one or more headings, chosen from the outline.

    Normally pass a heading's text exactly as it appears in the outline. Each
    heading includes its subsections.

    Indentation in the outline shows which headings sit under which. If a
    heading appears in more than one place, prefix a parent from above it,
    separated by ">", to say which one you mean — "R&D progress > Koselugo".
    Any ancestor will do; the full chain is not needed. An unqualified heading
    that matches several places returns all of them.

    Args:
        doc_id: Handle from parse.
        headings: Heading texts from the outline, exact, or qualified as
            "Parent > Heading". Pass every one you want in a single call.
    """
    result = _store.get(doc_id)

    sections: list[str] = []
    body_chars = 0
    seen: set[int] = set()
    for heading in headings:
        for node in _resolve(result, heading):
            if node.heading_id in seen:
                continue
            seen.add(node.heading_id)

            merged = _merged_chunk(result, node.text)
            body = (
                _slice_merged(merged, node.text)
                if merged is not None
                else _join_chunks(result.chunks_under(node))
            )
            body_chars += len(body)
            sections.append(_format(node, body))

    return {
        "doc_id": doc_id,
        "text": "\n\n".join(sections),
        "tokens": body_chars // 4,  # the chunk builder's own default estimate
    }


def _resolve(result, query: str) -> list[HierarchyNode]:
    """Headings a request names, most specific reading first.

    ``query`` is a heading, optionally qualified by ancestors the way the
    outline nests them and the way this tool prints its path lines:
    "R&D progress > Koselugo". Ancestors match as an ordered subsequence, so
    naming only the parent that distinguishes two identical headings is
    enough — the full chain is never required.

    An exact heading beats a substring: "Koselugo" is the medicine's own
    section, not "Agreement with Merck on Koselugo", which merely contains it.
    Substrings still resolve when nothing matches exactly, so a half-remembered
    heading is not a dead end. Ties are returned in full rather than silently
    resolved to the first — two headings with one text are usually two
    different subjects.
    """
    *ancestors, leaf = [part.strip() for part in query.split(_PATH_SEP)]
    leaf = leaf or query.strip()

    candidates = result.find_heading(leaf)
    exact = [n for n in candidates if n.text.strip().casefold() == leaf.casefold()]
    matches = exact or candidates

    if ancestors:
        matches = [n for n in matches if _under(n, ancestors)]
    return matches


def _under(node: HierarchyNode, ancestors: list[str]) -> bool:
    """Whether the node's path contains these ancestors, in order."""
    remaining = [a.casefold() for a in ancestors]
    for element in node.path:
        if remaining and remaining[0] in element.casefold():
            remaining.pop(0)
    return not remaining


def _merged_chunk(result, heading: str) -> Chunk | None:
    """The chunk several small headings were folded into, if this is one of them.

    merge_small_chunks joins the headings it absorbs with " | " into a single
    chunk_heading, and attributes the chunk to the first of them only — every
    other heading is left with no chunk_ids at all. Finding the chunk by its
    heading is what keeps those headings readable.
    """
    for chunk in result.chunks:
        parts = [p.strip() for p in (chunk.heading or "").split(_MERGED_SEP)]
        if len(parts) > 1 and heading in parts:
            return chunk
    return None


def _slice_merged(chunk: Chunk, heading: str) -> str:
    """The one heading's share of a merged chunk, from its marker to the next.

    Falls back to the whole chunk when the marker is missing: over-returning is
    recoverable, returning nothing is not.
    """
    lines = chunk.text.splitlines()
    marker = f"## {heading}"
    try:
        start = lines.index(marker)
    except ValueError:
        return chunk.text

    end = len(lines)
    for i in range(start + 1, len(lines)):
        if lines[i].startswith("## "):
            end = i
            break
    return "\n".join(lines[start:end]).strip()


def _join_chunks(chunks: list[Chunk]) -> str:
    """Chunks joined into one contiguous run.

    A heading whose content spans several chunks is repeated at the top of each
    one — right for RAG, where a chunk is retrieved alone, redundant here where
    the chunks are joined back together. Consecutive repeats are dropped; a new
    subsection heading is kept, since it is real structure.
    """
    parts: list[str] = []
    previous = ""
    for chunk in chunks:
        text = chunk.text
        first, _, rest = text.partition("\n\n")
        if first.startswith("## "):
            if first == previous:
                text = rest
            previous = first
        if text:
            parts.append(text)
    return "\n\n".join(parts)


def _format(node: HierarchyNode, body: str) -> str:
    """One section: its heading path, then its text."""
    path = f" {_PATH_SEP} ".join(node.path + [node.text])
    return f"{path}\n\n{body}"


# ===========================================================================
# Entrypoint
# ===========================================================================


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="docslicer-mcp",
        description="Run the docslicer MCP server.",
    )
    parser.add_argument(
        "--transport",
        default="stdio",
        choices=["stdio", "http", "sse"],
        help="stdio (default) for local clients; http for a networked server.",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Bind host for http/sse.")
    parser.add_argument("--port", type=int, default=8000, help="Bind port for http/sse.")
    parser.add_argument(
        "--root",
        help="Restrict file sources to this directory tree (sets DOCSLICER_MCP_ROOT).",
    )
    parser.add_argument("--version", action="version", version=f"docslicer {__version__}")
    args = parser.parse_args(argv)

    if args.root:
        os.environ["DOCSLICER_MCP_ROOT"] = args.root

    if args.transport == "stdio":
        mcp.run(transport="stdio")
        return

    transport = "streamable-http" if args.transport == "http" else "sse"
    if _MCP_V2:
        mcp.run(transport=transport, host=args.host, port=args.port)
    else:  # pragma: no cover - mcp 1.x took host/port off settings
        mcp.settings.host = args.host
        mcp.settings.port = args.port
        mcp.run(transport=transport)


if __name__ == "__main__":  # pragma: no cover
    main()
