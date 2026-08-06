# SPDX-FileCopyrightText: 2026 Market Framer Inc.
# SPDX-License-Identifier: AGPL-3.0-only
"""docslicer MCP server — four tools.

    parse(source)                  -> doc_id + hierarchy.to_outline()
    get_outline(doc_id)            -> outline (refresh if lost in context)
    read(doc_id, headings=[...])   -> chunks_under() for each, as text
    to_markdown(source)            -> full document as markdown

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
import math
import os
import re
import sys
from collections import Counter
from pathlib import Path

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
Read documents in steps.

1. `parse` the document. You get its full heading outline and a doc_id.

2. `read` the headings you need. Normally pass each heading's text exactly as
   it appears in the outline. Pass all of them in one call.

   Indentation in the outline shows which headings sit under which. If a
   heading appears in more than one place, prefix a parent from above it,
   separated by ">", to say which one you mean:

       "Consolidated Statement of Operations"                    ambiguous — appears twice
       "Financial Statements > Consolidated Statement of Operations"     the one under Financial Statements

   Any ancestor will do; the full chain is not needed.

Each outline line ends with what reading it costs, in tokens: "~48k" is a
section to descend into by naming a subsection under it, "~900" is one to
just read. The figure covers a heading's subsections, so a parent is never
cheaper than the children listed beneath it.

If what you read answers the question, stop. If it does not, read more
headings — the outline is still valid.

3. `get_outline` to refresh the outline if you've lost it from your context
   during a long conversation. It is fast and takes only the doc_id.

4. `search` when the outline does not tell you where to look — headings that
   name nothing useful ("Note 14"), a figure in a table no heading mentions,
   or a section too large to read whole.

   Search finds the place; it does not replace reading it. Each hit carries a
   snippet cut around the match, usually out of something longer — marked
   "snippet_is_partial" when so. Normally `read` the heading it names. Stop at
   the snippet only when it answers the question by itself: a figure whose
   table header you cannot see is not an answer, it is a guess.
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
    outline, not the document text. Every outline line carries an estimate of
    the tokens `read` would return for it, subsections included — use it to
    choose how deep to go before you spend the context.

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
        "outline": _outline(result),
        "headings": len(result.hierarchy.flatten()),
        "tokens": meta.token_count,
    }


@mcp.tool()
async def get_outline(doc_id: str) -> dict:
    """Retrieve the outline of a previously parsed document.

    Call this to refresh the outline in your context if you've lost it during
    a long conversation. The outline shows the document structure and estimated
    tokens for each section — use it to navigate and budget your reads.

    Args:
        doc_id: Handle from parse.
    """
    result = _store.get(doc_id)

    meta = result.metadata
    return {
        "doc_id": doc_id,
        "title": meta.title,
        "outline": _outline(result),
    }


@mcp.tool()
async def read(doc_id: str, headings: list[str]) -> dict:
    """Return the text under one or more headings, chosen from the outline.

    Normally pass a heading's text exactly as it appears in the outline. Each
    heading includes its subsections — so it returns roughly the token count
    the outline printed against it. Prefer a subsection when the parent is
    large.

    Indentation in the outline shows which headings sit under which. If a
    heading appears in more than one place, prefix a parent from above it,
    separated by ">", to say which one you mean — "Financial Statements > Consolidated Statement of Operations".
    Any ancestor will do; the full chain is not needed. An unqualified heading
    that matches several places returns all of them.

    A "[Page X]" line means everything below it is on page X, until the next
    such line. X is the page as the document itself numbers it ("S-23", "iv"),
    which is what a reader can look up. To cite something, use the nearest
    "[Page X]" above it — a section usually runs over several pages, so the
    page it opens on is the right citation only for its first part.

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
            if merged is not None:
                body = f"{_page_marker(merged)}\n\n{_slice_merged(merged, node.text)}"
            else:
                body = _join_chunks(result.chunks_under(node))
            body_chars += len(body)
            sections.append(_format(node, body))

    return {
        "doc_id": doc_id,
        "text": "\n\n".join(sections),
        "tokens": body_chars // 4,  # the chunk builder's own default estimate
    }


@mcp.tool()
async def to_markdown(
    source: str,
    password: str | None = None,
    output_path: str | None = None,
) -> dict:
    """Convert a document to markdown and save to disk.

    Uses the same powerful parser as parse/read/search but saves the entire
    document as markdown to a file. Does not load the markdown into context —
    only returns the file path. Use when you want the full document as a markdown
    file, especially for large documents with big tables/figures.

    Preserves tables, charts, page markers, and document structure.

    Args:
        source: File path or URL to convert.
        password: Password for an encrypted document.
        output_path: Where to save the markdown. If not provided, saves to
            current directory with an auto-generated name like "document.md".
    """
    resolved = resolve_source(source)
    options = _server_options()

    with _stdout_guard():
        result = await _to_thread(
            parse_document,
            resolved,
            password=password,
            **options,
        )

    markdown = result.export_to_markdown()

    # Generate output path if not provided
    if output_path is None:
        title = result.metadata.title or "document"
        # Sanitize title for filename
        safe_title = "".join(c if c.isalnum() or c in "-_ " else "" for c in title)
        safe_title = safe_title.strip().replace(" ", "_")[:50] or "document"
        output_path = f"{safe_title}.md"

    # Ensure parent directory exists
    path_obj = Path(output_path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)

    # Write markdown to disk
    await _to_thread(path_obj.write_text, markdown, encoding="utf-8")

    meta = result.metadata
    return {
        "title": meta.title,
        "pages": meta.page_count,
        "output_path": str(path_obj.absolute()),
        "markdown_size_bytes": len(markdown),
        "tokens": meta.token_count,
    }


@mcp.tool()
async def search(doc_id: str, query: str, limit: int = 8) -> dict:
    """Find where in the document something is discussed. Use when the outline is not enough.

    The outline is the faster route and should be tried first. Search is for
    when it does not settle the question: headings that name nothing useful
    ("Note 14", "Item 7A"), a figure buried in a table no heading mentions, or
    a section too large to read whole.

    Returns places, not prose: each hit is a heading you can pass straight to
    `read`, the tokens reading it would cost, and a snippet showing the match.

    A snippet is a window cut around the match, not the whole section. Having
    found the right place, decide between two things:

    - `read` that heading. This is the usual case. A snippet marked
      "snippet_is_partial" was cut out of something longer and is missing
      whatever sat around it — most importantly, a table's header row, so
      figures appear without the column that says what they are. Anything you
      would have to assume in order to use the snippet is a reason to read.
    - Stop, when the snippet answers the question on its own and nothing in it
      depends on context you cannot see.

    Never infer what a truncated table's columns are. Read the section.

    Matches exact wording and related wording both, so a quoted term
    ("NCT0123456", a trial ID, a number) and a description of a topic ("revenue
    by product") each work.

    Args:
        doc_id: Handle from parse.
        query: What to look for — a term, a phrase, or a description.
        limit: Most sections to return. Each is a distinct place in the document.
    """
    result = _store.get(doc_id)
    found = await _to_thread(_search, result, query, limit)

    response = {
        "doc_id": doc_id,
        "query": query,
        "hits": found["hits"],
        "note": (
            "Pass a hit's 'heading' to read; 'read_tokens' is what that costs."
            " Snippets marked 'snippet_is_partial' are windows into a longer"
            " section — read it rather than assuming what was cut."
        ),
    }
    if found["unmatched_terms"]:
        response["unmatched_terms"] = found["unmatched_terms"]
        response["note"] += (
            " These query words appear nowhere in the document, so the hits rest"
            " on the rest of the query — judge them by their snippets, and"
            " re-query if they are beside the point."
        )
    return response


# ===========================================================================
# Outline sizing
# ===========================================================================


def _outline(result) -> str:
    """The heading outline, each line marked with what reading it would cost.

    A bare outline says what a section is called but not what it costs to open.
    On a long document that is the difference between navigating and guessing:
    "Financial statements ~48k" is a section to descend into, "Note 14 ~900" is
    one to just read. The figure is the same estimate ``read`` reports back, so
    a budget made here holds there.

    Sizes are cumulative, matching what ``read`` returns: a heading's figure
    covers its subsections, and those subsections are itemised beneath it.
    """
    sizes = _size_map(result)
    lines: list[str] = []

    def _visit(node: HierarchyNode, depth: int) -> None:
        size = _human(sizes.get(node.heading_id, 0))
        lines.append(f"{'  ' * depth}- {node.text}  ~{size}")
        for child in node.children:
            _visit(child, depth + 1)

    for root in result.hierarchy.roots:
        _visit(root, 0)
    return "\n".join(lines)


def _size_map(result) -> dict[int, int]:
    """Estimated tokens per heading_id, subsections included.

    Mirrors what ``read`` assembles: ``chunks_under`` recursively, less the
    double counting of merged chunks, which name several headings but are
    attributed to the first of them alone.
    """
    tokens = {chunk.id: chunk.token_count for chunk in result.chunks}
    shares = _merged_shares(result)
    totals: dict[int, int] = {}

    def _visit(node: HierarchyNode) -> int:
        if node.text in shares:
            own = shares[node.text]
        else:
            own = sum(tokens.get(cid, 0) for cid in node.chunk_ids)
        total = own + sum(_visit(child) for child in node.children)
        totals[node.heading_id] = total
        return total

    for root in result.hierarchy.roots:
        _visit(root)
    return totals


def _merged_shares(result) -> dict[str, int]:
    """Per-heading token share of every merged chunk, keyed by heading text.

    Without this, a merged chunk counts once in full against whichever heading
    it was attributed to and zero against the rest — so one heading in the
    outline looks expensive and its neighbours look empty. Splitting the chunk
    the way ``read`` splits it puts each share on the right line.

    Keying by text is what ``_merged_chunk`` already does, and inherits its
    limit: two headings with identical text share one figure.
    """
    shares: dict[str, int] = {}
    for chunk in result.chunks:
        parts = [p.strip() for p in (chunk.heading or "").split(_MERGED_SEP)]
        if len(parts) < 2:
            continue
        for part in parts:
            shares[part] = len(_slice_merged(chunk, part)) // 4
    return shares


def _human(tokens: int) -> str:
    """A token count at the precision a budgeting decision actually needs."""
    if tokens >= 10_000:
        return f"{round(tokens / 1000)}k"
    if tokens >= 1_000:
        return f"{tokens / 1000:.1f}k"
    return str(tokens)


# ===========================================================================
# Search
# ===========================================================================

# Chunks that would match everything and locate nothing. A table of contents
# repeats every heading in the document, so it outranks the sections it names.
_SKIP_SECTIONS = frozenset({"toc", "header", "footer", "coverpage"})

# Carry no topic, and inflate every document's score against every query.
_STOPWORDS = frozenset(
    "a an and are as at be by for from has have in is it its of on or that the"
    " this to was were will with".split()
)

_WORD = re.compile(r"[\w'’]+")

_SNIPPET_CHARS = 320


def _search(result, query: str, limit: int) -> dict:
    """Sections matching a query, best first, plus any query terms the document
    does not contain.

    Two matchers, because one query is really two kinds of question. A literal
    match finds "NCT03785249", "€1.2m", a surname — the things BM25's tokenizer
    would split or its idf would misjudge. BM25 finds the sections about a topic
    the query only describes. Literal hits rank first: naming a string exactly
    is a stronger signal of intent than resembling one.

    One hit per heading. Results are places to `read`, and the same place twice
    is a wasted slot, not a stronger match.
    """
    limit = max(0, limit)
    chunks = [
        c
        for c in result.chunks
        if c.text.strip() and c.section not in _SKIP_SECTIONS
    ]
    if not chunks or not limit:
        return {"hits": [], "unmatched_terms": []}

    tokens = {c.id: _tokenize(c.text) for c in chunks}
    terms = _tokenize(query)
    pattern = _literal_pattern(query)

    scores = _bm25(tokens, terms)
    literal = {c.id for c in chunks if pattern and pattern.search(c.text)}

    ranked = sorted(
        (c for c in chunks if c.id in literal or c.id in scores),
        key=lambda c: (c.id not in literal, -scores.get(c.id, 0.0)),
    )

    sizes = _size_map(result)
    hits: list[dict] = []
    seen: set[str] = set()
    for chunk in ranked:
        if len(hits) >= limit:
            break
        anchor = _anchor(chunk.text, pattern, terms)
        heading = _hit_heading(chunk, anchor)
        if heading in seen:
            continue
        seen.add(heading)

        hit = {
            "heading": heading or "(untitled opening section)",
            "page": _page_ref(chunk),
            "snippet": _snippet(chunk.text, anchor),
        }
        # Stated rather than left to be inferred from the ellipses: a snippet
        # cut out of a table shows rows without the header row that names their
        # columns, and reads as complete when it is not.
        if len(chunk.text) > _SNIPPET_CHARS:
            hit["snippet_is_partial"] = True
        nodes = _resolve(result, heading) if heading else []
        if nodes:
            hit["read_tokens"] = sizes.get(nodes[0].heading_id, 0)
        hits.append(hit)

    return {"hits": hits, "unmatched_terms": _unmatched(tokens, terms)}


def _unmatched(tokens: dict[str, list[str]], terms: list[str]) -> list[str]:
    """Query terms that appear nowhere in the document.

    Score magnitude cannot tell a good query from a bad one — a nonsense query
    whose one real word happens to be rare outscores a sensible query about a
    common one, because idf rewards rarity, not relevance. Which terms simply
    are not in the document is the signal that survives that, and it needs no
    threshold. Reported rather than filtered on: a query with one absent term
    is often still a good query, and that is the caller's judgement to make.
    """
    if not terms:
        return []
    vocabulary: set[str] = set()
    for doc in tokens.values():
        vocabulary |= set(doc)
    return [t for t in dict.fromkeys(terms) if t not in vocabulary]


def _literal_pattern(query: str) -> re.Pattern | None:
    """The query as a whole-word literal match, or None if there is nothing to match.

    Bare substring matching made "AZ" match inside "AstraZeneca" — dozens of
    hits in a word the caller never asked about. Word boundaries are applied
    only at ends that are alphanumeric, so "€1.2m" and "(b)(ii)" still match:
    a boundary before "€" would demand a word character there and match nothing.
    """
    needle = query.strip()
    if not needle:
        return None
    lead = r"\b" if needle[0].isalnum() or needle[0] == "_" else ""
    tail = r"\b" if needle[-1].isalnum() or needle[-1] == "_" else ""
    return re.compile(lead + re.escape(needle) + tail, re.IGNORECASE)


def _tokenize(text: str) -> list[str]:
    """Words worth scoring. Unicode-aware, so accented text tokenizes normally."""
    return [t for t in _WORD.findall(text.casefold()) if t not in _STOPWORDS]


def _bm25(
    tokens: dict[str, list[str]],
    terms: list[str],
    k1: float = 1.5,
    b: float = 0.4,
) -> dict[str, float]:
    """Okapi BM25 over tokenized chunks, as {chunk_id: score} for the ones that match.

    Chunks are already the right retrieval unit — size-bounded and bounded by
    heading — so they are the documents here, with no separate index to build
    or keep in step with the parse.

    ``b`` controls how much a long chunk is penalised for its length. The usual
    0.75 is tuned for prose, and on business documents it buries tables: a
    six-row revenue table loses to a passing mention in a short paragraph,
    though the table is where the number actually is. 0.4 keeps the penalty —
    a term in a short chunk is still better evidence — without that inversion.
    """
    if not terms:
        return {}

    count = len(tokens)
    average = sum(len(t) for t in tokens.values()) / count or 1.0

    wanted = set(terms)
    frequency: Counter[str] = Counter()
    for doc in tokens.values():
        frequency.update(wanted & set(doc))

    # Rare terms carry the query; a term in every chunk distinguishes nothing.
    idf = {
        term: math.log(1 + (count - n + 0.5) / (n + 0.5))
        for term, n in frequency.items()
        if n
    }

    scores: dict[str, float] = {}
    for chunk_id, doc in tokens.items():
        counts = Counter(doc)
        length = len(doc)
        score = 0.0
        for term, weight in idf.items():
            f = counts[term]
            if f:
                # Saturating tf: the tenth mention adds far less than the second.
                score += weight * f * (k1 + 1) / (f + k1 * (1 - b + b * length / average))
        if score:
            scores[chunk_id] = score
    return scores


def _anchor(text: str, pattern: re.Pattern | None, terms: list[str]) -> int:
    """Where in the chunk the match is, for the snippet to centre on.

    The whole query if it appears verbatim, else the rarest-looking term that
    does. A snippet cut from the top of the chunk would often miss the match
    entirely — a chunk runs to 3200 characters.
    """
    lowered = text.casefold()
    if pattern:
        found = pattern.search(text)
        if found:
            return found.start()
    # Longer terms are the more distinctive ones; prefer their neighbourhood.
    for term in sorted(set(terms), key=len, reverse=True):
        at = lowered.find(term)
        if at >= 0:
            return at
    return 0


def _snippet(text: str, anchor: int, width: int = _SNIPPET_CHARS) -> str:
    """A window of the chunk around the match, cut at word boundaries."""
    if len(text) <= width:
        return text.strip()

    start = max(0, anchor - width // 3)
    end = min(len(text), start + width)
    if start:
        space = text.find(" ", start, anchor + 1)
        start = space + 1 if space != -1 else start
    if end < len(text):
        space = text.rfind(" ", anchor, end)
        end = space if space != -1 else end

    return (
        ("… " if start else "")
        + text[start:end].strip()
        + (" …" if end < len(text) else "")
    )


def _hit_heading(chunk: Chunk, anchor: int) -> str:
    """The chunk's heading path, in the form `read` accepts.

    Chunk paths carry markdown markers ("## EXECUTIVE COMMENTARY") that the
    outline and `read` do not use, so they are stripped. The full path is given
    rather than the bare heading: it is what disambiguates a heading that
    appears twice, and `read` ignores the ancestors it does not need.
    """
    parts = [p.lstrip("#").strip() for p in chunk.path]
    if not parts:
        return ""
    parts[-1] = _merged_part(chunk, parts[-1], anchor)
    return f" {_PATH_SEP} ".join(parts)


def _merged_part(chunk: Chunk, heading: str, anchor: int) -> str:
    """Which of a merged chunk's headings the match actually falls under.

    Pointing at the whole merged run would send `read` to the first heading in
    it, which may be a different subject entirely.
    """
    parts = [p.strip() for p in heading.split(_MERGED_SEP)]
    if len(parts) < 2:
        return heading
    for part in parts:
        marker = f"## {part}"
        start = chunk.text.find(marker)
        if start != -1 and start <= anchor < _next_marker(chunk.text, start + len(marker)):
            return part
    return parts[0]


def _next_marker(text: str, offset: int) -> int:
    """Where the next `## ` heading starts after offset, or the end of the text."""
    at = text.find("\n## ", offset)
    return len(text) if at == -1 else at


def _resolve(result, query: str) -> list[HierarchyNode]:
    """Headings a request names, most specific reading first.

    ``query`` is a heading, optionally qualified by ancestors the way the
    outline nests them and the way this tool prints its path lines:
    "Financial Statements > Consolidated Statement of Operations". Ancestors match as an ordered subsequence, so
    naming only the parent that distinguishes two identical headings is
    enough — the full chain is never required.

    An exact heading beats a substring: "Consolidated Statement of Operations" is the financial statement's own
    section, not "Management Discussion > Consolidated Statement of Operations", which merely contains it.
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
    """Chunks joined into one contiguous run, marked where the page turns.

    A heading whose content spans several chunks is repeated at the top of each
    one — right for RAG, where a chunk is retrieved alone, redundant here where
    the chunks are joined back together. Consecutive repeats are dropped; a new
    subsection heading is kept, since it is real structure.

    Page markers are what make the result citable. A section is regularly a
    dozen pages long, so a single page attributed to the whole of it would be
    wrong for everything after the first — a citation the reader follows to a
    page that does not hold the figure. Marking each turn lets a claim be
    traced to the page it actually came from.
    """
    parts: list[str] = []
    previous = ""
    page = None
    for chunk in chunks:
        text = chunk.text
        first, _, rest = text.partition("\n\n")
        if first.startswith("## "):
            if first == previous:
                text = rest
            previous = first
        if not text:
            continue
        if chunk.page_number != page:
            page = chunk.page_number
            parts.append(_page_marker(chunk))
        parts.append(text)
    return "\n\n".join(parts)


def _page_ref(chunk: Chunk) -> str:
    """How the document itself refers to the page a chunk is on.

    The label when there is one, the ordinal only as a fallback. On a filing
    whose 245th page prints "S-23", the ordinal is the one number a reader
    cannot look up — it appears nowhere in the document they are holding.
    Matches the fallback ``export_to_markdown`` uses for its page markers, so
    a citation means the same thing whichever route produced the text.
    """
    return chunk.page_label or str(chunk.page_number)


def _page_marker(chunk: Chunk) -> str:
    """Marks the text below it as being on this page.

    Deliberately not ``export_to_markdown``'s ``<!-- page 19 -->``, which reads
    as a page *ending* as readily as a page starting — an ambiguity that costs a
    citation its meaning when it resolves the wrong way. A bracketed label is
    what it says: everything under it is on that page, until the next one.
    """
    return f"[Page {_page_ref(chunk)}]"


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
