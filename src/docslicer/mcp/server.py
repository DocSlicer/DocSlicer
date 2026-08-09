# SPDX-FileCopyrightText: 2026 Market Framer Inc.
# SPDX-License-Identifier: AGPL-3.0-only
"""docslicer MCP server — five tools.

    parse(source)                  -> doc_id + hierarchy.to_outline()
    get_outline(doc_id)            -> outline (refresh if lost in context)
    read(doc_id, headings=[...])   -> chunks_under() for each, as text
    search(doc_id, query)          -> headings to read, ranked, with snippets
    to_markdown(source | doc_id)   -> full document as markdown, written to disk

Run with::

    docslicer-mcp                 # stdio, for local clients
    docslicer-mcp --transport http --port 8000

Environment:
    DOCSLICER_MCP_CACHE         Where parsed results are persisted.
    DOCSLICER_MCP_FULL_TEXT     Token ceiling under which parse returns the whole
                                document instead of an outline (default 6000; 0
                                disables and always returns the outline).
    DOCSLICER_MCP_CACHE_MAX_MB  Cache size ceiling (default 2048; 0 disables pruning).
    DOCSLICER_MCP_ROOT          Restrict file sources and outputs to this directory tree.
    DOCSLICER_MCP_ALLOW_URLS    Set to 0 to reject http(s) sources.
"""

from __future__ import annotations

import argparse
import functools
import io
import math
import os
import re
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import TextIO

import anyio

from .. import __version__, parse_document
from .._result import Chunk, HierarchyNode
from ._store import (
    DocumentStore,
    allowed_root,
    is_url,
    make_doc_id,
    resolve_output,
    resolve_source,
    slug,
)

try:
    from mcp.server import MCPServer as _Server
    from mcp.types import ToolAnnotations
except ImportError as exc:  # pragma: no cover - depends on installed SDK
    raise ImportError(
        "The MCP server requires the 'mcp' package (2.0 or newer). Install it with:\n"
        "    pip install 'docslicer[mcp]'"
    ) from exc


INSTRUCTIONS = """\
To answer questions from a document:

1. `parse` it — you get a doc_id and the heading outline, each line marked with
   what reading it costs. A short document comes back as its full text instead,
   marked `is_complete: true`; you already have everything, so answer from it.
2. `read` the headings that look relevant, all in one call. Stop if the answer
   is there; otherwise read more — the outline is still valid.
3. `search` when no heading in the outline looks relevant to the question — 
   whether because all headings are non-descriptive ("Article 14"), the hierarchy is 
   malformed, or target facts aren't named in headings. It returns headings 
   to `read`, not answers.
4. `get_outline` if the outline scrolls out of your context. Cheap.

If the user instead wants the document itself as a file, `to_markdown` writes
it to disk and returns a path — nothing enters your context, so size does not
matter.
"""


mcp = _Server("docslicer", instructions=INSTRUCTIONS)

_store = DocumentStore()

# What merge_small_chunks joins absorbed headings with (step_08_chunk_builder).
_MERGED_SEP = " | "

# Separates a heading from its ancestors, both in requests and in path lines.
_PATH_SEP = ">"

# Under this many tokens, `parse` returns the document rather than its outline.
# Navigating costs an outline plus a round trip, and on a short document that
# overhead approaches what the whole text costs — while the outline it buys is
# a handful of lines that resolve nothing. See `_serve_whole`.
_FULL_TEXT_TOKENS = 6_000

# The ceiling the ratio rule in `_serve_whole` may reach, never the token gate.
_FULL_TEXT_CEILING = 2 * _FULL_TEXT_TOKENS


def _server_options() -> dict:
    """Chunking options for every parse on this server. Not model-facing."""
    return {
        "max_chunk_size": 3200,
        "optimal_chunk_size": 1500,
        "min_chunk_size": 700,
        "table_representation": "markdown",
        "include_headers_footers": False,
        "include_comments": False,
        # The outline's token figures are what the model budgets against, and
        # chars/4 misses by up to a third either way. Ships with the `mcp` extra;
        # degrades to estimation on its own if it cannot be used here.
        "exact_tokens": True,
    }


async def _to_thread(func, *args, **kwargs):
    return await anyio.to_thread.run_sync(functools.partial(func, *args, **kwargs))


# ===========================================================================
# Tools
# ===========================================================================


@mcp.tool(
    annotations=ToolAnnotations(
        title="Parse document",
        read_only_hint=True,
        # Accepts http(s) sources, so it can reach outside the local machine.
        open_world_hint=True,
    )
)
async def parse(
    source: str,
    password: str | None = None,
    refresh: bool = False,
) -> dict:
    """Parse a document and return its heading outline. Call this first.

    Local path or http(s) URL, format detected automatically.

    A short document returns as `text` — the whole thing, with `is_complete:
    true`. There is nothing left to fetch: `read` and `search` on it would
    return only what you are already holding. Answer from it directly.

    Everything else returns the outline, not the text. Each line carries the
    tokens `read` would return for
    that heading — "~48k" is a section to descend into, "~900" one to just
    read. The figure includes subsections, so a parent never costs less than
    the children under it. Use it to choose how deep to go before spending the
    context. `document_tokens` sizes the whole document, not this response —
    it is what reading everything would cost.

    HTML only: `renderer: "static"` means no browser rendered the page —
    scripts never ran and heading detection is weaker, so a thin outline may be
    incomplete.

    Args:
        source: File path or URL to parse.
        password: For an encrypted document.
        refresh: Re-parse even if an identical parse is cached.
    """
    resolved = resolve_source(source)
    options = _server_options()
    doc_id = make_doc_id(resolved, options)

    cached = None if refresh else _store.find(resolved, options)
    if cached:
        result = _store.get(cached)
    else:
        result = await _to_thread(
            parse_document,
            resolved,
            password=password,
            **options,
        )
        _store.put(doc_id, result, resolved, options)

    meta = result.metadata
    payload = {
        "doc_id": doc_id,
        "title": meta.title,
        "pages": meta.page_count,
        "headings": len(result.hierarchy.flatten()),
        "document_tokens": meta.token_count,
    }
    if _serve_whole(result):
        # The whole document, in `read`'s format — so a citation drawn from it
        # means what a citation drawn from `read` means.
        payload["text"] = _join_chunks(result.chunks)
        payload["is_complete"] = True
        payload["note"] = (
            "Short document — this is its full text, not an outline."
            " Nothing further to read; `read` and `search` would only return"
            " what is already here."
        )
    else:
        payload["outline"] = _outline(result)
        # Named for the document, not this response: nothing but the outline
        # entered context here, and the two figures are wildly different.
        payload["is_complete"] = False
    # HTML only — every other format has a single extraction path.
    if meta.renderer is not None:
        payload["renderer"] = meta.renderer
    return payload


@mcp.tool(
    annotations=ToolAnnotations(
        title="Get outline",
        read_only_hint=True,
        open_world_hint=False,
    )
)
async def get_outline(doc_id: str) -> dict:
    """Re-fetch a parsed document's outline, with per-heading read costs.

    Use when the outline has scrolled out of your context. Cheap.

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


@mcp.tool(
    annotations=ToolAnnotations(
        title="Read sections",
        read_only_hint=True,
        open_world_hint=False,
    )
)
async def read(doc_id: str, headings: list[str]) -> dict:
    """Return the text under one or more headings, chosen from the outline.

    Pass a heading's text exactly as the outline shows it. A heading includes
    its subsections, so it returns roughly the tokens the outline printed
    against it — prefer a subsection when the parent is large.

    Outline indentation shows nesting. If a heading appears in more than one
    place, prefix any ancestor with ">" to pick one — "Notes > Revenue". The
    full chain is not needed. Unqualified, it returns every match.

    A "[Page X]" line means the text below it is on page X until the next such
    line, numbered as the document numbers it ("S-23", "iv"). Cite the nearest
    one above what you quote — sections run over several pages, so the page a
    section opens on is right only for its first part.

    Args:
        doc_id: Handle from parse.
        headings: Exact outline headings, or "Parent > Heading". Pass every one
            you want in a single call.
    """
    result = _store.get(doc_id)

    sizes = _size_map(result)
    sections: list[str] = []
    tokens = 0
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
            # The same per-heading figure the outline quoted for this node, so
            # what a read cost matches what it was advertised to cost.
            tokens += sizes.get(node.heading_id, 0)
            sections.append(_format(node, body))

    return {
        "doc_id": doc_id,
        "text": "\n\n".join(sections),
        "tokens_returned": tokens,
    }


@mcp.tool(
    annotations=ToolAnnotations(
        title="Search document",
        read_only_hint=True,
        open_world_hint=False,
    )
)
async def search(doc_id: str, query: str, limit: int = 8) -> dict:
    """Find where something is discussed. Use when the outline is not enough.

    Try the outline first; it is faster. Search is for when it does not settle
    the question: headings that name nothing useful ("Note 14", "Item 7A"), a
    figure buried in a table no heading mentions, or a section too big to read
    whole.

    Returns places, not answers: each hit is a heading to pass to `read`, what
    reading it costs, and a snippet — a window cut around the match, flagged
    "snippet_is_partial" when cut from something longer.

    Usually `read` the heading. Stop at the snippet only when it answers the
    question alone and depends on nothing you cannot see. A partial snippet is
    missing what sat around it — often a table's header row, so figures appear
    without the column naming them. Never infer a truncated table's columns.

    Exact and related wording both match, so a quoted term ("NCT0123456") and a
    topic ("revenue by product") each work.

    Args:
        doc_id: Handle from parse.
        query: A term, a phrase, or a description.
        limit: Most sections to return, each a distinct place in the document.
    """
    result = _store.get(doc_id)
    found = await _to_thread(_search, result, query, limit)

    response = {
        "doc_id": doc_id,
        "query": query,
        "hits": found["hits"],
        # Repeated here rather than left to the tool description: the choice to
        # trust a snippet is made now, not when the description was read.
        "note": (
            "Pass a hit's 'heading' to read; 'read_tokens' is the cost."
            " A 'snippet_is_partial' hit was cut from a longer section — read it"
            " rather than assuming what was cut."
        ),
    }
    if found["unmatched_terms"]:
        response["unmatched_terms"] = found["unmatched_terms"]
        response["note"] += (
            " The listed words appear nowhere in the document, so these hits rest"
            " on the rest of the query — judge them by their snippets and re-query"
            " if they are beside the point."
        )
    return response


@mcp.tool(
    annotations=ToolAnnotations(
        title="Export to markdown file",
        # Writes to the filesystem and replaces an existing file at the
        # destination, so it is neither read-only nor non-destructive.
        read_only_hint=False,
        destructive_hint=True,
        # Re-running with the same arguments lands the same file.
        idempotent_hint=True,
        open_world_hint=True,
    )
)
async def to_markdown(
    source: str | None = None,
    doc_id: str | None = None,
    password: str | None = None,
    output_path: str | None = None,
) -> dict:
    """Write the whole document to disk as markdown. Returns a path, not the text.

    For when the user wants the document as a file rather than an answer drawn
    from it. Nothing enters your context, so size does not matter. Tables,
    charts, page markers and structure are preserved.

    Args:
        source: File path or URL. Omit if passing doc_id.
        doc_id: Handle from a previous parse — reuses it instead of parsing again.
        password: For an encrypted document. Ignored with doc_id.
        output_path: Where to write it. Absolute paths are used as given;
            relative paths resolve against the source file's directory.
            Defaults to the source filename with a .md extension, written next
            to the source (for a URL source, to a temp directory). Parent
            directories are created. An existing file is overwritten without
            prompting; the response sets `overwrote: true` when that happened,
            which is worth telling the user about. The absolute path is always
            returned.
    """
    # The source a doc_id was parsed from is carried on its cache record, so a
    # handle picks the same destination a fresh source would.
    if doc_id:
        origin = _store.source_of(doc_id)
    elif source:
        origin = resolve_source(source)
    else:
        raise ValueError("Pass either source (a path or URL) or doc_id from a previous parse.")

    # Resolved before the parse, not after: a destination the sandbox will
    # refuse should cost nothing to find out about.
    destination = _destination(output_path, origin)

    if doc_id:
        result = _store.get(doc_id)
    else:
        resolved = origin
        options = _server_options()
        cached = _store.find(resolved, options)
        if cached:
            doc_id = cached
            result = _store.get(cached)
        else:
            doc_id = make_doc_id(resolved, options)
            result = await _to_thread(
                parse_document,
                resolved,
                password=password,
                **options,
            )
            # Cached like any other parse, so a later parse/read/search on the
            # same document does not pay for it twice either.
            _store.put(doc_id, result, resolved, options)

    markdown = await _to_thread(result.export_to_markdown)

    # Checked before the write, so the caller can tell the user a file was
    # replaced rather than created.
    overwrote = destination.exists()

    destination.parent.mkdir(parents=True, exist_ok=True)
    await _to_thread(destination.write_text, markdown, encoding="utf-8")

    meta = result.metadata
    return {
        "doc_id": doc_id,
        "title": meta.title,
        "pages": meta.page_count,
        "output_path": str(destination),
        "markdown_size_bytes": len(markdown),
        "overwrote": overwrote,
        "document_tokens": meta.token_count,
    }


# ===========================================================================
# Outline sizing
# ===========================================================================


def _serve_whole(result) -> bool:
    """Whether `parse` should return the text itself rather than an outline.

    Two conditions, both bounded by a token ceiling that is never crossed —
    over-returning is the one failure mode that cannot be walked back, since
    the text is in the caller's context before they can judge it.

    The first is size. An outline earns its round trip by letting most of a
    document go unread; on a two-page memo there is nothing to leave out, and
    the outline plus a `read` call costs more than the memo did.

    The second is that the outline says nothing. A document with two headings,
    or one whose headings are so dense they cost a quarter of the text they
    index, is not navigable by outline whatever its size — that is the case
    where the outline is not merely wasteful but actively misleading, offering
    a structure to choose from that does not carve the document anywhere.
    """
    budget = _full_text_budget()
    if not budget:
        return False

    tokens = result.metadata.token_count or 0
    if tokens <= budget:
        return True

    if tokens > min(_FULL_TEXT_CEILING, 2 * budget):
        return False

    headings = len(result.hierarchy.flatten())
    # Estimated rather than counted: this is a ratio test against a quarter of
    # the document, and chars/4 is nowhere near that margin of error.
    outline_tokens = len(_outline(result)) // 4
    return headings < 3 or outline_tokens * 4 >= tokens


def _full_text_budget() -> int:
    """The token ceiling for returning whole text, overridable per deployment.

    A client with a 200k window and one with 32k do not want the same cutoff,
    and the operator knows which they are running.
    """
    raw = os.environ.get("DOCSLICER_MCP_FULL_TEXT")
    if raw is None:
        return _FULL_TEXT_TOKENS
    try:
        return max(0, int(raw))
    except ValueError:
        return _FULL_TEXT_TOKENS


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
        # Split the chunk's own token count by each heading's share of its
        # characters, rather than re-estimating the slices: a merged heading is
        # then counted on the same basis as every other one.
        chars = len(chunk.text) or 1
        for part in parts:
            shares[part] = round(chunk.token_count * len(_slice_merged(chunk, part)) / chars)
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


def _destination(output_path: str | None, origin: str | None) -> Path:
    """Where to_markdown writes, anchored on the source rather than our cwd.

    The caller cannot see this process's working directory, so anything derived
    from it — a bare default, or a relative output_path — produces a path they
    can neither predict beforehand nor find afterwards. The source file's own
    directory is a location both sides can name.

    The name comes from the source filename, not the document title: titles run
    long and carry `/`, `:` and newlines, and `azn.pdf -> azn.md` is what a
    caller would guess. A URL has no directory to sit beside, so it falls back
    to the sandbox root if one is configured and a temp directory otherwise,
    with the name slugified from the URL's last segment.
    """
    local = Path(origin) if origin and not is_url(origin) else None
    base = local.parent if local else (allowed_root() or Path(tempfile.gettempdir()))

    if output_path:
        return resolve_output(output_path, base=base)

    stem = local.stem if local else slug(origin or "document")
    return resolve_output(str(base / f"{stem}.md"))


# ===========================================================================
# Entrypoint
# ===========================================================================


class _StdoutToStderr(io.TextIOBase):
    """Text writes go to stderr; ``.buffer`` still points at the real stdout.

    Under stdio transport, stdout is the JSON-RPC wire: one stray ``print`` from
    the parsing pipeline (or any dependency) lands mid-frame and the client fails
    to parse the message. The transport writes frames through ``.buffer``, so it
    keeps the wire while everything text-level is pushed onto stderr, where a
    client shows it as server logs.
    """

    def __init__(self, real: TextIO) -> None:
        self._real = real

    @property
    def buffer(self):  # what the transport claims for the wire
        return self._real.buffer

    def write(self, s: str) -> int:
        return sys.stderr.write(s)

    def flush(self) -> None:
        sys.stderr.flush()

    def isatty(self) -> bool:
        return False

    def writable(self) -> bool:
        return True


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
        help="Restrict file sources and outputs to this directory tree (sets DOCSLICER_MCP_ROOT).",
    )
    parser.add_argument("--version", action="version", version=f"docslicer {__version__}")
    args = parser.parse_args(argv)

    if args.root:
        os.environ["DOCSLICER_MCP_ROOT"] = args.root

    if args.transport == "stdio":
        sys.stdout = _StdoutToStderr(sys.stdout)
        mcp.run(transport="stdio")
        return

    transport = "streamable-http" if args.transport == "http" else "sse"
    mcp.run(transport=transport, host=args.host, port=args.port)


if __name__ == "__main__":  # pragma: no cover
    main()