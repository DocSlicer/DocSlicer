# SPDX-FileCopyrightText: 2026 Market Framer Inc.
# SPDX-License-Identifier: AGPL-3.0-only
"""docslicer — deterministic hierarchical document parser and chunker.

Parse PDF, DOCX, PPTX, and HTML (local files or URLs) into a structured
ParseResult of chunks, blocks, tables, charts, and a heading hierarchy.
"""

from __future__ import annotations

import atexit
import dataclasses
import io
import zipfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Callable, Iterator, Optional, Union

try:
    from importlib.metadata import version as _pkg_version
    __version__ = _pkg_version("docslicer")
except Exception:
    __version__ = "0.1.0"

from ._config import ParseConfig
from ._result import ParseResult, Chunk, Block, Table, TableCell, Chart, ChartPoint, BBox, HierarchyNode, HierarchyTree
from .metadata.schema import DocumentMetadata
from ._orchestrator import _run_pipeline

__all__ = [
    "parse_pdf",
    "parse_html",
    "parse_docx",
    "parse_pptx",
    "parse_document",
    "parse_all",
    "DocumentParser",
    "ParseConfig",
    "ParseResult",
    "Chunk",
    "Block",
    "Table",
    "TableCell",
    "Chart",
    "ChartPoint",
    "DocumentMetadata",
    "BBox",
    "HierarchyNode",
    "HierarchyTree",
]

_Source = Union[str, Path, bytes, io.IOBase]


def _is_url(value: str) -> bool:
    return value.startswith(("http://", "https://"))


def _extract_filename(source: _Source) -> str | None:
    """Return a filename from source when it can be inferred, otherwise None."""
    if isinstance(source, (str, Path)):
        s = str(source)
        if not _is_url(s):
            p = Path(s)
            if p.suffix:  # has an extension → looks like a real path
                return p.name
    if isinstance(source, io.IOBase) and hasattr(source, "name"):
        name = getattr(source, "name", None)
        if name:
            return Path(name).name
    return None


def _looks_like_raw_html(value: str) -> bool:
    sample = value.lstrip()[:512].lower()
    return (
        sample.startswith("<!doctype html")
        or sample.startswith("<html")
        or sample.startswith("<?xml")
        or "<html" in sample
        or "<body" in sample
    )


def _load_bytes(source: _Source) -> bytes:
    if isinstance(source, bytes):
        return source
    if isinstance(source, io.IOBase):
        return source.read()
    path = Path(source)
    return path.read_bytes()


def _load_text(source: _Source) -> str:
    if isinstance(source, str):
        if _looks_like_raw_html(source):
            return source
        try:
            path = Path(source)
            if not path.exists():
                return source  # treat as raw HTML string
        except OSError:
            return source

    if isinstance(source, io.IOBase):
        return source.read()

    path = Path(source)
    return path.read_text(encoding="utf-8")


def _detect_openxml_type(data: bytes) -> str | None:
    """Return docx/pptx for Office Open XML ZIP payloads, otherwise None."""
    if not data.startswith(b"PK"):
        return None
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            names = set(zf.namelist())
    except zipfile.BadZipFile:
        return None

    if "word/document.xml" in names:
        return "docx"
    if "ppt/presentation.xml" in names:
        return "pptx"
    return None


def _looks_like_pdf(data: bytes, content_type: str | None = None) -> bool:
    if data.lstrip()[:4] == b"%PDF":
        return True
    return bool(content_type and "application/pdf" in content_type.lower())


def _parse_fetched_url(url: str, **kwargs) -> ParseResult:
    """Fetch a URL once, then route by delivered content instead of URL shape."""
    from .scraping.dispatcher import _is_sec_url, fetch_url

    scraped = fetch_url(url)
    content_type = scraped.content_type or ""
    raw = scraped.raw_bytes

    if _looks_like_pdf(raw, content_type):
        return parse_pdf(raw, **kwargs)

    openxml_type = _detect_openxml_type(raw)
    if openxml_type == "docx":
        return parse_docx(raw, **kwargs)
    if openxml_type == "pptx":
        return parse_pptx(raw, **kwargs)

    if not _is_sec_url(scraped.final_url):
        return parse_html(scraped.final_url, **kwargs)

    html = raw.decode(scraped.encoding or "utf-8", errors="replace")
    return parse_html(html, source_url=scraped.final_url, **kwargs)


def parse_pdf(
    source: _Source,
    max_chunk_size: int = 3200,
    optimal_chunk_size: int = 1500,
    min_chunk_size: int = 700,
    chunking: bool = True,
    merge_small_chunks: bool = True,
    table_representation: str = "markdown",
    exact_tokens: bool = False,
    debug: bool = False,
    extra_fields: list[str] | None = None,
    password: str | None = None,
    max_workers: int | None = None,
    use_browser: bool = True,
    include_headers_footers: bool = False,
    include_footnotes: bool = True,
    include_comments: bool = False,
    include_speaker_notes: bool = True,
    on_stage: Optional[Callable[[str], None]] = None,
) -> ParseResult:
    """Parse a PDF document and return a ParseResult.

    Args:
        source: PDF file path, raw bytes, or file-like object.
        max_chunk_size: Maximum characters per chunk (default 3200).
        optimal_chunk_size: Target characters per chunk (default 1500).
        min_chunk_size: Soft minimum characters per chunk (default 700). Chunks
            below this size are merged when possible; short chunks at section
            boundaries may still be smaller.
        chunking: Build chunks from blocks (default True). Set to False to skip
            chunking and return only blocks, which is faster.
        merge_small_chunks: Merge chunks that fall below min_chunk_size into
            adjacent chunks (default True).
        table_representation: How tables are serialized into chunk text. One of
            "markdown" (default), "jsonl", or "melted".
        exact_tokens: Use tiktoken (cl100k_base) for exact token counts on each
            chunk and the document total (default False). Falls back to char/4
            estimation if tiktoken is not installed. Requires
            ``pip install tiktoken`` for exact counts.
        debug: Populate result.pipeline_steps with intermediate DataFrames.
        extra_fields: Additional pipeline DataFrame columns to attach to each
            Block and Chunk under their ``.extra`` dict, e.g.
            ["is_bold", "font_name", "font_size"]. Unknown columns get None.
        max_workers: Process-pool width for word extraction, cell building,
            and OCR within this document (default None -> auto, sized to the
            machine's performance cores). Set to 1 to keep this document
            single-process — e.g. when parsing many documents concurrently
            yourself and you want to avoid nested process pools.
        use_browser: Accepted for API consistency with parse_html; the PDF
            pipeline never uses a browser, so this is unused.
        include_headers_footers: Accepted for API consistency with parse_docx;
            unused for PDF.
        include_footnotes: Accepted for API consistency with parse_docx;
            unused for PDF.
        include_comments: Accepted for API consistency with parse_docx;
            unused for PDF.
        include_speaker_notes: Accepted for API consistency with parse_pptx;
            unused for PDF.
        on_stage: Optional callback invoked with a stage name (e.g.
            "extract_elements", "process_layouts", "extract_tables") as the
            pipeline progresses, for driving a progress indicator.
    """
    config = ParseConfig(
        max_chunk_size=max_chunk_size,
        optimal_chunk_size=optimal_chunk_size,
        min_chunk_size=min_chunk_size,
        chunking=chunking,
        merge_small_chunks=merge_small_chunks,
        table_representation=table_representation,
        exact_tokens=exact_tokens,
        debug=debug,
        extra_fields=extra_fields or [],
        password=password,
        max_workers=max_workers,
        use_browser=use_browser,
        include_headers_footers=include_headers_footers,
        include_footnotes=include_footnotes,
        include_comments=include_comments,
        include_speaker_notes=include_speaker_notes,
    )
    source_filename = _extract_filename(source)
    content = _load_bytes(source)
    return _run_pipeline(content, "pdf", source_url=None, config=config, source_filename=source_filename, on_stage=on_stage)


def parse_docx(
    source: _Source,
    max_chunk_size: int = 3200,
    optimal_chunk_size: int = 1500,
    min_chunk_size: int = 700,
    chunking: bool = True,
    merge_small_chunks: bool = True,
    table_representation: str = "markdown",
    exact_tokens: bool = False,
    debug: bool = False,
    extra_fields: list[str] | None = None,
    password: str | None = None,
    max_workers: int | None = None,
    use_browser: bool = True,
    include_headers_footers: bool = False,
    include_footnotes: bool = True,
    include_comments: bool = False,
    include_speaker_notes: bool = True,
    on_stage: Optional[Callable[[str], None]] = None,
) -> ParseResult:
    """Parse a DOCX document and return a ParseResult.

    Args:
        source: DOCX file path, raw bytes, or file-like object.
        max_chunk_size: Maximum characters per chunk (default 3200).
        optimal_chunk_size: Target characters per chunk (default 1500).
        min_chunk_size: Soft minimum characters per chunk (default 700).
        chunking: Build chunks from blocks (default True). Set to False to skip
            chunking and return only blocks, which is faster.
        merge_small_chunks: Merge chunks that fall below min_chunk_size into
            adjacent chunks (default True).
        table_representation: How tables are serialized into chunk text. One of
            "markdown" (default), "jsonl", or "melted".
        exact_tokens: Use tiktoken (cl100k_base) for exact token counts (default
            False). Falls back to char/4 estimation if tiktoken is not installed.
        debug: Populate result.pipeline_steps with intermediate DataFrames.
        extra_fields: Additional pipeline DataFrame columns to attach to each
            Block and Chunk under their ``.extra`` dict. Unknown columns get None.
        max_workers: Accepted for API consistency with parse_pdf; the DOCX
            pipeline has no intra-document parallel steps, so this is unused.
        use_browser: Accepted for API consistency with parse_html; the DOCX
            pipeline never uses a browser, so this is unused.
        include_headers_footers: Include header and footer content as blocks
            with block_type "header" / "footer" (default False).
        include_footnotes: Include footnotes and endnotes as extracted content
            (default True).
        include_comments: Include reviewer comments as extracted content
            (default False) — these are review annotations, not document
            content.
        include_speaker_notes: Accepted for API consistency with parse_pptx;
            unused for DOCX.
        on_stage: Optional callback invoked with a stage name (e.g.
            "extract_elements", "process_layouts") as the pipeline progresses,
            for driving a progress indicator.
    """
    config = ParseConfig(
        max_chunk_size=max_chunk_size,
        optimal_chunk_size=optimal_chunk_size,
        min_chunk_size=min_chunk_size,
        chunking=chunking,
        merge_small_chunks=merge_small_chunks,
        table_representation=table_representation,
        exact_tokens=exact_tokens,
        debug=debug,
        extra_fields=extra_fields or [],
        password=password,
        max_workers=max_workers,
        use_browser=use_browser,
        include_headers_footers=include_headers_footers,
        include_footnotes=include_footnotes,
        include_comments=include_comments,
        include_speaker_notes=include_speaker_notes,
    )
    source_filename = _extract_filename(source)
    content = _load_bytes(source)
    return _run_pipeline(content, "docx", source_url=None, config=config, source_filename=source_filename, on_stage=on_stage)


def parse_pptx(
    source: _Source,
    max_chunk_size: int = 3200,
    optimal_chunk_size: int = 1500,
    min_chunk_size: int = 700,
    chunking: bool = True,
    merge_small_chunks: bool = True,
    table_representation: str = "markdown",
    exact_tokens: bool = False,
    debug: bool = False,
    extra_fields: list[str] | None = None,
    password: str | None = None,
    max_workers: int | None = None,
    use_browser: bool = True,
    include_headers_footers: bool = False,
    include_footnotes: bool = True,
    include_comments: bool = False,
    include_speaker_notes: bool = True,
    on_stage: Optional[Callable[[str], None]] = None,
) -> ParseResult:
    """Parse a PPTX document and return a ParseResult.

    Args:
        source: PPTX file path, raw bytes, or file-like object.
        max_chunk_size: Maximum characters per chunk (default 3200).
        optimal_chunk_size: Target characters per chunk (default 1500).
        min_chunk_size: Soft minimum characters per chunk (default 700).
        chunking: Build chunks from blocks (default True). Set to False to skip
            chunking and return only blocks, which is faster.
        merge_small_chunks: Merge chunks that fall below min_chunk_size into
            adjacent chunks (default True).
        table_representation: How tables are serialized into chunk text. One of
            "markdown" (default), "jsonl", or "melted".
        exact_tokens: Use tiktoken (cl100k_base) for exact token counts (default
            False). Falls back to char/4 estimation if tiktoken is not installed.
        debug: Populate result.pipeline_steps with intermediate DataFrames.
        extra_fields: Additional pipeline DataFrame columns to attach to each
            Block and Chunk under their ``.extra`` dict. Unknown columns get None.
        max_workers: Accepted for API consistency with parse_pdf; the PPTX
            pipeline has no intra-document parallel steps, so this is unused.
        use_browser: Accepted for API consistency with parse_html; the PPTX
            pipeline never uses a browser, so this is unused.
        include_headers_footers: Accepted for API consistency with parse_docx;
            unused for PPTX.
        include_footnotes: Accepted for API consistency with parse_docx;
            unused for PPTX.
        include_comments: Accepted for API consistency with parse_docx;
            unused for PPTX.
        include_speaker_notes: Include speaker notes as extracted content
            (default True). Set to False to exclude them.
        on_stage: Optional callback invoked with a stage name (e.g.
            "extract_elements", "process_layouts") as the pipeline progresses,
            for driving a progress indicator.
    """
    config = ParseConfig(
        max_chunk_size=max_chunk_size,
        optimal_chunk_size=optimal_chunk_size,
        min_chunk_size=min_chunk_size,
        chunking=chunking,
        merge_small_chunks=merge_small_chunks,
        table_representation=table_representation,
        exact_tokens=exact_tokens,
        debug=debug,
        extra_fields=extra_fields or [],
        password=password,
        max_workers=max_workers,
        use_browser=use_browser,
        include_headers_footers=include_headers_footers,
        include_footnotes=include_footnotes,
        include_comments=include_comments,
        include_speaker_notes=include_speaker_notes,
    )
    source_filename = _extract_filename(source)
    content = _load_bytes(source)
    return _run_pipeline(content, "pptx", source_url=None, config=config, source_filename=source_filename, on_stage=on_stage)


def parse_html(
    source: _Source,
    source_url: str | None = None,
    max_chunk_size: int = 3200,
    optimal_chunk_size: int = 1500,
    min_chunk_size: int = 700,
    chunking: bool = True,
    merge_small_chunks: bool = True,
    table_representation: str = "markdown",
    exact_tokens: bool = False,
    debug: bool = False,
    extra_fields: list[str] | None = None,
    password: str | None = None,
    max_workers: int | None = None,
    use_browser: bool = True,
    include_headers_footers: bool = False,
    include_footnotes: bool = True,
    include_comments: bool = False,
    include_speaker_notes: bool = True,
    on_stage: Optional[Callable[[str], None]] = None,
) -> ParseResult:
    """Parse an HTML document and return a ParseResult.

    Args:
        source: HTML string, URL, file path, or file-like object.
            When a URL is passed (starts with http:// or https://), the page is
            fetched automatically — SEC/Congress URLs use a rate-limited fetcher,
            all others are rendered via Playwright for full JS execution.
            URL parsing requires two steps: ``pip install 'docslicer[html]'`` to
            install the Python package, then ``playwright install`` to download
            the browser binaries. Without the second step, parsing a URL raises
            an error even if the package is installed — unless use_browser=False.
        source_url: Original URL of the page (used for link normalisation when
            source is an HTML string or file, not a URL).
        max_chunk_size: Maximum characters per chunk (default 3200).
        optimal_chunk_size: Target characters per chunk (default 1500).
        min_chunk_size: Soft minimum characters per chunk (default 700).
        chunking: Build chunks from blocks (default True). Set to False to skip
            chunking and return only blocks, which is faster.
        merge_small_chunks: Merge chunks that fall below min_chunk_size into
            adjacent chunks (default True).
        table_representation: How tables are serialized into chunk text. One of
            "markdown" (default), "jsonl", or "melted".
        exact_tokens: Use tiktoken (cl100k_base) for exact token counts (default
            False). Falls back to char/4 estimation if tiktoken is not installed.
        debug: Populate result.pipeline_steps with intermediate DataFrames.
        extra_fields: Additional pipeline DataFrame columns to attach to each
            Block and Chunk under their ``.extra`` dict. Unknown columns get None.
        password: Accepted for API consistency with parse_pdf; unused for HTML.
        max_workers: Accepted for API consistency with parse_pdf; the HTML
            pipeline has no intra-document parallel steps, so this is unused.
        use_browser: Default True renders via Playwright (full JS execution,
            layout coordinates, CSS-resolved styling). Set to False to skip
            Playwright entirely and use the static (BeautifulSoup) box
            extractor instead — no browser launch, ~15x faster, and works
            without Playwright installed. Tradeoffs: no layout coordinates
            (x_right/y_top/y_bottom/width/height are all 0.0) and only
            inline-style + semantic-tag typography is resolved (no CSS class
            rules or external stylesheets, no JS-rendered content). URL
            sources are still fetched over plain HTTP — SEC/Congress URLs use
            the same rate-limited fetcher either way. Works well for
            inline-style-heavy documents (SEC filings, Word-exported HTML,
            legal documents); degrades on CSS-class-heavy modern pages. See
            docslicer.html.step_01_static_box_extractor for details.
        include_headers_footers: Accepted for API consistency with parse_docx;
            unused for HTML.
        include_footnotes: Accepted for API consistency with parse_docx;
            unused for HTML.
        include_comments: Accepted for API consistency with parse_docx;
            unused for HTML.
        include_speaker_notes: Accepted for API consistency with parse_pptx;
            unused for HTML.
        on_stage: Optional callback invoked with a stage name (e.g.
            "extract_elements", "process_layouts", "extract_tables") as the
            pipeline progresses, for driving a progress indicator.
    """
    config = ParseConfig(
        max_chunk_size=max_chunk_size,
        optimal_chunk_size=optimal_chunk_size,
        min_chunk_size=min_chunk_size,
        chunking=chunking,
        merge_small_chunks=merge_small_chunks,
        table_representation=table_representation,
        exact_tokens=exact_tokens,
        debug=debug,
        extra_fields=extra_fields or [],
        password=password,
        max_workers=max_workers,
        use_browser=use_browser,
        include_headers_footers=include_headers_footers,
        include_footnotes=include_footnotes,
        include_comments=include_comments,
        include_speaker_notes=include_speaker_notes,
    )
    # URL passed as source — route through the fetch pipeline
    if isinstance(source, str) and _is_url(source):
        return _run_pipeline(None, "html", source_url=source, config=config, on_stage=on_stage)
    source_filename = _extract_filename(source)
    content = _load_text(source)
    return _run_pipeline(content, "html", source_url=source_url, config=config, source_filename=source_filename, on_stage=on_stage)


def parse_document(
    source: _Source,
    source_url: str | None = None,
    **kwargs,
) -> ParseResult:
    """Auto-detect document type and parse.

    Detection order:
    1. Extension of a file path (.pdf / .html .htm)
    2. Magic bytes (PDF starts with ``%PDF``)
    3. Falls back to HTML

    All keyword arguments are forwarded to parse_pdf, parse_docx, parse_pptx, or parse_html.
    """
    # Bytes: check magic bytes
    if isinstance(source, bytes):
        if source[:4] == b"%PDF":
            return parse_pdf(source, **kwargs)
        openxml_type = _detect_openxml_type(source)
        if openxml_type == "docx":
            return parse_docx(source, **kwargs)
        if openxml_type == "pptx":
            return parse_pptx(source, **kwargs)
        return parse_html(source.decode("utf-8", errors="replace"), source_url=source_url, **kwargs)

    # File-like: read once, then recurse
    if isinstance(source, io.IOBase):
        data = source.read()
        if isinstance(data, str):
            return parse_html(data, source_url=source_url, **kwargs)
        return parse_document(data, source_url=source_url, **kwargs)

    # URL strings are HTML inputs, not filesystem paths.
    if isinstance(source, str) and _is_url(source):
        return _parse_fetched_url(source, **kwargs)

    if isinstance(source, str) and _looks_like_raw_html(source):
        return parse_html(source, source_url=source_url, **kwargs)

    # Path or string — check extension first
    try:
        path = Path(source)
    except OSError:
        return parse_html(str(source), source_url=source_url, **kwargs)
    if path.exists():
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            return parse_pdf(path, **kwargs)
        if suffix == ".docx":
            return parse_docx(path, **kwargs)
        if suffix == ".pptx":
            return parse_pptx(path, **kwargs)
        if suffix in (".html", ".htm", ".xhtml"):
            return parse_html(path, source_url=source_url, **kwargs)
        # Unknown extension: peek at magic bytes
        data = path.read_bytes()
        header = data[:4]
        if header == b"%PDF":
            return parse_pdf(path, **kwargs)
        openxml_type = _detect_openxml_type(data)
        if openxml_type == "docx":
            return parse_docx(path, **kwargs)
        if openxml_type == "pptx":
            return parse_pptx(path, **kwargs)
        if suffix in (".ppt", ".doc"):
            raise ValueError(f"Unsupported legacy Office format: {suffix}. Use .pptx/.docx.")
        return parse_html(path, source_url=source_url, **kwargs)

    # A nonexistent path that was clearly meant as a file — a Path object, or a
    # string with a document extension — is a mistake, not HTML content. Raise
    # instead of silently parsing the filename itself as HTML (a typo'd path
    # would otherwise return an empty-looking ParseResult). Extensionless or
    # markup-ish strings still fall back to HTML below.
    if isinstance(source, Path) or path.suffix.lower() in _DOCUMENT_SUFFIXES:
        raise FileNotFoundError(f"No such file: {source!r}")

    # Raw string that doesn't point to a file → treat as HTML
    return parse_html(str(source), source_url=source_url, **kwargs)


_SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".pptx", ".html", ".htm", ".xhtml"}

# File extensions that unambiguously denote a document on disk. A nonexistent
# path with one of these is treated as a missing file, not as HTML text.
_DOCUMENT_SUFFIXES = _SUPPORTED_EXTENSIONS | {".ppt", ".doc"}


def parse_all(
    source: "str | Path | list[_Source]",
    recursive: bool = False,
    **kwargs,
) -> "Iterator[tuple[_Source, ParseResult | Exception]]":
    """Parse all documents in a folder, or a list of sources, yielding (source, result) pairs.

    Args:
        source: A folder path to discover all supported files in, or a list of
            individual sources (paths, URLs, bytes).
        recursive: When source is a folder, search subdirectories too (default False).
        **kwargs: Forwarded to parse_document (max_chunk_size, extract_tables, …).

    Yields:
        ``(source, ParseResult)`` on success, ``(source, Exception)`` on failure,
        so a failed file never aborts the batch.

    Example::

        for path, result in parse_all("documents/", recursive=True, max_chunk_size=2000):
            if isinstance(result, Exception):
                print(f"Failed {path}: {result}")
            else:
                print(f"{path}: {len(result.chunks)} chunks")
    """
    if isinstance(source, list):
        items: list = source
    else:
        folder = Path(source)
        if not folder.is_dir():
            raise ValueError(
                f"{folder!r} is not a directory. "
                "Pass a list of paths/URLs to parse individual files."
            )
        glob_fn = folder.rglob if recursive else folder.glob
        items = sorted(
            p for p in glob_fn("*")
            if p.is_file() and p.suffix.lower() in _SUPPORTED_EXTENSIONS
        )

    for item in items:
        try:
            yield item, parse_document(item, **kwargs)
        except Exception as exc:
            yield item, exc


_worker_parser: "DocumentParser | None" = None


def _init_document_parser_worker(config: ParseConfig) -> None:
    """ProcessPoolExecutor initializer: build one DocumentParser per worker process.

    A DocumentParser can hold a live Playwright Browser once it has parsed HTML,
    and Browser objects aren't picklable, so worker processes each build their
    own instance (and their own browser, lazily) rather than inheriting the
    parent's. atexit closes it so a Chromium subprocess isn't orphaned when the
    pool shuts the worker down.
    """
    global _worker_parser
    _worker_parser = DocumentParser(config)
    atexit.register(_worker_parser.close)


def _document_parser_worker_parse(source: "_Source"):
    try:
        return source, _worker_parser.parse(source)
    except Exception as exc:
        return source, exc


class DocumentParser:
    """Reusable parser that holds a fixed ParseConfig across multiple documents.

    Also holds a single Playwright browser open across documents for the HTML
    pipeline, so a batch of HTML/URL inputs launches Chromium once instead of
    once per document. The browser launches lazily on the first HTML extraction
    (PDF/DOCX/PPTX parses never start one) and is released by :meth:`close`.
    Use as a context manager to guarantee cleanup::

        with DocumentParser(config) as parser:
            for path, result in parser.parse_all(paths):
                ...

    Pass ``workers`` to parse multiple whole documents in parallel processes
    (in addition to, or instead of, intra-document parallelism controlled by
    ``config.max_workers``). Each worker process builds its own DocumentParser
    (and its own browser, if it hits HTML). To avoid oversubscribing the
    machine with nested process pools, a worker's own ``config.max_workers``
    defaults to 1 when ``workers`` is set and the caller hasn't explicitly
    picked a value::

        with DocumentParser(config, workers=4) as parser:
            for path, result in parser.parse_all(paths):
                ...
    """

    def __init__(self, config: ParseConfig | None = None, workers: int | None = None) -> None:
        self.config = config or ParseConfig()
        self.workers = workers
        self._session = None

    def _get_session(self):
        """Lazily create the reusable browser session shared across parses.

        Returns ``None`` when Playwright is not installed; the HTML pipeline then
        raises its usual install hint. The session object is created eagerly but
        the browser itself launches lazily on first HTML extraction.
        """
        if self._session is None:
            try:
                from .html.step_01_box_extractor import BrowserSession
            except ImportError:
                return None
            self._session = BrowserSession()
        return self._session

    def close(self) -> None:
        """Close the reusable browser session, if one was started."""
        if self._session is not None:
            self._session.close()
            self._session = None

    def __enter__(self) -> "DocumentParser":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def parse(
        self,
        source: _Source,
        source_url: str | None = None,
        on_stage: Optional[Callable[[str], None]] = None,
    ) -> ParseResult:
        """Parse a single document, reusing this instance's config and browser.

        on_stage: Optional callback invoked with a stage name as the pipeline
            progresses. Only supported on this single-process path — the
            parse_all(workers=N) fan-out can't forward callbacks across
            process boundaries, the same way it can't share this instance's
            browser session across processes either.
        """
        from functools import partial
        _run = partial(_run_pipeline, config=self.config, session=self._get_session(), on_stage=on_stage)

        if isinstance(source, bytes):
            if source[:4] == b"%PDF":
                return _run(source, "pdf", source_url=None)
            openxml_type = _detect_openxml_type(source)
            if openxml_type == "docx":
                return _run(source, "docx", source_url=None)
            if openxml_type == "pptx":
                return _run(source, "pptx", source_url=None)
            return _run(source.decode("utf-8", errors="replace"), "html", source_url=source_url)

        if isinstance(source, io.IOBase):
            data = source.read()
            if isinstance(data, str):
                return _run(data, "html", source_url=source_url)
            return self.parse(data, source_url=source_url)

        if isinstance(source, str) and _is_url(source):
            from .scraping.dispatcher import _is_sec_url, fetch_url
            scraped = fetch_url(source)
            raw = scraped.raw_bytes
            ct = scraped.content_type or ""
            if _looks_like_pdf(raw, ct):
                return _run(raw, "pdf", source_url=None)
            openxml_type = _detect_openxml_type(raw)
            if openxml_type == "docx":
                return _run(raw, "docx", source_url=None)
            if openxml_type == "pptx":
                return _run(raw, "pptx", source_url=None)
            final_url = scraped.final_url
            if not _is_sec_url(final_url):
                return _run(None, "html", source_url=final_url)
            html = raw.decode(scraped.encoding or "utf-8", errors="replace")
            return _run(html, "html", source_url=final_url)

        if isinstance(source, str) and _looks_like_raw_html(source):
            return _run(source, "html", source_url=source_url)

        try:
            path = Path(source)
        except OSError:
            return _run(str(source), "html", source_url=source_url)

        if path.exists():
            suffix = path.suffix.lower()
            if suffix == ".pdf":
                return _run(path.read_bytes(), "pdf", source_url=None)
            if suffix == ".docx":
                return _run(path.read_bytes(), "docx", source_url=None)
            if suffix == ".pptx":
                return _run(path.read_bytes(), "pptx", source_url=None)
            if suffix in (".html", ".htm", ".xhtml"):
                return _run(path.read_text(encoding="utf-8"), "html", source_url=source_url)
            data = path.read_bytes()
            if data[:4] == b"%PDF":
                return _run(data, "pdf", source_url=None)
            openxml_type = _detect_openxml_type(data)
            if openxml_type == "docx":
                return _run(data, "docx", source_url=None)
            if openxml_type == "pptx":
                return _run(data, "pptx", source_url=None)
            if suffix in (".ppt", ".doc"):
                raise ValueError(f"Unsupported legacy Office format: {suffix}. Use .pptx/.docx.")
            return _run(path.read_text(encoding="utf-8"), "html", source_url=source_url)

        return _run(str(source), "html", source_url=source_url)

    def parse_all(self, sources: list[_Source], /) -> "Iterator[tuple[_Source, ParseResult | Exception]]":
        """Parse multiple documents, reusing this instance's config and browser.

        Yields ``(source, ParseResult)`` on success, ``(source, Exception)`` on
        failure — a failed file never aborts the batch.

        With ``workers`` unset (default), documents are parsed one at a time in
        this process, lazily, reusing a single browser across all sources; call
        :meth:`close` (or use the parser as a context manager) when done to
        release it.

        With ``workers`` set, documents are fanned out across that many worker
        processes (each with its own browser) via a ``ProcessPoolExecutor``, and
        results arrive in submission order once the whole batch is scheduled —
        this path is not lazy per-document the way the single-worker path is.
        """
        if self.workers is None or self.workers <= 1:
            for source in sources:
                try:
                    yield source, self.parse(source)
                except Exception as exc:
                    yield source, exc
            return

        worker_config = self.config
        if worker_config.max_workers is None:
            worker_config = dataclasses.replace(worker_config, max_workers=1)

        with ProcessPoolExecutor(
            max_workers=self.workers,
            initializer=_init_document_parser_worker,
            initargs=(worker_config,),
        ) as ex:
            yield from ex.map(_document_parser_worker_parse, sources)
