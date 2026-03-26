from __future__ import annotations

import io
from pathlib import Path
from typing import Union

from ._config import ParseConfig
from ._result import ParseResult, Chunk, Block, Table, DocMetadata
from ._orchestrator import _run_pipeline

__all__ = [
    "parse_pdf",
    "parse_html",
    "parse_document",
    "ParseConfig",
    "ParseResult",
    "Chunk",
    "Block",
    "Table",
    "DocMetadata",
]

_Source = Union[str, Path, bytes, io.IOBase]


def _load_bytes(source: _Source) -> bytes:
    if isinstance(source, bytes):
        return source
    if isinstance(source, io.IOBase):
        return source.read()
    path = Path(source)
    return path.read_bytes()


def _load_text(source: _Source) -> str:
    if isinstance(source, str) and not Path(source).exists():
        return source  # treat as raw HTML string
    if isinstance(source, io.IOBase):
        return source.read()
    path = Path(source)
    return path.read_text(encoding="utf-8")


def parse_pdf(
    source: _Source,
    max_chunk_size: int = 3200,
    optimal_chunk_size: int = 1500,
    extract_tables: bool = True,
    regions: list[str] | None = None,
    debug: bool = False,
) -> ParseResult:
    """Parse a PDF document and return a ParseResult.

    Args:
        source: PDF file path, raw bytes, or file-like object.
        max_chunk_size: Maximum characters per chunk (default 3200).
        optimal_chunk_size: Target characters per chunk (default 1500).
        extract_tables: Include table extraction (default True).
        regions: Filter output to these regions only, e.g. ["body", "toc"].
            Allowed values: body | header | footer | toc | exhibit.
            None means all regions (default).
        debug: Populate result.pipeline_steps with intermediate DataFrames.
    """
    config = ParseConfig(
        max_chunk_size=max_chunk_size,
        optimal_chunk_size=optimal_chunk_size,
        extract_tables=extract_tables,
        regions=regions,
        debug=debug,
    )
    content = _load_bytes(source)
    return _run_pipeline(content, "pdf", source_url=None, config=config)


def parse_html(
    source: _Source,
    source_url: str | None = None,
    max_chunk_size: int = 3200,
    optimal_chunk_size: int = 1500,
    extract_tables: bool = True,
    regions: list[str] | None = None,
    debug: bool = False,
) -> ParseResult:
    """Parse an HTML document and return a ParseResult.

    Args:
        source: HTML string, URL, file path, or file-like object.
            When a URL is passed (starts with http:// or https://), the page is
            fetched automatically — SEC/Congress URLs use a rate-limited fetcher,
            all others are rendered via Playwright for full JS execution.
        source_url: Original URL of the page (used for link normalisation when
            source is an HTML string or file, not a URL).
        max_chunk_size: Maximum characters per chunk (default 3200).
        optimal_chunk_size: Target characters per chunk (default 1500).
        extract_tables: Include table extraction (default True).
        regions: Filter output to these regions only.
        debug: Populate result.pipeline_steps with intermediate DataFrames.
    """
    config = ParseConfig(
        max_chunk_size=max_chunk_size,
        optimal_chunk_size=optimal_chunk_size,
        extract_tables=extract_tables,
        regions=regions,
        debug=debug,
    )
    # URL passed as source — route through the fetch pipeline
    if isinstance(source, str) and source.startswith(("http://", "https://")):
        return _run_pipeline(None, "html", source_url=source, config=config)
    content = _load_text(source)
    return _run_pipeline(content, "html", source_url=source_url, config=config)


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

    All keyword arguments are forwarded to parse_pdf or parse_html.
    """
    # Bytes: check magic bytes
    if isinstance(source, bytes):
        if source[:4] == b"%PDF":
            return parse_pdf(source, **kwargs)
        return parse_html(source.decode("utf-8", errors="replace"), source_url=source_url, **kwargs)

    # File-like: read once, then recurse
    if isinstance(source, io.IOBase):
        data = source.read()
        if isinstance(data, str):
            return parse_html(data, source_url=source_url, **kwargs)
        return parse_document(data, source_url=source_url, **kwargs)

    # Path or string — check extension first
    path = Path(source)
    if path.exists():
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            return parse_pdf(path, **kwargs)
        if suffix in (".html", ".htm", ".xhtml"):
            return parse_html(path, source_url=source_url, **kwargs)
        # Unknown extension: peek at magic bytes
        header = path.read_bytes()[:4]
        if header == b"%PDF":
            return parse_pdf(path, **kwargs)
        return parse_html(path, source_url=source_url, **kwargs)

    # Raw string that doesn't point to a file → treat as HTML
    return parse_html(str(source), source_url=source_url, **kwargs)
