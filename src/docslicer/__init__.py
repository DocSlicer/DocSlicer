from __future__ import annotations

import io
import zipfile
from pathlib import Path
from typing import Iterator, Union

try:
    from importlib.metadata import version as _pkg_version
    __version__ = _pkg_version("docslicer")
except Exception:
    __version__ = "0.1.0"

from ._config import ParseConfig
from ._result import ParseResult, Chunk, Block, Table, TableCell, BBox, HierarchyNode, HierarchyTree
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
    extract_tables: bool = True,
    chunking: bool = True,
    regions: list[str] | None = None,
    debug: bool = False,
    extra_fields: list[str] | None = None,
) -> ParseResult:
    """Parse a PDF document and return a ParseResult.

    Args:
        source: PDF file path, raw bytes, or file-like object.
        max_chunk_size: Maximum characters per chunk (default 3200).
        optimal_chunk_size: Target characters per chunk (default 1500).
        extract_tables: Include table extraction (default True).
        chunking: Build chunks from blocks (default True). Set to False to skip
            chunking and return only blocks, which is faster.
        regions: Filter output to these regions only, e.g. ["body", "toc"].
            Allowed values: body | header | footer | toc | exhibit.
            None means all regions (default).
        debug: Populate result.pipeline_steps with intermediate DataFrames.
        extra_fields: Additional pipeline DataFrame columns to attach to each
            Block and Chunk under their ``.extra`` dict, e.g.
            ["is_bold", "font_name", "font_size"]. Unknown columns get None.
    """
    config = ParseConfig(
        max_chunk_size=max_chunk_size,
        optimal_chunk_size=optimal_chunk_size,
        extract_tables=extract_tables,
        chunking=chunking,
        regions=regions,
        debug=debug,
        extra_fields=extra_fields or [],
    )
    source_filename = _extract_filename(source)
    content = _load_bytes(source)
    return _run_pipeline(content, "pdf", source_url=None, config=config, source_filename=source_filename)


def parse_docx(
    source: _Source,
    max_chunk_size: int = 3200,
    optimal_chunk_size: int = 1500,
    extract_tables: bool = True,
    chunking: bool = True,
    regions: list[str] | None = None,
    debug: bool = False,
    extra_fields: list[str] | None = None,
) -> ParseResult:
    """Parse a DOCX document and return a ParseResult.

    Args:
        source: DOCX file path, raw bytes, or file-like object.
        max_chunk_size: Maximum characters per chunk (default 3200).
        optimal_chunk_size: Target characters per chunk (default 1500).
        extract_tables: Include table extraction (default True).
        chunking: Build chunks from blocks (default True). Set to False to skip
            chunking and return only blocks, which is faster.
        regions: Filter output to these regions only.
        debug: Populate result.pipeline_steps with intermediate DataFrames.
        extra_fields: Additional pipeline DataFrame columns to attach to each
            Block and Chunk under their ``.extra`` dict. Unknown columns get None.
    """
    config = ParseConfig(
        max_chunk_size=max_chunk_size,
        optimal_chunk_size=optimal_chunk_size,
        extract_tables=extract_tables,
        chunking=chunking,
        regions=regions,
        debug=debug,
        extra_fields=extra_fields or [],
    )
    source_filename = _extract_filename(source)
    content = _load_bytes(source)
    return _run_pipeline(content, "docx", source_url=None, config=config, source_filename=source_filename)


def parse_pptx(
    source: _Source,
    max_chunk_size: int = 3200,
    optimal_chunk_size: int = 1500,
    extract_tables: bool = True,
    chunking: bool = True,
    regions: list[str] | None = None,
    debug: bool = False,
    extra_fields: list[str] | None = None,
) -> ParseResult:
    """Parse a PPTX document and return a ParseResult.

    Args:
        source: PPTX file path, raw bytes, or file-like object.
        max_chunk_size: Maximum characters per chunk (default 3200).
        optimal_chunk_size: Target characters per chunk (default 1500).
        extract_tables: Include table extraction (default True).
        chunking: Build chunks from blocks (default True). Set to False to skip
            chunking and return only blocks, which is faster.
        regions: Filter output to these regions only.
        debug: Populate result.pipeline_steps with intermediate DataFrames.
        extra_fields: Additional pipeline DataFrame columns to attach to each
            Block and Chunk under their ``.extra`` dict. Unknown columns get None.
    """
    config = ParseConfig(
        max_chunk_size=max_chunk_size,
        optimal_chunk_size=optimal_chunk_size,
        extract_tables=extract_tables,
        chunking=chunking,
        regions=regions,
        debug=debug,
        extra_fields=extra_fields or [],
    )
    source_filename = _extract_filename(source)
    content = _load_bytes(source)
    return _run_pipeline(content, "pptx", source_url=None, config=config, source_filename=source_filename)


def parse_html(
    source: _Source,
    source_url: str | None = None,
    max_chunk_size: int = 3200,
    optimal_chunk_size: int = 1500,
    extract_tables: bool = True,
    chunking: bool = True,
    regions: list[str] | None = None,
    debug: bool = False,
    extra_fields: list[str] | None = None,
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
        chunking: Build chunks from blocks (default True). Set to False to skip
            chunking and return only blocks, which is faster.
        regions: Filter output to these regions only.
        debug: Populate result.pipeline_steps with intermediate DataFrames.
        extra_fields: Additional pipeline DataFrame columns to attach to each
            Block and Chunk under their ``.extra`` dict. Unknown columns get None.
    """
    config = ParseConfig(
        max_chunk_size=max_chunk_size,
        optimal_chunk_size=optimal_chunk_size,
        extract_tables=extract_tables,
        chunking=chunking,
        regions=regions,
        debug=debug,
        extra_fields=extra_fields or [],
    )
    # URL passed as source — route through the fetch pipeline
    if isinstance(source, str) and _is_url(source):
        return _run_pipeline(None, "html", source_url=source, config=config)
    source_filename = _extract_filename(source)
    content = _load_text(source)
    return _run_pipeline(content, "html", source_url=source_url, config=config, source_filename=source_filename)


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

    # Raw string that doesn't point to a file → treat as HTML
    return parse_html(str(source), source_url=source_url, **kwargs)


_SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".pptx", ".html", ".htm", ".xhtml"}


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


class DocumentParser:
    """Reusable parser that holds a fixed ParseConfig across multiple documents."""

    def __init__(self, config: ParseConfig | None = None) -> None:
        self.config = config or ParseConfig()

    def parse(self, source: _Source, source_url: str | None = None) -> ParseResult:
        """Parse a single document, reusing this instance's config."""
        config = self.config

        if isinstance(source, bytes):
            if source[:4] == b"%PDF":
                return _run_pipeline(source, "pdf", source_url=None, config=config)
            openxml_type = _detect_openxml_type(source)
            if openxml_type == "docx":
                return _run_pipeline(source, "docx", source_url=None, config=config)
            if openxml_type == "pptx":
                return _run_pipeline(source, "pptx", source_url=None, config=config)
            return _run_pipeline(source.decode("utf-8", errors="replace"), "html", source_url=source_url, config=config)

        if isinstance(source, io.IOBase):
            data = source.read()
            if isinstance(data, str):
                return _run_pipeline(data, "html", source_url=source_url, config=config)
            return self.parse(data, source_url=source_url)

        if isinstance(source, str) and _is_url(source):
            from .scraping.dispatcher import _is_sec_url, fetch_url
            scraped = fetch_url(source)
            raw = scraped.raw_bytes
            ct = scraped.content_type or ""
            if _looks_like_pdf(raw, ct):
                return _run_pipeline(raw, "pdf", source_url=None, config=config)
            openxml_type = _detect_openxml_type(raw)
            if openxml_type == "docx":
                return _run_pipeline(raw, "docx", source_url=None, config=config)
            if openxml_type == "pptx":
                return _run_pipeline(raw, "pptx", source_url=None, config=config)
            final_url = scraped.final_url
            if not _is_sec_url(final_url):
                return _run_pipeline(None, "html", source_url=final_url, config=config)
            html = raw.decode(scraped.encoding or "utf-8", errors="replace")
            return _run_pipeline(html, "html", source_url=final_url, config=config)

        if isinstance(source, str) and _looks_like_raw_html(source):
            return _run_pipeline(source, "html", source_url=source_url, config=config)

        try:
            path = Path(source)
        except OSError:
            return _run_pipeline(str(source), "html", source_url=source_url, config=config)

        if path.exists():
            suffix = path.suffix.lower()
            if suffix == ".pdf":
                return _run_pipeline(path.read_bytes(), "pdf", source_url=None, config=config)
            if suffix == ".docx":
                return _run_pipeline(path.read_bytes(), "docx", source_url=None, config=config)
            if suffix == ".pptx":
                return _run_pipeline(path.read_bytes(), "pptx", source_url=None, config=config)
            if suffix in (".html", ".htm", ".xhtml"):
                return _run_pipeline(path.read_text(encoding="utf-8"), "html", source_url=source_url, config=config)
            data = path.read_bytes()
            if data[:4] == b"%PDF":
                return _run_pipeline(data, "pdf", source_url=None, config=config)
            openxml_type = _detect_openxml_type(data)
            if openxml_type == "docx":
                return _run_pipeline(data, "docx", source_url=None, config=config)
            if openxml_type == "pptx":
                return _run_pipeline(data, "pptx", source_url=None, config=config)
            if suffix in (".ppt", ".doc"):
                raise ValueError(f"Unsupported legacy Office format: {suffix}. Use .pptx/.docx.")
            return _run_pipeline(path.read_text(encoding="utf-8"), "html", source_url=source_url, config=config)

        return _run_pipeline(str(source), "html", source_url=source_url, config=config)

    def parse_all(self, sources: list[_Source], /) -> Iterator[ParseResult]:
        """Parse multiple documents lazily, reusing this instance's config."""
        for source in sources:
            yield self.parse(source)
