from __future__ import annotations

import json
import logging
from typing import Literal

import pandas as pd

from .pdf.pdf_orchestrator import run_pipeline as _run_pdf_pipeline
from .shared.shared_orchestrator import run_pipeline as _run_shared_pipeline
from .shared.step_06_block_merger import _format_table_markdown
from ._config import ParseConfig, DEFAULT_CONFIG
from ._result import Block, Chunk, DocMetadata, ParseResult, Table

_log = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Metadata resolution
# ─────────────────────────────────────────────

def _resolve_metadata(discovered: dict, source_url: str | None) -> DocMetadata:
    """Resolve author/title from raw discovered_metadata dict and return DocMetadata."""
    # Author: prefer whichever of author_meta / author_text is longer
    author_meta = discovered.get("author_meta") or []
    author_text = discovered.get("author_text") or []
    author_meta_str = json.dumps(author_meta) if isinstance(author_meta, list) else str(author_meta or "")
    author_text_str = json.dumps(author_text) if isinstance(author_text, list) else str(author_text or "")
    author = author_text_str if len(author_text_str) > len(author_meta_str) else author_meta_str
    author = author or None

    # Title: prefer whichever of title_meta / title_text is longer
    title_meta = str(discovered.get("title_meta") or "")
    title_text = str(discovered.get("title_text") or "")
    title = title_text if len(title_text) > len(title_meta) else title_meta
    title = title or None

    return DocMetadata(
        title=title,
        author=author,
        page_count=int(discovered.get("page_count") or 0),
        language=discovered.get("language"),
        has_ocr=bool(discovered.get("has_ocr", False)),
        source_url=source_url,
    )


# ─────────────────────────────────────────────
# DataFrame → result objects
# ─────────────────────────────────────────────

def _bbox(row: pd.Series) -> tuple[float, float, float, float] | None:
    for col in ("x_left", "y_top", "x_right", "y_bottom"):
        if col not in row.index or pd.isna(row[col]):
            return None
    return (float(row["x_left"]), float(row["y_top"]), float(row["x_right"]), float(row["y_bottom"]))


def _build_chunks(df_chunks: pd.DataFrame) -> list[Chunk]:
    out = []
    for _, row in df_chunks.iterrows():
        raw_path = row.get("chunk_path", "") if "chunk_path" in row.index else ""
        if isinstance(raw_path, list):
            hierarchy = raw_path
        elif raw_path:
            # chunk_path is stored as newline-joined heading strings
            hierarchy = [s.strip() for s in str(raw_path).split("\n") if s.strip()]
        else:
            hierarchy = []
        out.append(Chunk(
            id=str(row.get("chunk_id", "")),
            text=str(row.get("text", "")),
            page=int(row.get("page_number", 0)),
            hierarchy=hierarchy,
            region=str(row.get("document_region", "")),
            chunk_index=int(row.get("chunk_index", 0)),
            char_count=int(row.get("embed_char_count", 0)),
            bbox=_bbox(row),
        ))
    return out


def _build_blocks(df_blocks: pd.DataFrame) -> list[Block]:
    out = []
    for _, row in df_blocks.iterrows():
        out.append(Block(
            id=str(row.get("block_id", "")),
            text=str(row.get("text", "")),
            page=int(row.get("page_number", 0)),
            role=str(row.get("block_role", "")),
            region=str(row.get("document_region", "")),
            chunk_id=None,  # block→chunk link not available without re-running chunk assignment
            char_count=int(row.get("embed_char_count", 0)),
            bbox=_bbox(row),
        ))
    return out


def _build_tables(df_table_cells: pd.DataFrame | None) -> list[Table]:
    if df_table_cells is None or df_table_cells.empty:
        return []
    out = []
    for table_id in df_table_cells["table_id"].unique():
        cells = df_table_cells[df_table_cells["table_id"] == table_id]
        page = int(cells["page_number"].iloc[0]) if "page_number" in cells.columns else 0
        caption = None
        if "caption" in cells.columns:
            cap_vals = cells["caption"].dropna()
            caption = str(cap_vals.iloc[0]) if not cap_vals.empty else None

        markdown = _format_table_markdown(cells)

        bbox: tuple[float, float, float, float] | None = None
        if all(c in cells.columns for c in ("x_left", "y_top", "x_right", "y_bottom")):
            vals = (cells["x_left"].min(), cells["y_top"].min(),
                    cells["x_right"].max(), cells["y_bottom"].max())
            if not any(pd.isna(v) for v in vals):
                bbox = tuple(float(v) for v in vals)  # type: ignore[assignment]

        out.append(Table(
            id=str(table_id),
            caption=caption,
            page=page,
            markdown=markdown,
            chunk_id="",  # linked by _orchestrator after chunk building if needed
            bbox=bbox,
        ))
    return out


def _build_result(
    discovered_metadata: dict,
    df_blocks: pd.DataFrame,
    df_chunks: pd.DataFrame,
    df_table_cells: pd.DataFrame | None,
    source_url: str | None,
    config: ParseConfig,
) -> ParseResult:
    metadata = _resolve_metadata(discovered_metadata, source_url)
    chunks = _build_chunks(df_chunks)
    blocks = _build_blocks(df_blocks)
    tables = _build_tables(df_table_cells)

    # Apply region filter if requested
    if config.regions:
        allowed = set(config.regions)
        chunks = [c for c in chunks if c.region in allowed]
        blocks = [b for b in blocks if b.region in allowed]

    pipeline_steps: dict[str, pd.DataFrame] = {}
    if config.debug:
        pipeline_steps["blocks"] = df_blocks
        pipeline_steps["chunks"] = df_chunks
        if df_table_cells is not None:
            pipeline_steps["table_cells"] = df_table_cells

    return ParseResult(
        chunks=chunks,
        blocks=blocks,
        tables=tables,
        metadata=metadata,
        pipeline_steps=pipeline_steps,
    )


# ─────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────

def _run_pipeline(
    content: str | bytes | None,
    content_type: Literal["html", "pdf"],
    source_url: str | None,
    config: ParseConfig,
) -> ParseResult:
    if content_type == "pdf":
        if not isinstance(content, bytes):
            raise TypeError("PDF content must be bytes")
        discovered_metadata, df_lines, df_table_cells = _run_pdf_pipeline(
            pdf_bytes=content, source_url=source_url
        )
    elif content_type == "html":
        if content is not None and not isinstance(content, str):
            raise TypeError("HTML content must be a string")
        try:
            from playwright.sync_api import sync_playwright as _pw  # noqa: F401
        except ImportError:
            raise ImportError(
                "HTML parsing requires playwright. Install it with: "
                "pip install 'docslicer[html]' && playwright install"
            )
        from .html.html_orchestrator import run_pipeline as _run_html_pipeline
        discovered_metadata, df_lines, df_table_cells = _run_html_pipeline(
            html=content, source_url=source_url
        )
    else:
        raise ValueError(f"Unsupported content type: {content_type!r}. Use 'pdf' or 'html'.")

    if df_lines.empty:
        _log.warning("Pipeline returned empty df_lines, returning empty result")
        return ParseResult(
            chunks=[],
            blocks=[],
            tables=[],
            metadata=DocMetadata(
                title=None, author=None, page_count=0,
                language=None, has_ocr=False, source_url=source_url,
            ),
        )

    df_blocks, df_chunks = _run_shared_pipeline(
        lines_df=df_lines,
        df_table_cells=df_table_cells,
        config=config,
    )

    return _build_result(discovered_metadata, df_blocks, df_chunks, df_table_cells, source_url, config)
