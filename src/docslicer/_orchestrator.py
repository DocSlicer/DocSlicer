from __future__ import annotations

import json
import logging
from typing import Literal

import pandas as pd

from .pdf.pdf_orchestrator import run_pipeline as _run_pdf_pipeline
from .shared.shared_orchestrator import run_pipeline as _run_shared_pipeline
from .shared.step_07_block_merger import _format_table_markdown
from ._config import ParseConfig, DEFAULT_CONFIG
from ._result import BBox, Block, Chunk, DocMetadata, HierarchyNode, HierarchyTree, ParseResult, Table, TableCell

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

def _bbox(row: pd.Series) -> BBox | None:
    for col in ("x_left", "y_top", "x_right", "y_bottom"):
        if col not in row.index or pd.isna(row[col]):
            return None
    return BBox(
        x_left=float(row["x_left"]),
        y_top=float(row["y_top"]),
        x_right=float(row["x_right"]),
        y_bottom=float(row["y_bottom"]),
    )


def _str_list(row: pd.Series, col: str) -> list[str]:
    """Safely extract a list of strings from a column that may hold a list, None, or scalar."""
    val = row.get(col) if col in row.index else None
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return []
    if isinstance(val, list):
        return [str(v) for v in val if v is not None and str(v).strip()]
    s = str(val).strip()
    return [s] if s else []


_HEADING_BLOCK_TYPES = {"heading", "toc_heading", "exhibit_heading", "hybrid_heading_paragraph"}


def _build_hierarchy(df_blocks: pd.DataFrame, df_chunks: pd.DataFrame) -> HierarchyTree:
    if "heading_id" not in df_blocks.columns or "block_type" not in df_blocks.columns:
        return HierarchyTree(roots=[])

    heading_mask = (
        df_blocks["block_type"].astype("string").str.strip().str.lower()
        .isin(_HEADING_BLOCK_TYPES).fillna(False)
    ) & df_blocks["heading_id"].notna()

    if not heading_mask.any():
        return HierarchyTree(roots=[])

    heading_df = df_blocks[heading_mask].drop_duplicates(subset=["heading_id"], keep="first")

    # heading_id (str) -> list of chunk_ids under that heading
    chunk_ids_map: dict[str, list[str]] = {}
    if "active_heading_id" in df_chunks.columns and "chunk_id" in df_chunks.columns:
        for active_hid, grp in df_chunks.groupby("active_heading_id", sort=False):
            key = str(active_hid).strip()
            if key:
                chunk_ids_map[key] = list(grp["chunk_id"].astype(str))

    nodes: dict[int, HierarchyNode] = {}
    parent_map: dict[int, int | None] = {}

    for _, row in heading_df.iterrows():
        hid = int(row["heading_id"])

        raw_parent = row.get("parent_heading_id") if "parent_heading_id" in row.index else None
        parent_id = None if raw_parent is None or pd.isna(raw_parent) else int(raw_parent)

        raw_label = row.get("page_label") if "page_label" in row.index else None
        page_label = str(raw_label).strip() if raw_label and not (isinstance(raw_label, float) and pd.isna(raw_label)) else None

        raw_page = row.get("page_number") if "page_number" in row.index else None
        page_number = int(raw_page) if raw_page is not None and not (isinstance(raw_page, float) and pd.isna(raw_page)) else None

        raw_level = row.get("heading_level") if "heading_level" in row.index else None
        level = int(raw_level) if raw_level is not None and not (isinstance(raw_level, float) and pd.isna(raw_level)) else 1

        raw_ht = row.get("heading_type") if "heading_type" in row.index else None
        heading_type = str(raw_ht).strip() if raw_ht is not None and not (isinstance(raw_ht, float) and pd.isna(raw_ht)) else "free_form"

        nodes[hid] = HierarchyNode(
            heading_id=hid,
            text=str(row.get("text", "")).strip(),
            level=level,
            heading_type=heading_type,
            page_number=page_number,
            page_label=page_label,
            chunk_ids=chunk_ids_map.get(str(hid), []),
            children=[],
        )
        parent_map[hid] = parent_id

    roots: list[HierarchyNode] = []
    for hid, node in nodes.items():
        pid = parent_map[hid]
        if pid is None or pid not in nodes:
            roots.append(node)
        else:
            nodes[pid].children.append(node)

    return HierarchyTree(roots=roots)


def _build_chunks(df_chunks: pd.DataFrame) -> list[Chunk]:
    out = []
    for _, row in df_chunks.iterrows():
        raw_path = row.get("chunk_path", "") if "chunk_path" in row.index else ""
        if isinstance(raw_path, list):
            path = raw_path
        elif raw_path:
            path = [s.strip() for s in str(raw_path).split("\n") if s.strip()]
        else:
            path = []

        raw_heading = row.get("chunk_heading") if "chunk_heading" in row.index else None
        heading = str(raw_heading).strip() if raw_heading and not (isinstance(raw_heading, float) and pd.isna(raw_heading)) else None

        raw_label = row.get("page_label") if "page_label" in row.index else None
        page_label = str(raw_label).strip() if raw_label and not (isinstance(raw_label, float) and pd.isna(raw_label)) else None

        raw_parent = row.get("parent_chunk_id") if "parent_chunk_id" in row.index else None
        parent_chunk_id = str(raw_parent) if raw_parent and not (isinstance(raw_parent, float) and pd.isna(raw_parent)) else None

        out.append(Chunk(
            id=str(row.get("chunk_id", "")),
            text=str(row.get("text", "")),
            page_number=int(row.get("page_number", 0)),
            page_label=page_label,
            section=str(row.get("section", "")),
            chunk_index=int(row.get("chunk_index", 0)),
            char_count=int(row.get("embed_char_count", 0)),
            heading=heading,
            path=path,
            parent_chunk_id=parent_chunk_id,
            bbox=_bbox(row),
            link_url=_str_list(row, "link_url"),
            ixbrl_ids=_str_list(row, "ixbrl_id"),
            table_ids=_str_list(row, "table_ids"),
        ))
    return out


def _build_blocks(df_blocks: pd.DataFrame) -> list[Block]:
    out = []
    for _, row in df_blocks.iterrows():
        raw_label = row.get("page_label") if "page_label" in row.index else None
        page_label = str(raw_label).strip() if raw_label and not (isinstance(raw_label, float) and pd.isna(raw_label)) else None

        out.append(Block(
            id=str(row.get("block_id", "")),
            text=str(row.get("text", "")),
            page_number=int(row.get("page_number", 0)),
            page_label=page_label,
            role=str(row.get("block_type", "")),
            section=str(row.get("section", "")),
            chunk_id=None,  # block→chunk link not available without re-running chunk assignment
            char_count=int(row.get("embed_char_count", 0)),
            bbox=_bbox(row),
            link_url=_str_list(row, "link_url"),
            ixbrl_ids=_str_list(row, "ixbrl_id"),
            table_ids=_str_list(row, "table_ids"),
        ))
    return out


def _build_tables(df_table_cells: pd.DataFrame | None) -> list[Table]:
    if df_table_cells is None or df_table_cells.empty:
        return []
    out = []
    for table_id in df_table_cells["table_id"].unique():
        cells_df = df_table_cells[df_table_cells["table_id"] == table_id]
        page = int(cells_df["page_number"].iloc[0]) if "page_number" in cells_df.columns else 0

        raw_label = cells_df["page_label"].iloc[0] if "page_label" in cells_df.columns else None
        page_label = str(raw_label).strip() if raw_label and not (isinstance(raw_label, float) and pd.isna(raw_label)) else None

        caption = None
        if "caption" in cells_df.columns:
            cap_vals = cells_df["caption"].dropna()
            caption = str(cap_vals.iloc[0]) if not cap_vals.empty else None

        markdown = _format_table_markdown(cells_df)

        bbox: BBox | None = None
        if all(c in cells_df.columns for c in ("x_left", "y_top", "x_right", "y_bottom")):
            vals = (cells_df["x_left"].min(), cells_df["y_top"].min(),
                    cells_df["x_right"].max(), cells_df["y_bottom"].max())
            if not any(pd.isna(v) for v in vals):
                bbox = BBox(x_left=float(vals[0]), y_top=float(vals[1]),
                            x_right=float(vals[2]), y_bottom=float(vals[3]))

        cells = []
        for _, crow in cells_df.iterrows():
            cells.append(TableCell(
                row=int(crow.get("row_start", 0)),
                col=int(crow.get("col_start", 0)),
                rowspan=int(crow.get("rowspan", 1)),
                colspan=int(crow.get("colspan", 1)),
                role=str(crow.get("role", "")),
                text=str(crow.get("text", "")),
                bbox=_bbox(crow),
            ))

        out.append(Table(
            id=str(table_id),
            caption=caption,
            page_number=page,
            page_label=page_label,
            markdown=markdown,
            chunk_id="",
            bbox=bbox,
            cells=cells,
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
    hierarchy = _build_hierarchy(df_blocks, df_chunks)

    # Apply region filter if requested
    if config.regions:
        allowed = set(config.regions)
        chunks = [c for c in chunks if c.section in allowed]
        blocks = [b for b in blocks if b.section in allowed]

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
        hierarchy=hierarchy,
        pipeline_steps=pipeline_steps,
    )


# ─────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────

def _run_pipeline(
    content: str | bytes | None,
    content_type: Literal["html", "pdf", "docx"],
    source_url: str | None,
    config: ParseConfig,
) -> ParseResult:
    if content_type == "pdf":
        if not isinstance(content, bytes):
            raise TypeError("PDF content must be bytes")
        discovered_metadata, df_lines, df_table_cells = _run_pdf_pipeline(
            pdf_bytes=content, source_url=source_url
        )
    elif content_type == "docx":
        if not isinstance(content, bytes):
            raise TypeError("DOCX content must be bytes")
        from .docx.docx_orchestrator import run_pipeline as _run_docx_pipeline
        from .docx.step_00_metadata import extract_core_properties
        from .metadata import add_document_information
        package, run_df, df_table_cells, _, df_lines = _run_docx_pipeline(content)
        discovered_metadata = extract_core_properties(package)
        discovered_metadata["has_ocr"] = False
        discovered_metadata["content_type"] = "DOCX"
        if not discovered_metadata["page_count"]:
            discovered_metadata["page_count"] = (
                int(run_df["page_number"].max())
                if not run_df.empty and "page_number" in run_df.columns
                else 0
            )
        add_document_information(discovered_metadata, df_lines=df_lines)
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
