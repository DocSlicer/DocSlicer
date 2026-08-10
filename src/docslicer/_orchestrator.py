# SPDX-FileCopyrightText: 2026 Market Framer Inc.
# SPDX-License-Identifier: AGPL-3.0-only
"""Assemble the pipeline's DataFrames into the public ParseResult (chunks, blocks, tables, charts, hierarchy)."""

from __future__ import annotations

import datetime
import logging
import time
import uuid
from typing import Callable, Literal, Optional

import pandas as pd

from .metadata.schema import DocumentMetadata
from .pdf.pdf_orchestrator import run_pipeline as _run_pdf_pipeline
from .shared.shared_orchestrator import run_pipeline as _run_shared_pipeline
from .shared.step_07_block_merger import _format_chart_markdown, _format_table_markdown
from ._config import ParseConfig, DEFAULT_CONFIG
from ._result import BBox, Block, Chart, ChartPoint, Chunk, HierarchyNode, HierarchyTree, ParseResult, Table, TableCell

_log = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Metadata resolution
# ─────────────────────────────────────────────

def _resolve_metadata(
    discovered: dict,
    source_url: str | None,
    source_filename: str | None,
    file_size_bytes: int | None,
    run_id: str,
    df_blocks: pd.DataFrame,
    processing_time: float | None = None,
    token_count: int | None = None,
    token_count_exact: bool = False,
) -> DocumentMetadata:
    # Every pipeline runs metadata/consolidate.py, which resolves title / author /
    # language on `discovered` (native-wins with a fake-author gate). This function
    # just consumes those already-processed fields.
    title = discovered.get("title") or None
    author_list = discovered.get("author") or []
    author = list(author_list) if author_list else None

    # Compute chars from the final blocks (consistent across all formats)
    if not df_blocks.empty and "text" in df_blocks.columns:
        chars = int(df_blocks["text"].str.len().fillna(0).sum())
    else:
        chars = discovered.get("chars") or 0
    chars = chars or None

    processed_at = datetime.datetime.now(datetime.timezone.utc).isoformat()

    content_type = str(discovered.get("content_type") or "unknown").lower()

    return DocumentMetadata(
        document_id=str(uuid.uuid4()),
        run_id=run_id,
        processed_at=processed_at,
        content_type=content_type,
        source_filename=source_filename,
        source_url=source_url,
        file_size_bytes=file_size_bytes,
        is_password_protected=bool(discovered.get("is_password_protected", False)),
        renderer=discovered.get("renderer"),
        page_count=int(discovered.get("page_count") or 0),
        page_width=discovered.get("page_width"),
        page_height=discovered.get("page_height"),
        page_format=discovered.get("page_format"),
        has_mixed_page_sizes=bool(discovered.get("has_mixed_page_sizes", False)),
        has_ocr=bool(discovered.get("has_ocr", False)),
        needs_ocr=bool(discovered.get("needs_ocr", False)),
        is_scanned=bool(discovered.get("is_scanned", False)),
        chars=chars,
        token_count=token_count,
        token_count_exact=token_count_exact,
        title=title,
        author=author,
        language=discovered.get("language"),
        # Native-only pass-through fields (set by each format's native_metadata.py)
        created=discovered.get("created"),
        modified=discovered.get("modified"),
        last_modified_by=discovered.get("last_modified_by"),
        generator=discovered.get("generator"),
        document_type=discovered.get("document_type"),
        parsing_quality_score=discovered.get("parsing_quality_score"),
        processing_time=processing_time,
        # Pipeline intermediates (not serialised)
        author_meta=discovered.get("author_meta") or None,
        author_text=discovered.get("author_text") or None,
        title_meta=discovered.get("title_meta"),
        title_text=discovered.get("title_text"),
        language_meta=discovered.get("language_meta"),
        language_text=discovered.get("language_text"),
        extra=discovered.get("extra") or {},
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


def _norm_id_str(v) -> str:
    """Stringify a value, normalising float-typed integers (83.0 → '83')."""
    if isinstance(v, float):
        if pd.isna(v):
            return ""
        if v == int(v):
            return str(int(v))
    s = str(v).strip()
    # Also handle string form "83.0" that may arrive after an earlier astype(str)
    if s.endswith(".0") and s[:-2].lstrip("-").isdigit():
        return s[:-2]
    return s


def _str_list(row: pd.Series, col: str) -> list[str]:
    """Safely extract a list of strings from a column that may hold a list, None, or scalar."""
    val = row.get(col) if col in row.index else None
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return []
    if isinstance(val, list):
        return [_norm_id_str(v) for v in val if v is not None and _norm_id_str(v)]
    s = _norm_id_str(val)
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

    # heading_id (str) -> list of block_ids in that heading's section
    # Computed by forward-filling heading_id across the sorted blocks df.
    block_ids_map: dict[str, list[str]] = {}
    if "block_id" in df_blocks.columns:
        needed = [c for c in ["page_number", "block_id", "heading_id"] if c in df_blocks.columns]
        blk = df_blocks[needed].copy()
        sort_cols = [c for c in ["page_number", "block_id"] if c in blk.columns]
        if sort_cols:
            blk = blk.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)
        hid_col = blk["heading_id"].astype(str).replace({"<NA>": "", "nan": "", "None": ""})
        # Normalize float-typed integer ids ("10.0" → "10") so keys match str(int(heading_id))
        hid_col = hid_col.str.replace(r"\.0+$", "", regex=True)
        blk["_active_hid"] = hid_col.where(hid_col != "", other=pd.NA).ffill().fillna("")
        for ahid, grp in blk.groupby("_active_hid", sort=False):
            if ahid:
                block_ids_map[str(ahid)] = list(grp["block_id"].astype(str))

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

        # For hybrid_heading_paragraph, use hybrid_heading_text (not the full paragraph text)
        raw_text = row.get("text", "")
        if str(row.get("block_type", "")).strip().lower() == "hybrid_heading_paragraph":
            hybrid = row.get("hybrid_heading_text") if "hybrid_heading_text" in row.index else None
            if hybrid is not None and pd.notna(hybrid) and str(hybrid).strip():
                raw_text = hybrid

        nodes[hid] = HierarchyNode(
            heading_id=hid,
            text=str(raw_text).strip() if pd.notna(raw_text) else "",
            level=level,
            heading_type=heading_type,
            page_number=page_number,
            page_label=page_label,
            chunk_ids=chunk_ids_map.get(str(hid), []),
            block_ids=block_ids_map.get(str(hid), []),
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


def _extra(row: pd.Series, fields: list[str]) -> dict:
    out = {}
    for col in fields:
        if col not in row.index:
            out[col] = None
            continue
        val = row[col]
        out[col] = None if (isinstance(val, float) and pd.isna(val)) else val
    return out


def _build_chunks(df_chunks: pd.DataFrame, extra_fields: list[str] | None = None) -> list[Chunk]:
    _extra_fields = extra_fields or []
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
            token_count=int(row.get("token_count", 0)),
            heading=heading,
            path=path,
            parent_chunk_id=parent_chunk_id,
            bbox=_bbox(row),
            link_url=_str_list(row, "link_url"),
            table_ids=_str_list(row, "table_id"),
            chart_ids=_str_list(row, "chart_id"),
            extra=_extra(row, _extra_fields),
        ))
    return out


def _build_blocks(df_blocks: pd.DataFrame, extra_fields: list[str] | None = None) -> list[Block]:
    _extra_fields = extra_fields or []
    out = []
    for _, row in df_blocks.iterrows():
        raw_label = row.get("page_label") if "page_label" in row.index else None
        page_label = str(raw_label).strip() if raw_label and not (isinstance(raw_label, float) and pd.isna(raw_label)) else None

        out.append(Block(
            id=str(row.get("block_id", "")),
            text=str(row.get("text", "")),
            page_number=int(row.get("page_number", 0)),
            page_label=page_label,
            type=str(row.get("block_type", "")),
            section=str(row.get("section", "")),
            chunk_id=None,  # block→chunk link not available without re-running chunk assignment
            char_count=int(row.get("embed_char_count", 0)),
            bbox=_bbox(row),
            link_url=_str_list(row, "link_url"),
            table_ids=_str_list(row, "table_id"),
            chart_ids=_str_list(row, "chart_id"),
            token_count=int(row.get("token_count", 0) or 0),
            extra=_extra(row, _extra_fields),
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
                role=str(crow.get("table_cell_role", "")),
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


def _build_charts(df_chart_points: pd.DataFrame | None) -> list[Chart]:
    if df_chart_points is None or df_chart_points.empty or "chart_id" not in df_chart_points.columns:
        return []

    def _opt_str(row: pd.Series, col: str) -> str | None:
        v = row.get(col) if col in row.index else None
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return None
        s = str(v).strip()
        return s or None

    def _opt_float(row: pd.Series, col: str) -> float | None:
        v = row.get(col) if col in row.index else None
        if v is None or pd.isna(v):
            return None
        return float(v)

    out = []
    for chart_id in df_chart_points["chart_id"].unique():
        points_df = df_chart_points[df_chart_points["chart_id"] == chart_id]
        first = points_df.iloc[0]

        page_raw = first.get("page_number") if "page_number" in first.index else None
        page = int(page_raw) if page_raw is not None and pd.notna(page_raw) else 0

        points = []
        for _, prow in points_df.iterrows():
            points.append(ChartPoint(
                series_index=int(prow.get("series_index", 0)),
                series_name=_opt_str(prow, "series_name"),
                point_index=int(prow.get("point_index", 0)),
                category=_opt_str(prow, "category"),
                label=_opt_str(prow, "label"),
                value=_opt_float(prow, "value"),
                x_value=_opt_float(prow, "x_value"),
                y_value=_opt_float(prow, "y_value"),
                bubble_size=_opt_float(prow, "bubble_size"),
                percent=_opt_float(prow, "percent"),
            ))

        out.append(Chart(
            id=str(int(chart_id)) if pd.notna(chart_id) else str(chart_id),
            chart_type=str(first.get("chart_type", "")),
            title=_opt_str(first, "chart_title"),
            axis_x_title=_opt_str(first, "axis_x_title"),
            axis_y_title=_opt_str(first, "axis_y_title"),
            page_number=page,
            page_label=_opt_str(first, "page_label"),
            chunk_id="",
            is_stacked=bool(first.get("is_stacked", False)),
            bbox=_bbox(first),
            markdown=_format_chart_markdown(points_df),
            points=points,
        ))
    return out


def _build_result(
    discovered_metadata: dict,
    df_blocks: pd.DataFrame,
    df_chunks: pd.DataFrame,
    df_table_cells: pd.DataFrame | None,
    df_chart_points: pd.DataFrame | None,
    source_url: str | None,
    source_filename: str | None,
    file_size_bytes: int | None,
    run_id: str,
    config: ParseConfig,
    processing_time: float | None = None,
    early_steps: dict[str, pd.DataFrame] | None = None,
) -> ParseResult:
    doc_token_count: int | None = None
    doc_token_count_exact = False
    if not df_chunks.empty and "token_count" in df_chunks.columns:
        from .shared.step_08_chunk_builder import token_encoder
        doc_token_count = int(df_chunks["token_count"].fillna(0).sum())
        # Requesting exact counts is not getting them: tiktoken may be missing, or
        # unable to fetch its vocabulary. Report what the counter actually did.
        doc_token_count_exact = bool(config.exact_tokens) and token_encoder() is not None

    metadata = _resolve_metadata(
        discovered_metadata, source_url, source_filename, file_size_bytes, run_id, df_blocks,
        processing_time=processing_time,
        token_count=doc_token_count,
        token_count_exact=doc_token_count_exact,
    )
    chunks = _build_chunks(df_chunks, config.extra_fields)
    blocks = _build_blocks(df_blocks, config.extra_fields)
    tables = _build_tables(df_table_cells)
    charts = _build_charts(df_chart_points)
    hierarchy = _build_hierarchy(df_blocks, df_chunks)

    pipeline_steps: dict[str, pd.DataFrame] = {}
    if config.debug:
        pipeline_steps.update(early_steps or {})
        pipeline_steps["blocks"] = df_blocks
        pipeline_steps["chunks"] = df_chunks
        if df_table_cells is not None and "table_cells" not in pipeline_steps:
            pipeline_steps["table_cells"] = df_table_cells

    return ParseResult(
        chunks=chunks,
        blocks=blocks,
        tables=tables,
        metadata=metadata,
        charts=charts,
        hierarchy=hierarchy,
        pipeline_steps=pipeline_steps,
    )


# ─────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────

def _run_pipeline(
    content: str | bytes | None,
    content_type: Literal["html", "pdf", "docx", "pptx"],
    source_url: str | None,
    config: ParseConfig,
    source_filename: str | None = None,
    session=None,
    on_stage: Optional[Callable[[str], None]] = None,
) -> ParseResult:
    _t0 = time.perf_counter()

    # Compute file size before format-specific processing consumes the bytes
    if isinstance(content, bytes):
        file_size_bytes: int | None = len(content)
    elif isinstance(content, str):
        file_size_bytes = len(content.encode("utf-8"))
    else:
        file_size_bytes = None

    early_steps: dict[str, pd.DataFrame] = {}
    df_chart_points: pd.DataFrame | None = None

    if content_type == "pdf":
        if not isinstance(content, bytes):
            raise TypeError("PDF content must be bytes")
        discovered_metadata, df_lines, df_table_cells, early_steps = _run_pdf_pipeline(
            pdf_bytes=content, source_url=source_url, debug=config.debug,
            password=config.password, source_filename=source_filename,
            max_workers=config.max_workers, on_stage=on_stage,
        )
        discovered_metadata["content_type"] = "pdf"
    elif content_type == "docx":
        if not isinstance(content, bytes):
            raise TypeError("DOCX content must be bytes")
        from .docx.docx_orchestrator import run_pipeline as _run_docx_pipeline
        docx_res = _run_docx_pipeline(
            content,
            include_headers_footers=config.include_headers_footers,
            include_footnotes=config.include_footnotes,
            include_comments=config.include_comments,
            password=config.password,
            source_filename=source_filename,
            on_stage=on_stage,
        )
        discovered_metadata = docx_res.discovered_metadata
        discovered_metadata["content_type"] = "docx"
        df_chart_points = docx_res.df_chart_points
        df_table_cells = docx_res.df_table_cells
        df_lines = docx_res.df_lines
        if config.debug:
            early_steps["runs"] = docx_res.df_runs
            early_steps["chart_points"] = df_chart_points
            early_steps["paragraphs"] = docx_res.df_paragraphs
            early_steps["lines"] = df_lines
            if df_table_cells is not None:
                early_steps["table_cells"] = df_table_cells
    elif content_type == "pptx":
        if not isinstance(content, bytes):
            raise TypeError("PPTX content must be bytes")
        from .pptx.pptx_orchestrator import run_pipeline as _run_pptx_pipeline
        pptx_res = _run_pptx_pipeline(
            content,
            include_speaker_notes=config.include_speaker_notes,
            password=config.password,
            source_filename=source_filename,
            on_stage=on_stage,
        )
        discovered_metadata = pptx_res.discovered_metadata
        discovered_metadata["content_type"] = "pptx"
        df_chart_points = pptx_res.df_chart_points
        df_table_cells = pptx_res.df_table_cells
        df_lines = pptx_res.df_lines
        if config.debug:
            early_steps["runs"] = pptx_res.df_runs
            early_steps["chart_points"] = df_chart_points
            early_steps["paragraphs"] = pptx_res.df_paragraphs
            early_steps["lines"] = df_lines
            if df_table_cells is not None:
                early_steps["table_cells"] = df_table_cells
    elif content_type == "html":
        if content is not None and not isinstance(content, str):
            raise TypeError("HTML content must be a string")
        use_browser = config.use_browser
        if use_browser:
            try:
                from playwright.sync_api import sync_playwright as _pw  # noqa: F401
            except ImportError:
                import warnings
                warnings.warn(
                    "HTML parsing requested browser mode (use_browser=True) but "
                    "playwright is not installed; falling back to the faster, "
                    "lower-fidelity static (non-Playwright) box extractor. Install "
                    "'docslicer[html]' && playwright install for full-fidelity "
                    "rendering, or pass use_browser=False to silence this warning.",
                    stacklevel=2,
                )
                use_browser = False
        from .html.html_orchestrator import run_pipeline as _run_html_pipeline
        discovered_metadata, df_lines, df_table_cells, html_steps = _run_html_pipeline(
            html=content, source_url=source_url, debug=config.debug, session=session,
            use_browser=use_browser, on_stage=on_stage,
        )
        discovered_metadata["content_type"] = "html"
        if config.debug:
            early_steps.update(html_steps)
    else:
        raise ValueError(f"Unsupported content type: {content_type!r}. Use 'pdf', 'html', 'docx', or 'pptx'.")

    if df_lines.empty:
        _log.warning("Pipeline returned empty df_lines, returning empty result")
        return ParseResult(
            chunks=[],
            blocks=[],
            tables=[],
            metadata=DocumentMetadata(
                run_id=config.run_id,
                content_type=content_type,
                source_filename=source_filename,
                source_url=source_url,
                file_size_bytes=file_size_bytes,
            ),
        )

    df_blocks, df_chunks = _run_shared_pipeline(
        lines_df=df_lines,
        df_table_cells=df_table_cells,
        chart_points_df=df_chart_points,
        config=config,
        on_stage=on_stage,
    )

    return _build_result(
        discovered_metadata,
        df_blocks,
        df_chunks,
        df_table_cells,
        df_chart_points,
        source_url,
        source_filename,
        file_size_bytes,
        config.run_id,
        config,
        processing_time=time.perf_counter() - _t0,
        early_steps=early_steps,
    )
