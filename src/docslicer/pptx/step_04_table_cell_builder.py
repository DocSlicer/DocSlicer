"""
PPTX table cell builder.

Builds df_table_cells: one row per logical table cell, with grid geometry
(row_start, col_start, rowspan, colspan), role, and aggregated text.

Counter-matching walk
---------------------
`table_cell_id` values are assigned by the run extractor using global counters
that increment across all slides. To assign matching IDs here we replicate the
same walk (slide bodies → notes, in the same order) but skip all non-table
content — shape, chart, and image processing do not touch the three table
counters (table_id, table_row_id, table_cell_id).
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from lxml import etree

from .step_01_package_reader import PptxPackage
from .step_02_run_extractor import (
    A,
    P,
    _TABLE_GRAPHIC_URI,
    _notes_part_for,
)
from .._utils.df_schemas import TABLE_CELLS_COLS, conform_table_cells
from .._utils.table.table_header import detect_cell_roles


# ---------------------------------------------------------------------------
# Geometry extraction — counter-matching XML walk
# ---------------------------------------------------------------------------


@dataclass
class _Counters:
    table_id: int = 0
    table_row_id: int = 0
    table_cell_id: int = 0


@dataclass
class _CellGeom:
    table_id: int
    table_row_id: int
    table_cell_id: int
    slide_number: int
    slide_index: int
    row_index: int
    col_start: int
    colspan: int   # 0 = covered by hMerge
    rowspan: int   # 0 = covered by vMerge
    first_row_style: bool


def _tc_colspan(tc: etree._Element) -> int:
    if tc.get("hMerge") == "1":
        return 0
    val = tc.get("gridSpan")
    try:
        return max(1, int(val)) if val is not None else 1
    except ValueError:
        return 1


def _tc_rowspan(tc: etree._Element) -> int:
    if tc.get("vMerge") == "1":
        return 0
    val = tc.get("rowSpan")
    try:
        return max(1, int(val)) if val is not None else 1
    except ValueError:
        return 1


def _tbl_first_row_style(tbl: etree._Element) -> bool:
    tbl_pr = tbl.find(f"{A}tblPr")
    if tbl_pr is None:
        return False
    return tbl_pr.get("firstRow") in ("1", "true", "True")


def _walk_tbl_geom(
    graphic_frame: etree._Element,
    slide_number: int,
    slide_index: int,
    counters: _Counters,
    geoms: list[_CellGeom],
) -> None:
    tbl = graphic_frame.find(f".//{A}tbl")
    if tbl is None:
        return

    counters.table_id += 1
    table_id = counters.table_id
    first_row_style = _tbl_first_row_style(tbl)

    for row_idx, tr in enumerate(tbl.findall(f"{A}tr")):
        counters.table_row_id += 1
        row_id = counters.table_row_id
        col = 0
        for tc in tr.findall(f"{A}tc"):
            counters.table_cell_id += 1
            cell_id = counters.table_cell_id
            colspan = _tc_colspan(tc)
            rowspan = _tc_rowspan(tc)
            geoms.append(
                _CellGeom(
                    table_id=table_id,
                    table_row_id=row_id,
                    table_cell_id=cell_id,
                    slide_number=slide_number,
                    slide_index=slide_index,
                    row_index=row_idx,
                    col_start=col,
                    colspan=colspan,
                    rowspan=rowspan,
                    first_row_style=first_row_style,
                )
            )
            # hMerge cells (colspan=0) still occupy one physical column slot.
            col += max(1, colspan)


def _walk_graphic_frame_geom(
    frame: etree._Element,
    slide_number: int,
    slide_index: int,
    counters: _Counters,
    geoms: list[_CellGeom],
) -> None:
    graphic = frame.find(f"{A}graphic")
    graphic_data = graphic.find(f"{A}graphicData") if graphic is not None else None
    uri = graphic_data.get("uri", "") if graphic_data is not None else ""
    if uri == _TABLE_GRAPHIC_URI:
        _walk_tbl_geom(frame, slide_number, slide_index, counters, geoms)


def _walk_sp_tree_geom(
    sp_tree: etree._Element,
    slide_number: int,
    slide_index: int,
    counters: _Counters,
    geoms: list[_CellGeom],
) -> None:
    for child in sp_tree:
        if child.tag == f"{P}graphicFrame":
            _walk_graphic_frame_geom(child, slide_number, slide_index, counters, geoms)
        elif child.tag == f"{P}grpSp":
            inner_tree = child.find(f"{P}spTree")
            inner = inner_tree if inner_tree is not None else child
            _walk_sp_tree_geom(inner, slide_number, slide_index, counters, geoms)


def _collect_cell_geoms(
    package: PptxPackage,
    include_speaker_notes: bool,
) -> list[_CellGeom]:
    counters = _Counters()
    geoms: list[_CellGeom] = []

    for slide in package.slides:
        root = package.get_xml(slide.part_name)
        if root is not None:
            sp_tree = root.find(f".//{P}spTree")
            if sp_tree is not None:
                _walk_sp_tree_geom(
                    sp_tree, slide.slide_number, slide.slide_index, counters, geoms
                )

        if include_speaker_notes:
            notes_part = _notes_part_for(slide, package)
            if notes_part:
                notes_root = package.get_xml(notes_part)
                if notes_root is not None:
                    notes_sp_tree = notes_root.find(f".//{P}spTree")
                    if notes_sp_tree is not None:
                        _walk_sp_tree_geom(
                            notes_sp_tree, slide.slide_number, slide.slide_index, counters, geoms
                        )

    return geoms


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_table_cells(
    package: PptxPackage,
    run_df: pd.DataFrame,
    include_speaker_notes: bool = True,
    debug: bool = False,
) -> pd.DataFrame:
    """
    Build df_table_cells from a PPTX package and its run-level DataFrame.

    Args:
        package: Parsed PPTX package (from step 01).
        run_df: Run-level DataFrame (from step 02). Must use the same
            include_speaker_notes setting.
        include_speaker_notes: Match the setting used in extract_runs.
        debug: Keep the detect_cell_roles diagnostic columns (table_row_style,
            hdr_*) in the output.

    Returns:
        DataFrame with the canonical df_table_cells schema (TABLE_CELLS_COLS).
        page_number carries the slide number.

        rowspan = 0 means the cell is covered by a vertically spanning cell
        above it (vMerge). colspan = 0 means covered by a horizontal span
        (hMerge). Values >= 1 are the actual span extents.
    """
    if run_df.empty:
        return pd.DataFrame(columns=TABLE_CELLS_COLS)

    geoms = _collect_cell_geoms(package, include_speaker_notes)
    if not geoms:
        return pd.DataFrame(columns=TABLE_CELLS_COLS)

    geom_df = pd.DataFrame(
        [
            {
                "table_id": g.table_id,
                "table_row_id": g.table_row_id,
                "table_cell_id": g.table_cell_id,
                "page_number": g.slide_number,
                "slide_index": g.slide_index,
                "row_start": g.row_index,
                "col_start": g.col_start,
                "colspan": g.colspan,
                "rowspan": g.rowspan,
                "_first_row_style": g.first_row_style,
            }
            for g in geoms
        ]
    )

    table_runs = run_df[run_df["table_cell_id"].notna()].copy()
    table_runs["table_cell_id"] = table_runs["table_cell_id"].astype(int)

    text_agg = (
        table_runs[table_runs["run_type"].isin({"text", "math", "tab"})]
        .groupby("table_cell_id")["text"]
        .apply(lambda xs: "".join(str(x) for x in xs if pd.notna(x)))
        .rename("text")
    )
    result = geom_df.merge(text_agg, on="table_cell_id", how="left")
    result["text"] = result["text"].fillna("")
    result = result.drop(columns=["_first_row_style"])

    # detect_cell_roles processes every table in one vectorized pass, grouping
    # internally on (table_id, row_start).
    result = detect_cell_roles(result, with_row_label=False)

    return result #conform_table_cells(result, debug=debug)


__all__ = ["build_table_cells"]
