"""
DOCX table cell builder.

Builds df_table_cells: one row per logical table cell, with grid geometry
(row_start, col_start, rowspan, colspan), role, aggregated text, and caption.

Counter-matching walk
---------------------
`table_cell_id` values are assigned by the run extractor using global counters
that increment across the whole document walk. To assign matching IDs here we
replicate the same walk (body → footnotes → endnotes → comments → headers →
footers, in the same order) but skip paragraph processing — paragraphs do not
touch any of the three table counters (table_id, table_row_id, table_cell_id).
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from lxml import etree

from .step_01_package_reader import DocxPackage
from .step_02_run_extractor import (
    NS,
    W,
    _content_part_specs,
    _iter_part_roots,
)
from .._utils.df_schemas import TABLE_CELLS_COLS, conform_table_cells
from .._utils.table_utils import detect_cell_roles


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
    row_index: int
    col_start: int
    colspan: int
    vmerge_restart: bool
    vmerge_continue: bool
    is_header_row: bool


def _grid_span(tc: etree._Element) -> int:
    tc_pr = tc.find("w:tcPr", namespaces=NS)
    if tc_pr is None:
        return 1
    gs = tc_pr.find("w:gridSpan", namespaces=NS)
    if gs is None:
        return 1
    try:
        return max(1, int(gs.get(f"{W}val", "1")))
    except ValueError:
        return 1


def _vmerge_state(tc: etree._Element) -> tuple[bool, bool]:
    """Return (is_restart, is_continue) from w:vMerge."""
    tc_pr = tc.find("w:tcPr", namespaces=NS)
    if tc_pr is None:
        return False, False
    vm = tc_pr.find("w:vMerge", namespaces=NS)
    if vm is None:
        return False, False
    is_restart = vm.get(f"{W}val") == "restart"
    return is_restart, not is_restart


def _row_is_header(tr: etree._Element) -> bool:
    tr_pr = tr.find("w:trPr", namespaces=NS)
    return tr_pr is not None and tr_pr.find("w:tblHeader", namespaces=NS) is not None


def _walk_tbl(
    tbl: etree._Element,
    counters: _Counters,
    geoms: list[_CellGeom],
) -> None:
    counters.table_id += 1
    table_id = counters.table_id
    for row_idx, tr in enumerate(tbl.findall("w:tr", namespaces=NS)):
        counters.table_row_id += 1
        row_id = counters.table_row_id
        is_hdr = _row_is_header(tr)
        col = 0
        for tc in tr.findall("w:tc", namespaces=NS):
            counters.table_cell_id += 1
            cell_id = counters.table_cell_id
            colspan = _grid_span(tc)
            v_restart, v_continue = _vmerge_state(tc)
            geoms.append(
                _CellGeom(
                    table_id=table_id,
                    table_row_id=row_id,
                    table_cell_id=cell_id,
                    row_index=row_idx,
                    col_start=col,
                    colspan=colspan,
                    vmerge_restart=v_restart,
                    vmerge_continue=v_continue,
                    is_header_row=is_hdr,
                )
            )
            col += colspan
            _walk_container(tc, counters, geoms)


def _walk_container(
    container: etree._Element,
    counters: _Counters,
    geoms: list[_CellGeom],
) -> None:
    for child in container:
        if child.tag == f"{W}tbl":
            _walk_tbl(child, counters, geoms)
        elif child.tag == f"{W}sdt":
            content = child.find("w:sdtContent", namespaces=NS)
            if content is not None:
                _walk_container(content, counters, geoms)


def _collect_cell_geoms(
    package: DocxPackage,
    include_headers_footers: bool,
    include_footnotes: bool,
    include_comments: bool,
) -> list[_CellGeom]:
    counters = _Counters()
    geoms: list[_CellGeom] = []
    for part_name, part_type, _ in _content_part_specs(package):
        if part_type in {"header", "footer"} and not include_headers_footers:
            continue
        if part_type in {"footnote", "endnote"} and not include_footnotes:
            continue
        if part_type == "comment" and not include_comments:
            continue
        root = package.get_xml(part_name)
        if root is None:
            continue
        for item_root, _ in _iter_part_roots(root, part_type):
            _walk_container(item_root, counters, geoms)
    return geoms


# ---------------------------------------------------------------------------
# Rowspan computation from vMerge
# ---------------------------------------------------------------------------


def _compute_rowspans(geoms: list[_CellGeom]) -> dict[int, int]:
    """
    Map table_cell_id → rowspan.

    vMerge restart cells get rowspan = count of their continuation cells + 1.
    vMerge continuation cells get rowspan = 0 (covered by the span above).
    All other cells get rowspan = 1.
    """
    by_table: dict[int, list[_CellGeom]] = {}
    for g in geoms:
        by_table.setdefault(g.table_id, []).append(g)

    rowspans: dict[int, int] = {}
    for cells in by_table.values():
        grid: dict[tuple[int, int], _CellGeom] = {
            (c.row_index, c.col_start): c for c in cells
        }
        for cell in cells:
            if cell.vmerge_continue:
                rowspans[cell.table_cell_id] = 0
            elif cell.vmerge_restart:
                span = 1
                r = cell.row_index + 1
                while True:
                    cont = grid.get((r, cell.col_start))
                    if cont is None or not cont.vmerge_continue:
                        break
                    span += 1
                    r += 1
                rowspans[cell.table_cell_id] = span
            else:
                rowspans[cell.table_cell_id] = 1
    return rowspans


# ---------------------------------------------------------------------------
# Caption detection from run_df
# ---------------------------------------------------------------------------


def _find_captions(run_df: pd.DataFrame) -> dict[int, str]:
    """
    Return {table_id: caption_text} for tables that have an immediately
    adjacent paragraph with a "caption"-named style (case-insensitive).

    "Immediately adjacent" means no other non-table, non-empty paragraph
    appears between the caption paragraph and the table.
    """
    needed = {
        "header_footer_type",
        "source_part",
        "table_id",
        "paragraph_id",
        "order_index",
        "paragraph_style_name",
        "text",
    }
    if not needed.issubset(run_df.columns) or run_df.empty:
        return {}

    body = run_df[
        run_df["header_footer_type"].eq("body")
        & run_df["source_part"].eq("word/document.xml")
    ]
    if body.empty:
        return {}

    non_table = body[body["table_id"].isna()]
    table_body = body[body["table_id"].notna()]
    if non_table.empty or table_body.empty:
        return {}

    # Summary per non-table paragraph (one row per paragraph_id)
    para_info = (
        non_table.groupby("paragraph_id")
        .agg(
            order_index=("order_index", "min"),
            order_index_max=("order_index", "max"),
            style_name=("paragraph_style_name", "first"),
        )
        .reset_index()
        .sort_values("order_index")
    )
    # Caption text from text-type runs only (excludes field markers, field code, tabs…)
    text_only = non_table[non_table["run_type"].eq("text")]
    para_text = (
        text_only.groupby("paragraph_id")["text"]
        .apply(lambda xs: "".join(str(x) for x in xs if pd.notna(x)))
        .rename("text")
    )
    para_info = para_info.join(para_text, on="paragraph_id")
    is_caption = para_info["style_name"].str.lower().str.contains("caption", na=False)
    caption_paras = para_info[is_caption]
    if caption_paras.empty:
        return {}

    table_bounds = table_body.groupby("table_id")["order_index"].agg(
        t_min="min", t_max="max"
    )

    captions: dict[int, str] = {}
    nt_orders = non_table["order_index"]

    for table_id, bounds in table_bounds.iterrows():
        t_min, t_max = int(bounds["t_min"]), int(bounds["t_max"])

        before = caption_paras[caption_paras["order_index"] < t_min]
        if not before.empty:
            nearest = before.iloc[-1]
            # Use max order_index of the caption paragraph so runs within
            # the same caption paragraph don't appear in the gap window.
            gap_start = int(nearest["order_index_max"]) + 1
            if gap_start <= t_min - 1:
                has_gap = nt_orders.between(gap_start, t_min - 1).any()
            else:
                has_gap = False
            if not has_gap:
                captions[int(table_id)] = nearest["text"]
                continue

        after = caption_paras[caption_paras["order_index"] > t_max]
        if not after.empty:
            nearest = after.iloc[0]
            gap_end = int(nearest["order_index"]) - 1
            if t_max + 1 <= gap_end:
                has_gap = nt_orders.between(t_max + 1, gap_end).any()
            else:
                has_gap = False
            if not has_gap:
                captions[int(table_id)] = nearest["text"]

    return captions


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_table_cells(
    package: DocxPackage,
    run_df: pd.DataFrame,
    include_headers_footers: bool = True,
    include_footnotes: bool = True,
    include_comments: bool = False,
    debug: bool = False,
) -> pd.DataFrame:
    """
    Build df_table_cells from a DOCX package and its run-level DataFrame.

    Args:
        package: Parsed DOCX package (from step 01).
        run_df: Run-level DataFrame (from step 02). Must use the same
            include_headers_footers / include_footnotes / include_comments settings.
        include_headers_footers: Match the setting used in extract_runs.
        include_footnotes: Match the setting used in extract_runs.
        include_comments: Match the setting used in extract_runs.
        debug: Keep the detect_cell_roles diagnostic columns (table_row_style,
            hdr_*) in the output.

    Returns:
        DataFrame with the canonical df_table_cells schema (TABLE_CELLS_COLS).

        rowspan = 0 means the cell is covered by a vertically spanning cell
        above it (w:vMerge continuation). rowspan >= 1 is the actual span.
    """
    if run_df.empty:
        return pd.DataFrame(columns=TABLE_CELLS_COLS)

    geoms = _collect_cell_geoms(package, include_headers_footers, include_footnotes, include_comments)
    if not geoms:
        return pd.DataFrame(columns=TABLE_CELLS_COLS)

    rowspans = _compute_rowspans(geoms)

    geom_df = pd.DataFrame(
        [
            {
                "table_id": g.table_id,
                "table_row_id": g.table_row_id,
                "table_cell_id": g.table_cell_id,
                "row_start": g.row_index,
                "col_start": g.col_start,
                "colspan": g.colspan,
                "rowspan": rowspans.get(g.table_cell_id, 1),
                "_tbl_header": g.is_header_row,
            }
            for g in geoms
        ]
    )

    # Aggregate page position, visible text, and paragraph style from run_df
    table_runs = run_df[run_df["table_cell_id"].notna()].copy()
    table_runs["table_cell_id"] = table_runs["table_cell_id"].astype(int)

    # Cell text: join runs within each paragraph (contiguous character runs, so
    # no separator — explicit space runs are preserved), then join the cell's
    # paragraphs with a space so multi-paragraph cells don't run together. A
    # space (rather than a newline) keeps the cell on one physical line, which
    # markdown table rows require. tab / line_break (<w:br/>) runs become a space.
    _ref_types = {"footnote_reference", "endnote_reference"}
    cell_text_runs = table_runs[
        table_runs["run_type"].isin({"text", "tab", "line_break"} | _ref_types)
    ].copy()
    cell_text_runs["text"] = cell_text_runs["text"].fillna("").astype(str)
    cell_text_runs.loc[cell_text_runs["run_type"].isin({"tab", "line_break"}), "text"] = " "
    # Render note reference markers as [^N] (matching the paragraph builder) so a
    # footnoted cell keeps its marker, e.g. "<5[^37]".
    _ref_mask = cell_text_runs["run_type"].isin(_ref_types) & cell_text_runs["text"].str.strip().ne("")
    cell_text_runs.loc[_ref_mask, "text"] = "[^" + cell_text_runs.loc[_ref_mask, "text"].str.strip() + "]"
    cell_text_runs = cell_text_runs.sort_values(
        ["table_cell_id", "paragraph_id", "order_index"]
    )
    # Rows are now sorted so every cell (and every paragraph within it) is
    # contiguous. A single linear pass over the arrays joins runs -> paragraphs
    # -> cell text; this avoids the two-level groupby.apply, whose per-group
    # object construction dominated this step on table-heavy documents.
    _cids = cell_text_runs["table_cell_id"].to_numpy()
    _pids = cell_text_runs["paragraph_id"].to_numpy()
    _txts = cell_text_runs["text"].to_numpy()
    _n = len(_cids)
    _cell_text: dict[int, str] = {}
    _i = 0
    while _i < _n:
        cid = _cids[_i]
        para_strs: list[str] = []
        while _i < _n and _cids[_i] == cid:
            pid = _pids[_i]
            parts: list[str] = []
            while _i < _n and _cids[_i] == cid and _pids[_i] == pid:
                parts.append(_txts[_i])
                _i += 1
            s = "".join(parts).strip()
            if s:
                para_strs.append(s)
        _cell_text[int(cid)] = " ".join(para_strs)
    text_agg = pd.Series(_cell_text, name="text")
    text_agg.index.name = "table_cell_id"
    page_agg = table_runs.groupby("table_cell_id").agg(
        page_number=("page_number", "first"),
        page_label=("page_label", "first"),
        _para_style=("paragraph_style_name", "first"),
    )
    agg = page_agg.join(text_agg, how="left").reset_index()

    captions = _find_captions(run_df)

    result = geom_df.merge(agg, on="table_cell_id", how="left")
    result["caption"] = result["table_id"].map(captions)
    result["text"] = result["text"].fillna("")

    # Empty cells have no runs in run_df so page_number/page_label are NaN.
    # Fill from neighbouring cells within the same table.
    for col in ("page_number", "page_label"):
        result[col] = result.groupby("table_id")[col].transform(
            lambda s: s.ffill().bfill()
        )

    # Fold DOCX-specific header signals into table_header_flag for
    # detect_cell_roles. w:tblHeader and style names containing "header" count.
    style_is_header = result["_para_style"].str.lower().str.contains("header", na=False)
    result["table_header_flag"] = result["_tbl_header"] | style_is_header
    result = result.drop(columns=["_tbl_header", "_para_style"])
    # detect_cell_roles processes every table in one vectorized pass, grouping
    # internally on (table_id, row_start).
    result = detect_cell_roles(result, with_row_label=False)

    return result #conform_table_cells(result, debug=debug)
