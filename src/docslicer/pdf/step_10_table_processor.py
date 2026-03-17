from __future__ import annotations

from typing import Dict, List, Tuple, Any, Set
import pandas as pd
import numpy as np
import re

#TODO: Add header padding

# ============================================================
# Helpers
# ============================================================

def _compute_underline_last(df: pd.DataFrame) -> Dict[Tuple[int, int, int], int]:
    df_u = df.dropna(subset=["shape_id_underline"]).copy()
    if df_u.empty:
        return {}
    df_u["underline_id"] = df_u["shape_id_underline"].astype(int)
    last_df = (
        df_u.groupby(["layout_id", "page_number", "underline_id"])["temp_line_id"]
        .max()
        .reset_index()
        .rename(columns={"temp_line_id": "last_temp_line_id"})
    )
    result: Dict[Tuple[int, int, int], int] = {}
    for row in last_df.itertuples(index=False):
        key = (row.layout_id, row.page_number, row.underline_id)
        result[key] = int(row.last_temp_line_id)
    return result


def _compute_covered_cols(group_df: pd.DataFrame) -> set[int]:
    covered_cols: set[int] = set()
    for row in group_df.itertuples(index=False):
        col_start = int(row.col_start)
        col_end = int(row.col_end)
        for c in range(col_start, col_end + 1):
            covered_cols.add(c)
    return covered_cols


def _get_completion_threshold(band_total_cols: int, row_index: int) -> int:
    """
    Return how many columns must be covered for this row to count as 'complete',
    given the total number of columns in the band and the row index (1-based).

    Scheme (from your table):

        band_total_cols   min_normal   min_top   num_top_rows
        ----------------------------------------------------
        2                 2            1         1
        3                 3            2         1
        4                 4            3         2
        5                 4            3         2
        6                 4            3         3
        7                 4            3         3
        8                 4            3         3

    For >8 cols we just treat it like 8 (plateau).
    """
    if band_total_cols <= 0:
        return 0

    # Default values; we'll override below
    min_normal = band_total_cols
    min_top = band_total_cols
    num_top_rows = 0

    if band_total_cols == 1:
        min_normal = 1
        min_top = 1
        num_top_rows = 1
    elif band_total_cols == 2:
        min_normal = 2
        min_top = 1
        num_top_rows = 1
    elif band_total_cols == 3:
        min_normal = 3
        min_top = 2
        num_top_rows = 1
    elif band_total_cols == 4:
        min_normal = 4
        min_top = 3
        num_top_rows = 2
    elif band_total_cols == 5:
        min_normal = 4      # plateau at 4
        min_top = 3
        num_top_rows = 2
    elif 6 <= band_total_cols <= 8:
        min_normal = 4      # plateau at 4
        min_top = 3
        num_top_rows = 3
    else:
        # band_total_cols > 8 → same as 8
        min_normal = 4
        min_top = 3
        num_top_rows = 3

    # Relaxed requirement for the first N "top" rows (1-based indexing)
    if row_index <= num_top_rows:
        return min_top
    return min_normal


def _is_complete_row(
    covered_cols: set[int],
    band_total_cols: int,
    row_index: int,
) -> bool:
    """
    Decide if the group covers enough columns to be considered a complete row,
    using relaxed thresholds for the first few 'top' rows depending on
    band_total_cols.
    """
    if band_total_cols <= 0:
        return False

    required = _get_completion_threshold(band_total_cols, row_index)
    return len(covered_cols) >= required


def _has_last_underline_instance(
    line_underline_ids: List[int],
    layout_id: int,
    page_number: int,
    temp_line_id: int,
    underline_last_map: Dict[Tuple[int, int, int], int],
) -> bool:
    if not line_underline_ids:
        return False
    for u in line_underline_ids:
        key = (layout_id, page_number, u)
        last_line = underline_last_map.get(key)
        if last_line is not None and int(temp_line_id) == int(last_line):
            return True
    return False


def _try_attach_to_existing_row(
    df_line: pd.DataFrame,
    layout_id: int,
    page_number: int,
    records: List[Dict[str, Any]],
    cell_meta: Dict[int, Dict[str, Any]],
    cell_record_idx: Dict[int, int],
    row_to_cell_ids: Dict[Tuple[int, int, int], List[int]],
    underline_row_anchor: Dict[Tuple[int, int, int], int],
) -> Tuple[bool, Set[int]]:
    """
    Attach this line's cells to already-flushed row(s) based on underline.

    Returns:
        attached_any: whether anything was attached
        anchor_rows:  set of row_index values we attached to
    """
    underline_ids = (
        df_line["shape_id_underline"]
        .dropna()
        .astype(int)
        .unique()
        .tolist()
    )
    if not underline_ids:
        return False, set()

    attached_any = False
    anchor_rows: Set[int] = set()

    for u in underline_ids:
        key_u = (layout_id, page_number, u)
        if key_u not in underline_row_anchor:
            continue

        row_index_anchor = underline_row_anchor[key_u]
        anchor_rows.add(row_index_anchor)

        row_key = (layout_id, page_number, row_index_anchor)
        dest_cell_ids = row_to_cell_ids.get(row_key, [])
        if not dest_cell_ids:
            continue

        df_line_u = df_line[
            df_line["shape_id_underline"].astype("Int64") == u
        ]

        for _, cell in df_line_u.iterrows():
            col_start = int(cell["col_start"])

            chosen_id = None
            for cid in dest_cell_ids:
                meta = cell_meta[cid]
                col_start = int(meta["col_start"])
                colspan = int(meta["colspan"])
                if col_start <= col_start <= col_start + colspan - 1:
                    chosen_id = cid
                    break

            if chosen_id is None:
                chosen_id = max(
                    dest_cell_ids,
                    key=lambda cid_: int(cell_meta[cid_]["col_start"]),
                )

            rec_idx = cell_record_idx[chosen_id]
            rec = records[rec_idx]

            text_add = str(cell["text"] or "").strip()
            if text_add:
                if rec["text"]:
                    rec["text"] = rec["text"] + " " + text_add
                else:
                    rec["text"] = text_add

            rec["cell_ids"].append(cell["cell_id"])
            rec["temp_line_ids"].append(cell["temp_line_id"])

            attached_any = True

    return attached_any, anchor_rows


def _attach_pending_group_to_rows(
    group_df: pd.DataFrame,
    layout_id: int,
    page_number: int,
    anchor_rows: Set[int],
    records: List[Dict[str, Any]],
    cell_meta: Dict[int, Dict[str, Any]],
    cell_record_idx: Dict[int, int],
    row_to_cell_ids: Dict[Tuple[int, int, int], List[int]],
) -> None:
    """
    NEW: when a line with an existing underline attaches to an anchored row,
    any previously pending lines (too few columns, no underline) are also
    attached to that anchored row.

    For each cell in group_df, we:
      - find a cell on the anchor row whose span covers col_start
      - else append to the rightmost cell on that row
    """
    if group_df.empty or not anchor_rows:
        return

    for row_index_anchor in sorted(anchor_rows):
        row_key = (layout_id, page_number, row_index_anchor)
        dest_cell_ids = row_to_cell_ids.get(row_key, [])
        if not dest_cell_ids:
            continue

        group_df_sorted = group_df.sort_values(
            ["temp_line_id", "col_start", "cell_id"]
        )

        for _, cell in group_df_sorted.iterrows():
            col_start = int(cell["col_start"])

            chosen_id = None
            for cid in dest_cell_ids:
                meta = cell_meta[cid]
                col_start = int(meta["col_start"])
                colspan = int(meta["colspan"])
                if col_start <= col_start <= col_start + colspan - 1:
                    chosen_id = cid
                    break

            if chosen_id is None:
                chosen_id = max(
                    dest_cell_ids,
                    key=lambda cid_: int(cell_meta[cid_]["col_start"]),
                )

            rec_idx = cell_record_idx[chosen_id]
            rec = records[rec_idx]

            text_add = str(cell["text"] or "").strip()
            if text_add:
                if rec["text"]:
                    rec["text"] = rec["text"] + " " + text_add
                else:
                    rec["text"] = text_add

            rec["cell_ids"].append(cell["cell_id"])
            rec["temp_line_ids"].append(cell["temp_line_id"])


def _flush_group_to_row(
    group_df: pd.DataFrame,
    layout_id: int,
    page_number: int,
    table_id: int,
    row_index: int,
    band_total_cols: int,
    covered_cols: set[int],
    records: List[Dict[str, Any]],
    table_cell_id_counter: int,
    cell_meta: Dict[int, Dict[str, Any]],
    cell_record_idx: Dict[int, int],
    row_to_cell_ids: Dict[Tuple[int, int, int], List[int]],
    open_rowspan: Dict[Tuple[int, int, int], int],
    underline_row_anchor: Dict[Tuple[int, int, int], int],
    flush_reason: str,  # "complete_row" | "last_underline" (for now)
) -> Tuple[int, Dict[Tuple[int, int, int], int], Dict[Tuple[int, int, int], int]]:
    # 1) extend rowspans for missing columns
    for col in range(1, band_total_cols + 1):
        key_col = (layout_id, page_number, col)
        if col not in covered_cols:
            prev_tcell_id = open_rowspan.get(key_col)
            if prev_tcell_id is not None:
                rec_idx = cell_record_idx[prev_tcell_id]
                records[rec_idx]["rowspan"] += 1

    # 2) build new cells for columns that appear in this row
    group_df_sorted = group_df.sort_values(
        ["col_start", "col_end", "temp_line_id", "cell_id"]
    )

    row_key = (layout_id, page_number, row_index)
    row_to_cell_ids.setdefault(row_key, [])

    for (col_start, col_end), sub in group_df_sorted.groupby(
        ["col_start", "col_end"], sort=True
    ):
        col_start = int(col_start)
        col_end = int(col_end)
        colspan = col_end - col_start + 1

        texts = [str(t or "").strip() for t in sub["text"].tolist()]
        texts = [t for t in texts if t]
        merged_text = " ".join(texts).strip()

        text_raw_lines: List[str] = []
        for _, sub_line in sub.groupby("temp_line_id", sort=True):
            line_texts = [str(t or "").strip() for t in sub_line["text"].tolist()]
            line_texts = [t for t in line_texts if t]
            if line_texts:
                text_raw_lines.append(" ".join(line_texts).strip())

        temp_line_ids = (
            sub["temp_line_id"].dropna().drop_duplicates().astype(int).tolist()
        )
        cell_ids = sub["cell_id"].tolist()

        table_cell_id = table_cell_id_counter
        table_cell_id_counter += 1

        record = {
            "table_cell_id": table_cell_id,
            "page_number": page_number,
            "layout_id": layout_id,
            "table_id": table_id,
            "row_start": row_index,
            "col_start": col_start,
            "rowspan": 1,
            "colspan": colspan,
            "text": merged_text,
            "text_raw_lines": text_raw_lines,
            "cell_ids": cell_ids,
            "temp_line_ids": temp_line_ids,
            "flush_reason": flush_reason,
        }
        records.append(record)

        rec_idx = len(records) - 1
        cell_record_idx[table_cell_id] = rec_idx

        cell_meta[table_cell_id] = {
            "layout_id": layout_id,
            "page_number": page_number,
            "row_index": row_index,
            "col_start": col_start,
            "colspan": colspan,
        }

        row_to_cell_ids[row_key].append(table_cell_id)

        for col in range(col_start, col_end + 1):
            key_col = (layout_id, page_number, col)
            open_rowspan[key_col] = table_cell_id

    # 3) anchor any underline ids on this row
    group_underline_ids = (
        group_df["shape_id_underline"]
        .dropna()
        .astype(int)
        .unique()
        .tolist()
    )
    for u in group_underline_ids:
        key_u = (layout_id, page_number, u)
        underline_row_anchor.setdefault(key_u, row_index)

    return table_cell_id_counter, open_rowspan, underline_row_anchor


# ============================================================
# STEP 1: Build table_cell foundation by merging temp_lines into table rows
# ============================================================

def process_table_rows(cells_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build table_cell_df from cells_df.

    Only layout_type == "table" is processed into table cells.
    """
    # --- Work only on table layouts ---
    if "layout_type" in cells_df.columns:
        df_tables = cells_df[cells_df["layout_type"] == "table"].copy()
    else:
        df_tables = cells_df.copy()

    if df_tables.empty:
        table_cell_df = pd.DataFrame(
            columns=[
                "table_cell_id",
                "page_number",
                "layout_id",
                "table_id",
                "row_start",
                "col_start",
                "rowspan",
                "colspan",
                "text",
                "text_raw_lines",
                "cell_ids",
                "temp_line_ids",
            ]
        )
        return table_cell_df

    if "shape_id_underline" not in df_tables.columns:
        df_tables["shape_id_underline"] = np.nan

    # ---- Precompute last temp_line_id per underline / layout / page ----
    underline_last_map = _compute_underline_last(df_tables)

    # ---- State for building table_cell_df ----
    records: List[Dict[str, Any]] = []
    table_cell_id_counter = 1
    table_id_counter = 1

    cell_meta: Dict[int, Dict[str, Any]] = {}
    cell_record_idx: Dict[int, int] = {}
    row_to_cell_ids: Dict[Tuple[int, int, int], List[int]] = {}
    underline_row_anchor: Dict[Tuple[int, int, int], int] = {}
    open_rowspan: Dict[Tuple[int, int, int], int] = {}

    # ---- Process by (layout_id, page_number) ----
    for (layout_id, page_number), df_seg in (
        df_tables.groupby(["layout_id", "page_number"], sort=True)
    ):
        if df_seg.empty:
            continue

        df_seg = df_seg.sort_values(["temp_line_id", "col_start", "cell_id"])

        table_id = table_id_counter
        table_id_counter += 1

        row_index = 1  # 1-based row indexing (NOTE: col_start is now 0-based)
        open_group_temp_line_ids: List[int] = []

        temp_line_ids = (
            df_seg["temp_line_id"]
            .dropna()
            .drop_duplicates()
            .sort_values()
            .tolist()
        )

        for t_line in temp_line_ids:
            df_line = df_seg[df_seg["temp_line_id"] == t_line]

            # --------------------------------------------------------
            # Step 1: attach-only check
            #   If this line has an underline whose row is already anchored,
            #   attach its cells to that row.
            #   NEW: if this happens and we have a pending group, attach the
            #   entire pending group to that same anchored row.
            # --------------------------------------------------------
            attached, anchor_rows = _try_attach_to_existing_row(
                df_line=df_line,
                layout_id=layout_id,
                page_number=page_number,
                records=records,
                cell_meta=cell_meta,
                cell_record_idx=cell_record_idx,
                row_to_cell_ids=row_to_cell_ids,
                underline_row_anchor=underline_row_anchor,
            )

            if attached:
                # If we had a pending group waiting for a flush, it should now
                # be considered "belonging" to the anchored row(s).
                if open_group_temp_line_ids:
                    pending_df = df_seg[
                        df_seg["temp_line_id"].isin(open_group_temp_line_ids)
                    ]
                    _attach_pending_group_to_rows(
                        group_df=pending_df,
                        layout_id=layout_id,
                        page_number=page_number,
                        anchor_rows=anchor_rows,
                        records=records,
                        cell_meta=cell_meta,
                        cell_record_idx=cell_record_idx,
                        row_to_cell_ids=row_to_cell_ids,
                    )
                    open_group_temp_line_ids = []
                # This line was fully consumed
                continue

            # --------------------------------------------------------
            # Step 2: normal grouping → decide when to flush to a new row
            # --------------------------------------------------------
            open_group_temp_line_ids.append(t_line)

            group_df = df_seg[df_seg["temp_line_id"].isin(open_group_temp_line_ids)]

            band_total_cols = int(group_df["band_total_cols"].max())
            covered_cols = _compute_covered_cols(group_df)

            is_complete = _is_complete_row(
                covered_cols=covered_cols,
                band_total_cols=band_total_cols,
                row_index=row_index,     # 1-based index of the row we're about to flush
            )

            line_underline_ids = (
                df_line["shape_id_underline"]
                .dropna()
                .astype(int)
                .unique()
                .tolist()
            )
            has_last_underline = _has_last_underline_instance(
                line_underline_ids=line_underline_ids,
                layout_id=layout_id,
                page_number=page_number,
                temp_line_id=t_line,
                underline_last_map=underline_last_map,
            )

            flush_reason = None
            if is_complete:
                flush_reason = "complete_row"
            elif has_last_underline:
                flush_reason = "last_underline"

            if flush_reason is not None:
                (
                    table_cell_id_counter,
                    open_rowspan,
                    underline_row_anchor,
                ) = _flush_group_to_row(
                    group_df=group_df,
                    layout_id=layout_id,
                    page_number=page_number,
                    table_id=table_id,
                    row_index=row_index,
                    band_total_cols=band_total_cols,
                    covered_cols=covered_cols,
                    records=records,
                    table_cell_id_counter=table_cell_id_counter,
                    cell_meta=cell_meta,
                    cell_record_idx=cell_record_idx,
                    row_to_cell_ids=row_to_cell_ids,
                    open_rowspan=open_rowspan,
                    underline_row_anchor=underline_row_anchor,
                    flush_reason=flush_reason,  # <— new
                )

                row_index += 1
                open_group_temp_line_ids = []

        # optional: you can decide to flush trailing incomplete groups

    # ============================================================
    # Build table_cell_df
    # ============================================================
    if records:
        table_cell_df = pd.DataFrame.from_records(records)
    else:
        table_cell_df = pd.DataFrame(
            columns=[
                "table_cell_id",
                "page_number",
                "layout_id",
                "table_id",
                "row_start",
                "col_start",
                "rowspan",
                "colspan",
                "text",
                "text_raw_lines",
                "cell_ids",
                "temp_line_ids",
            ]
        )
        return table_cell_df

    # ============================================================
    # FINAL: recompute text & text_raw_lines from cells_df
    # ============================================================

    # Build mapping from cell_id to table_cell_id for text recomputation
    cell_to_table_cell: Dict[Any, int] = {}
    for rec in records:
        tcell_id = rec["table_cell_id"]
        for cid in rec["cell_ids"]:
            cell_to_table_cell[cid] = tcell_id

    # Create temporary df with table_cell_id for aggregation
    df_for_agg = df_tables[df_tables["cell_id"].isin(cell_to_table_cell.keys())].copy()
    df_for_agg["table_cell_id"] = df_for_agg["cell_id"].map(cell_to_table_cell)

    tmp = df_for_agg.sort_values(["table_cell_id", "temp_line_id", "col_start", "cell_id"])

    def _agg_text(series: pd.Series) -> str:
        parts = [str(x).strip() for x in series if str(x).strip()]
        return " ".join(parts).strip()

    text_agg = (
        tmp.groupby("table_cell_id")["text"]
        .apply(_agg_text)
        .reset_index(name="text_new")
    )

    def _agg_text_raw_lines(group: pd.DataFrame) -> List[str]:
        out: List[str] = []
        for _, gline in group.groupby("temp_line_id", sort=True):
            parts = [str(x).strip() for x in gline["text"] if str(x).strip()]
            if parts:
                out.append(" ".join(parts).strip())
        return out

    raw_agg = (
        tmp.groupby("table_cell_id")
        .apply(_agg_text_raw_lines)
        .reset_index(name="text_raw_lines_new")
    )

    table_cell_df = (
        table_cell_df
        .merge(text_agg, on="table_cell_id", how="left")
        .merge(raw_agg, on="table_cell_id", how="left")
    )

    table_cell_df["text"] = table_cell_df["text_new"].fillna(table_cell_df["text"])
    table_cell_df["text_raw_lines"] = table_cell_df["text_raw_lines_new"].where(
        table_cell_df["text_raw_lines_new"].notna(),
        table_cell_df["text_raw_lines"],
    )

    table_cell_df = table_cell_df.drop(columns=["text_new", "text_raw_lines_new"])

    return table_cell_df


# ============================================================
# Merge table info back to cells_df
# ============================================================

def merge_table_info_to_cells(
    cells_df: pd.DataFrame,
    table_cell_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Merge table_cell_id, table_id, and row_start from table_cell_df back onto cells_df.
    
    For each cell_id in cells_df that appears in table_cell_df, we add:
    - table_cell_id: The ID of the table cell this raw cell belongs to
    - table_id: The ID of the table
    - row_start: The starting row index within the table
    
    Non-table cells will have NaN for these columns.
    """
    df_all = cells_df.copy()
    
    # Initialize new columns
    df_all["table_cell_id"] = np.nan
    df_all["table_id"] = np.nan
    df_all["row_start"] = np.nan
    
    if table_cell_df.empty:
        return df_all
    
    # Build mapping from cell_id to table info
    cell_to_info: Dict[Any, Dict[str, Any]] = {}
    for _, row in table_cell_df.iterrows():
        table_cell_id = row["table_cell_id"]
        table_id = row["table_id"]
        row_start = row["row_start"]
        
        for cell_id in row["cell_ids"]:
            cell_to_info[cell_id] = {
                "table_cell_id": table_cell_id,
                "table_id": table_id,
                "row_start": row_start,
            }
    
    # Map the info onto df_all
    mask = df_all["cell_id"].isin(cell_to_info.keys())
    
    if mask.any():
        df_all.loc[mask, "table_cell_id"] = df_all.loc[mask, "cell_id"].map(
            lambda cid: cell_to_info[cid]["table_cell_id"]
        )
        df_all.loc[mask, "table_id"] = df_all.loc[mask, "cell_id"].map(
            lambda cid: cell_to_info[cid]["table_id"]
        )
        df_all.loc[mask, "row_start"] = df_all.loc[mask, "cell_id"].map(
            lambda cid: cell_to_info[cid]["row_start"]
        )
    
    return df_all


# ============================================================
# Mark Table Cell Roles
# ============================================================

def _looks_like_numberish(text: str) -> bool:
    """Check if text looks like numeric data."""
    if not text or not text.strip():
        return False
    
    text = text.strip()
    
    # Remove common formatting characters
    clean = text.replace(',', '').replace('$', '').replace('€', '').replace('£', '')
    clean = clean.replace('(', '').replace(')', '').replace('%', '').replace(' ', '')
    
    # Check if it's mostly numeric
    if not clean:
        return False
    
    # Try to parse as float
    try:
        float(clean)
        return True
    except ValueError:
        pass
    
    # Check if it contains significant digits
    digit_count = sum(c.isdigit() for c in text)
    if digit_count >= 3:  # At least 3 digits suggests numeric data
        return True
    
    return False


def assign_cell_roles(table_cell_df: pd.DataFrame) -> pd.DataFrame:
    """
    Assign roles to each cell in the table.
    
    Roles:
    - "header": Header rows (first row + additional rows based on heuristics)
    - "row_label": First cell (col_start=0) of each data row
    - "value_numeric": Data cells that look numberish
    - "value_text": Data cells that don't look numberish
    - "footnote": Bottom row spanning multiple columns starting at col_start=0
    
    Returns:
        DataFrame with added 'role' column
    """
    if table_cell_df.empty:
        table_cell_df["role"] = pd.Series(dtype=str)
        return table_cell_df
    
    df = table_cell_df.copy()
    df["role"] = None
    
    # Header detection patterns
    year_pattern = re.compile(r'\b20\d{2}\b')  # 2000-2099
    date_pattern = re.compile(
        r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+20\d{2}\b|'
        r'\b\d{1,2}-(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)-\d{2,4}\b',
        re.IGNORECASE
    )
    unit_phrases = {
        'in thousands', 'in millions', 'in billions',
        'except per share', 'per share',
        'percentage', '(%)',
        'year ended', 'years ended',
        'months ended', 'month ended',
        'quarters ended', 'quarter ended',
        'total', 'actual', 'adjusted',
        'number', 'shares', 'amount', 'value',
    }
    
    # Process each table separately
    for (table_id, page_number, layout_id), table_group in df.groupby(
        ["table_id", "page_number", "layout_id"], sort=False
    ):
        table_indices = table_group.index
        
        # Get all rows in this table, sorted
        rows = sorted(table_group["row_start"].unique())
        if not rows:
            continue
        
        max_row = max(rows)
        
        # Step 1: Detect header rows
        header_rows = set([rows[0]])  # First row is always a header
        
        # Check if first row has rowspan cells
        first_row_cells = table_group[table_group["row_start"] == rows[0]]
        for _, cell in first_row_cells.iterrows():
            if cell["rowspan"] > 1:
                # Add all rows spanned by this cell as potential headers
                for r_offset in range(1, int(cell["rowspan"])):
                    spanned_row = rows[0] + r_offset
                    if spanned_row in rows:
                        header_rows.add(spanned_row)
        
        # Check subsequent rows (up to first 6 rows total for headers)
        for i, r in enumerate(rows[1:min(6, len(rows))], start=1):
            if r in header_rows:
                continue  # Already marked as header
            
            # Collect text from all cells that originate in this row
            row_cells = table_group[table_group["row_start"] == r]
            row_texts = []
            has_numberish_data = False
            
            for _, cell in row_cells.iterrows():
                text = str(cell["text"] or "").strip()
                if text:
                    row_texts.append(text.lower())
                    # Check if this looks like numeric data (not a year or date)
                    if _looks_like_numberish(text) and not year_pattern.search(text):
                        has_numberish_data = True
            
            # If row has numeric data, stop checking for more headers
            if has_numberish_data:
                break
            
            # Check for header indicators
            row_text_combined = ' '.join(row_texts)
            
            # Check for years (2024, 2025, etc.)
            if year_pattern.search(row_text_combined):
                header_rows.add(r)
                continue
            
            # Check for dates
            if date_pattern.search(row_text_combined):
                header_rows.add(r)
                continue
            
            # Check for unit phrases
            if any(phrase in row_text_combined for phrase in unit_phrases):
                header_rows.add(r)
                continue
            
            # If we didn't find any header indicators, stop
            break
        
        # Step 2: Detect potential footnote (bottom row)
        # Footnote: starts at col_start=0, has colspan >= 2, and is the only cell in that row
        bottom_row = max_row
        bottom_row_cells = table_group[table_group["row_start"] == bottom_row]

        is_footnote_row = False
        if len(bottom_row_cells) == 1:
            footnote_cell = bottom_row_cells.iloc[0]
            if int(footnote_cell["col_start"]) == 0 and int(footnote_cell["colspan"]) >= 2:
                is_footnote_row = True
        
        # Step 3: Assign roles to all cells in this table
        for idx, cell in table_group.iterrows():
            row_start = int(cell["row_start"])
            col_start = int(cell["col_start"])
            text = str(cell["text"] or "").strip()
            
            if row_start in header_rows:
                # Header cell
                df.at[idx, "role"] = "header"
            elif is_footnote_row and row_start == bottom_row:
                # Footnote cell
                df.at[idx, "role"] = "footnote"
            elif col_start == 0 and row_start not in header_rows:
                # First column (col_start=0) of data row = row label
                df.at[idx, "role"] = "row_label"
            else:
                # Data cell - determine if numeric or text
                if _looks_like_numberish(text):
                    df.at[idx, "role"] = "value_numeric"
                else:
                    df.at[idx, "role"] = "value_text"
    
    return df



# ============================================================
# Public API
# ============================================================

def process_tables(cells_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Process tables from cells_df.
    
    Returns:
        table_cell_df: DataFrame with table cell information
        cells_df_with_table_info: Original cells_df with table_cell_id, table_id, and row_start added
    """
    # Step 1: Build table cells
    table_cell_df = process_table_rows(cells_df)
    
    # Step 2: Assign cell roles
    table_cell_df = assign_cell_roles(table_cell_df)
    
    # Step 3: Merge table info back to cells_df
    cells_df_with_table_info = merge_table_info_to_cells(cells_df, table_cell_df)
    
    return table_cell_df, cells_df_with_table_info