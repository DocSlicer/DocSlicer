# step_04_line_merger.py

import re
import numpy as np
import pandas as pd

from docslicer._utils.layout.line_merger import assign_line_id, LineMergerConfig
from docslicer._utils.df_aggregation.registry_aggregator import aggregate_to
from docslicer._utils.df_aggregation.text_merge import (
    merge_table_rows,
    merge_text_within_line,
)


# =================================
# Build Line Text
# =================================

def _build_line_text(df: pd.DataFrame) -> pd.Series:
    """
    Build the text for each line: boxes sorted by x_left, non-table lines joined
    with spaces, table lines joined with pipes so downstream shared stages see a
    row-shaped representation. Inline markup (script/strikethrough) is already
    baked into box text by the box cleaner.

    Returns:
        Series of joined text indexed by line_id.
    """
    ordered = df.sort_values(["line_id", "x_left"], kind="stable")
    texts = ordered["text"].astype("string").fillna("")
    # <pre> code-line boxes keep their leading whitespace — indentation is
    # semantically meaningful in code. Everything else is stripped as before.
    if "struct_tag" in ordered.columns:
        is_pre = ordered["struct_tag"].astype("string").eq("pre").fillna(False)
        texts = texts.str.strip().where(~is_pre, texts.str.rstrip())
    else:
        texts = texts.str.strip()
    texts = texts.astype(str)

    prose_text = merge_text_within_line(texts, ordered["line_id"])

    if "table_id" not in ordered.columns:
        return prose_text

    tid = ordered["table_id"]
    is_tagged = tid.notna() & (tid.astype(str).str.strip() != "")
    if not is_tagged.any():
        return prose_text

    # A line is a table row if any of its boxes carries a table_id; pipe-join
    # all boxes on such lines and splice those rows over the prose default.
    table_line = is_tagged.groupby(ordered["line_id"]).transform("any")
    table_text = merge_table_rows(texts[table_line], ordered.loc[table_line, "line_id"])
    line_text = prose_text.copy()
    line_text.loc[table_text.index] = table_text
    return line_text


# =================================
# Table vs Text
# =================================

_STARTS_WITH_NUMBER_OR_PARENS = re.compile(r'^(\d|\([a-zA-Z0-9]+\)|[•◦▪▸·‣⁃●○►▶◆◇□■])')


def _remove_single_row_tables(df: pd.DataFrame) -> pd.DataFrame:
    """
    Process single-row tables: merge or remove them.

    Consecutive runs of >=5 single-row tables with equal table_row_cell_count where
    <=10% of rows start with a digit or a parenthesised token like (1)/(i)/(a) are
    merged into one multi-row table with sequential table_row_ids. All other
    single-row tables are removed (table_id/table_row_id cleared).

    Remaining tables are reindexed sequentially and marked block_type="table".
    """
    if df is None or df.empty:
        return df

    if "table_id" not in df.columns or "table_row_id" not in df.columns:
        return df

    df = df.copy()
    df["original_table_id"] = df["table_id"]
    df["original_table_row_id"] = df["table_row_id"]

    # --- 1. Identify single-row tables ---
    tables_with_rows = (
        df[df["table_id"].notna()]
        .groupby("table_id")["table_row_id"]
        .nunique()
    )
    single_row_ids = set(tables_with_rows[tables_with_rows == 1].index)
    is_single = df["table_id"].isin(single_row_ids)

    # --- 2. Build consecutive-run groups (breaks on non-single or cell-count change) ---
    has_cell_count = "table_row_cell_count" in df.columns
    cc_vals = (
        df["table_row_cell_count"].fillna(-1).astype(str).values
        if has_cell_count
        else ["same"] * len(df)
    )
    is_single_arr = is_single.values
    group_arr = [None] * len(df)
    current_group = 0

    for i in range(len(df)):
        if not is_single_arr[i]:
            continue
        if i == 0 or not is_single_arr[i - 1] or cc_vals[i] != cc_vals[i - 1]:
            current_group += 1
        group_arr[i] = current_group

    df["_srg"] = group_arr

    # --- 3. Decide merge vs remove per group ---
    merge_groups: set = set()
    remove_groups: set = set()

    for gid, grp in df[df["_srg"].notna()].groupby("_srg"):
        if len(grp) >= 5:
            texts = grp["text"].fillna("").astype(str)
            n_flagged = texts.apply(
                lambda t: bool(_STARTS_WITH_NUMBER_OR_PARENS.match(t.strip()))
            ).sum()
            if n_flagged / len(texts) <= 0.10:
                merge_groups.add(gid)
                continue
        remove_groups.add(gid)

    # --- 4. Remove ---
    to_remove = df["_srg"].isin(remove_groups)
    df.loc[to_remove, "table_id"] = None
    df.loc[to_remove, "table_row_id"] = None
    if "text" in df.columns:
        df.loc[to_remove, "text"] = df.loc[to_remove, "text"].str.replace(" | ", " ", regex=False)

    # --- 5. Merge ---
    if merge_groups:
        existing_max = df["table_id"].max()
        next_tid = int(existing_max) + 1 if pd.notna(existing_max) else 1

        for gid in sorted(merge_groups):
            mask = df["_srg"] == gid
            indices = df[mask].index
            df.loc[mask, "table_id"] = next_tid
            for row_num, idx in enumerate(indices, start=1):
                df.loc[idx, "table_row_id"] = row_num
            next_tid += 1

    df = df.drop(columns=["_srg"])

    # --- 6. Reindex table_ids by first appearance ---
    if df["table_id"].notna().any():
        seen: dict = {}
        for tid in df.loc[df["table_id"].notna(), "table_id"]:
            if tid not in seen:
                seen[tid] = len(seen) + 1
        df["table_id"] = df["table_id"].map(lambda x: seen.get(x) if pd.notna(x) else None)

    # --- 7. Mark block_type = "table" ---
    if "block_type" not in df.columns:
        df["block_type"] = None
    has_table = df["table_id"].notna()
    no_existing_type = df["block_type"].isna()
    df.loc[has_table & no_existing_type, "block_type"] = "table"

    return df


# =================================
# Layout ID
# =================================

# Style columns compared line-to-line for layout grouping (x_left is compared
# separately, rounded to whole points).
_LAYOUT_STYLE_COLS = [
    "font_family",
    "font_size",
    "text_align",
    "non_stroking_color",
    "is_bold",
    "is_italic",
    "is_uppercase",
]

# Vertical gap (pt) between one line's y_bottom and the next line's y_top at or
# above which the next line starts a new layout. A negative gap (moving back up
# the page, e.g. a column jump) also breaks.
_LAYOUT_GAP_PT = 10.0

_LIST_TAGS = frozenset({"ul", "ol"})
_HEADING_TAGS = frozenset({"h1", "h2", "h3", "h4", "h5", "h6"})
_PRE_TAGS = frozenset({"pre"})


def _add_line_gap(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add ``line_gap``: the vertical gap (pt) above each line, i.e.
    ``y_top[i] - y_bottom[i-1]`` within a page, 0.0 for the first line of a page.

    Mirrors the PDF definition (``_assign_gaps`` in _utils/layout/layouts.py) so
    the shared heading scorer sees the same column on both pipelines.  Stays NaN
    when coordinates are absent (static extraction).
    """
    out = df.copy()
    if "y_top" not in out.columns or "y_bottom" not in out.columns:
        out["line_gap"] = np.nan
        return out

    y_top = pd.to_numeric(out["y_top"], errors="coerce")
    y_bottom = pd.to_numeric(out["y_bottom"], errors="coerce")
    if "page_number" in out.columns:
        prev_bottom = y_bottom.groupby(out["page_number"], sort=False).shift()
    else:
        prev_bottom = y_bottom.shift()

    out["line_gap"] = (y_top - prev_bottom).fillna(0.0)   # first line of page -> 0
    return out


def _struct_layout_groups(df: pd.DataFrame) -> pd.Series:
    """
    Per line, a grouping key from the deepest list (ul/ol), heading (h1-h6), or
    code block (pre) struct ancestor: ``"list_<id>"`` / ``"heading_<id>"`` /
    ``"pre_<id>"``, else None.

    Ancestors run root -> leaf, so the last matching tag is the innermost one —
    items of a nested <ol> group under the nested list's id, and a heading
    inside a list item groups under whichever of the two sits deeper. All code
    lines of one <pre> share its id, so the block stays in a single layout.
    Consecutive lines sharing a key are kept in one layout by _add_layout_id.
    """
    keys = np.full(len(df), None, dtype=object)
    if "struct_ancestors" not in df.columns or "struct_ancestor_ids" not in df.columns:
        return pd.Series(keys, index=df.index)

    anc_arr = df["struct_ancestors"].to_numpy(dtype=object)
    aid_arr = df["struct_ancestor_ids"].to_numpy(dtype=object)

    for i in range(len(df)):
        ancs = anc_arr[i] if isinstance(anc_arr[i], (list, tuple)) else []
        aids = aid_arr[i] if isinstance(aid_arr[i], (list, tuple)) else []
        key = None
        for tag, id_ in zip(ancs, aids):
            if tag in _LIST_TAGS:
                key = f"list_{id_}"
            elif tag in _HEADING_TAGS:
                key = f"heading_{id_}"
            elif tag in _PRE_TAGS:
                key = f"pre_{id_}"
        keys[i] = key
    return pd.Series(keys, index=df.index)


def _add_layout_id(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add layout_id column grouping consecutive lines into layouts.

    Rules:
    1. Start at 1 for the very first line
    2. New id when page_number changes
    3. Grouped lines share one layout_id for consecutive runs of the same
       group key (new id when entering/leaving/switching groups). A line's
       group key is, in order of precedence:
       - its table_id
       - its deepest ul/ol or h1-h6 struct ancestor (see
         :func:`_struct_layout_groups`), so list items and multi-line
         headings stay together
    4. Ungrouped lines continue the previous line's layout unless:
       - block_type changes from the previous line (hr, image, page label, ...)
       - the style set changes: x_left (rounded to whole pt), font_family,
         font_size, text_align, non_stroking_color, is_bold, is_italic,
         is_uppercase
       - the vertical gap to the previous line is >= 10pt or negative

    Args:
        df: DataFrame with lines

    Returns:
        DataFrame with layout_id column added
    """
    if df is None or df.empty:
        return df

    df = df.copy()
    prev = df.shift(1)
    false = pd.Series(False, index=df.index)

    def _neq(cur: pd.Series, prv: pd.Series) -> pd.Series:
        """NaN-safe inequality: NaN == NaN counts as equal."""
        return ~((cur == prv) | (cur.isna() & prv.isna()))

    # Rule 2: page change always breaks (and splits groups spanning pages)
    new_page = _neq(df["page_number"], prev["page_number"]) if "page_number" in df.columns else false

    # Rule 3: group key — table_id wins over the struct list/heading group
    group_key = _struct_layout_groups(df)
    if "table_id" in df.columns:
        has_table = df["table_id"].notna()
        group_key = group_key.mask(has_table, "table_" + df["table_id"].astype(str))
    prev_key = group_key.shift(1)
    grouped = group_key.notna()
    group_break = grouped & _neq(group_key, prev_key)

    # Rule 4: ungrouped lines break on block_type / style / vertical jump
    style_change = false.copy()
    if "block_type" in df.columns:
        style_change |= _neq(df["block_type"], prev["block_type"])
    if "x_left" in df.columns:
        style_change |= _neq(df["x_left"].round(0), prev["x_left"].round(0))
    for col in _LAYOUT_STYLE_COLS:
        if col in df.columns:
            style_change |= _neq(df[col], prev[col])

    gap_break = false
    if "y_top" in df.columns and "y_bottom" in df.columns:
        gap = df["y_top"] - prev["y_bottom"]
        gap_break = ((gap >= _LAYOUT_GAP_PT) | (gap < 0)).fillna(False)

    ungrouped_break = ~grouped & (prev_key.notna() | style_change | gap_break)

    breaks = (new_page | group_break | ungrouped_break).astype(bool)
    breaks.iloc[0] = True  # Rule 1
    df["layout_id"] = breaks.cumsum()
    return df


# =========================
# Public API
# =========================
def merge_boxes_to_lines(
    boxes_df: pd.DataFrame,
    remove_single_row_tables: bool = True,
    merge_by_coordinates: bool = True,
) -> pd.DataFrame:
    """
    Merge boxes into lines:
    1. Assign line_id using line_merger
    2. Merge text within each line (sorted by x_left) via text_merge
    3. Aggregate the remaining columns via the registry aggregator
    4. Optionally remove single-row tables and reindex
    5. Add layout_id

    Args:
        boxes_df: DataFrame with box-level data
        remove_single_row_tables: If True, remove table_id/table_row_id for tables with only 1 row
        merge_by_coordinates: When False, skip y-tolerance merging and give every non-table box
            its own line. Use for statically extracted boxes where y_top/y_bottom are all 0.
    """
    if boxes_df is None or boxes_df.empty:
        return boxes_df

    # Step 1: Assign line_id
    # When coordinates are absent (static extraction), y_top/y_bottom are all 0, which
    # causes every box to merge into one line via the tolerance check. Fix: synthetically
    # space y values so non-table boxes are always far enough apart to avoid merging.
    # Table cells still merge correctly via table_row_id (checked before coordinates).
    if not merge_by_coordinates:
        # With y_top=0 everywhere, assign_line_id's tolerance check merges everything.
        # Give every logical row a unique y so only table_row_id-based merging applies.
        # - Non-table boxes: unique y per box (each becomes its own line)
        # - Table cells: same y for cells sharing a table_row_id, unique per row group
        #   (coordinate check is skipped because table_row_id match fires first, but
        #    cells from *different* rows would also have dy=0 and spuriously merge)
        _STRIDE = LineMergerConfig().TOL_EXPANDED + 1  # > any merge tolerance
        boxes_df = boxes_df.copy()
        has_row_id = "table_row_id" in boxes_df.columns
        is_table_box = boxes_df["table_row_id"].notna() if has_row_id else pd.Series(False, index=boxes_df.index)

        # Assign a rank to each unique (table_id, table_row_id) pair in document order
        if has_row_id and is_table_box.any():
            row_groups = (
                boxes_df[is_table_box]
                .groupby(["table_id", "table_row_id"], sort=False)
                .ngroup()
            )

        sequential_y = pd.Series(range(len(boxes_df)), index=boxes_df.index, dtype=float) * _STRIDE
        synthetic_y = sequential_y.copy()

        if has_row_id and is_table_box.any():
            # All cells in the same TR share the same y (their row group rank × stride)
            synthetic_y[is_table_box] = row_groups.values * _STRIDE

        boxes_df["y_top"] = synthetic_y
        boxes_df["y_bottom"] = synthetic_y

    boxes_with_lines = assign_line_id(boxes_df, y_alignment="top")
    
    # Step 2: Create text for each line (sorted by x_left, joined with spaces,
    # pipe-joined on table lines). Text is merged from the x-sorted view while
    # aggregation runs on the frame in document order, so "first"/dominant
    # columns keep picking the same source boxes as before.
    line_text = _build_line_text(boxes_with_lines)

    # Step 3: Aggregate everything else via the central column registry
    lines_df = aggregate_to(
        boxes_with_lines,
        by="line_id",
        size_as="box_count",
    )
    lines_df["text"] = lines_df["line_id"].map(line_text)

    # Step 4: Remove single-row tables if requested
    if remove_single_row_tables:
        lines_df = _remove_single_row_tables(lines_df)

    # Step 5: Add layout_id column
    lines_df = _add_layout_id(lines_df)

    # Step 6: Add line_gap (vertical gap above each line, pt)
    lines_df = _add_line_gap(lines_df)

    return lines_df
