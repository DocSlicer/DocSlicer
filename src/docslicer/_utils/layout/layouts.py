"""
Horizontal band assignment.

Public API:
    df = assign_horizontal_bands(df, merge_by_vertical_lines=False)

Accepts any DataFrame that has a `line_id` column — words, cells, or lines.
When multiple rows share the same line_id the per-line geometry (y_top, y_bottom)
is derived vectorized before band assignment and the results are joined back.

Pipeline (internal, always line-level):
    1. Derive one row per line_id: min(y_top), max(y_bottom), first page_number,
       first block_type, first text_orientation.
    2. Sort by (page_number, line_id) — line_id already encodes reading order.
    3. Compute per-line vertical gap.  A large negative gap means the reading
       order jumped to a new column; it opens a new band and resets the
       prev_y_bottom reference.
    4. Compute adaptive gap threshold per page (median of positive gaps ×
       interpolated multiplier).
    5. Assign layout_id (1-based, monotonically increasing).
    6. Optionally merge bands by shared vertical grid-line IDs (union-find).
    7. Join layout_id, line_gap, median_gap, page_gap_thresh back
       onto the original input df.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

_MIN_PAGE_GAP_THRESH: float = 3.5

_HEADING_TAGS = frozenset({"H", "H1", "H2", "H3", "H4", "H5", "H6"})
_LIST_TAG     = "L"
_PARA_TAG     = "P"

# Style-change split thresholds (see _style_change_flags)
_DEFAULT_FONT_SIZE_SPLIT_DELTA: float = 1.0   # pt; |Δfont_size| ≥ this splits (toggle to 2.0 if too sensitive)

# Untagged-table line merge (see _merge_untagged_table_lines)
_MAX_SINGLE_CELL_BRIDGE: int   = 2     # max consecutive 1-cell rows allowed to bridge two table segments
_MAX_TABLE_ROW_GAP:      float = 20.0  # pt; a gap larger than this breaks the run (likely a separate table)
_TABLE_BRIDGE_FONT_TOL:  float = 2.0   # pt; a bridged 1-cell row must be within this of the table font size


# ============================================================
# HELPERS
# ============================================================

def _interpolate_gap_multiplier(
    median_gap: float,
    low_gap:    float = 3.0,
    high_gap:   float = 10.0,
    mult_at_low:  float = 1.60,
    mult_at_high: float = 1.10,
) -> float:
    """
    Adaptive multiplier for the band-gap threshold.

    Tighter line spacing → higher multiplier (more sensitive detection).
    Looser line spacing  → lower multiplier (avoids over-splitting).

        median_gap ≤ low_gap  → mult_at_low
        median_gap ≥ high_gap → mult_at_high
        in between            → linear interpolation
    """
    if median_gap <= low_gap:
        return mult_at_low
    if median_gap >= high_gap:
        return mult_at_high
    t = (median_gap - low_gap) / (high_gap - low_gap)
    return mult_at_low + t * (mult_at_high - mult_at_low)


def _struct_group_ids(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Per row, extract the deepest (last-occurring) heading, list and paragraph
    ancestor ids from struct_ancestors / struct_ancestor_ids.

        heading_id — id of the deepest H/H1-H6 ancestor, else None
        list_id    — id of the deepest L ancestor, else None
        para_id    — id of the deepest P ancestor, else None

    Ancestors run root → leaf, so the last matching tag is the innermost one
    (e.g. a nested <L> inside an outer <L><LI><LBody> yields the nested id).
    Returns arrays of None when the struct columns are absent.
    """
    n = len(df)
    heading_ids = np.empty(n, dtype=object); heading_ids[:] = None
    list_ids    = np.empty(n, dtype=object); list_ids[:]    = None
    para_ids    = np.empty(n, dtype=object); para_ids[:]    = None

    if "struct_ancestors" not in df.columns or "struct_ancestor_ids" not in df.columns:
        return heading_ids, list_ids, para_ids

    anc_arr = df["struct_ancestors"].to_numpy(dtype=object)
    aid_arr = df["struct_ancestor_ids"].to_numpy(dtype=object)

    for i in range(n):
        ancs = anc_arr[i] if isinstance(anc_arr[i], (list, tuple)) else []
        aids = aid_arr[i] if isinstance(aid_arr[i], (list, tuple)) else []
        for tag, id_ in zip(ancs, aids):
            if tag in _HEADING_TAGS:
                heading_ids[i] = id_
            elif tag == _LIST_TAG:
                list_ids[i] = id_
            elif tag == _PARA_TAG:
                para_ids[i] = id_

    return heading_ids, list_ids, para_ids


def _isna_scalar(v: object) -> bool:
    """NaN/None test that is safe for non-scalar cells (tuple/list/array colors)."""
    try:
        return bool(pd.isna(v))
    except (TypeError, ValueError):
        return v is None


def _style_change_flags(
    df:              pd.DataFrame,
    font_size_delta: float,
) -> np.ndarray:
    """
    Per line, flag when the visual style differs from the previous line.

    Style key: {font_family, font_size, is_bold, is_italic, non_stroking_color}.
    font_family / is_bold / is_italic are optional (the OCR pipeline lacks them)
    — absent columns and missing (NaN) cells never fire.

    A True at index i means "line i looks different from line i-1"; the caller
    uses it as an extra split trigger. It is only ever a *split* signal — struct
    rules keep priority and a struct force-merge still wins. Index 0 is always
    False (no previous line).

    Trigger rules
    -------------
        font_size    → split on |Δ| ≥ font_size_delta pt (both directions); hinting
                       jitter such as 9.63 vs 9.85 stays well under 1 pt and never fires
        font_family / is_bold / is_italic / non_stroking_color
                     → split on any change, compared only when both cells are
                       present (NaN-vs-value and NaN-vs-NaN never fire)
    """
    n = len(df)
    changed = np.zeros(n, dtype=bool)
    if n < 2:
        return changed

    fsize  = df["font_size"].to_numpy(dtype=float) if "font_size" in df.columns else None
    cats   = [
        df[c].to_numpy(dtype=object)
        for c in ("font_family", "is_bold", "is_italic", "non_stroking_color")
        if c in df.columns
    ]

    for i in range(1, n):
        c = False

        if fsize is not None and not (np.isnan(fsize[i]) or np.isnan(fsize[i - 1])):
            if abs(fsize[i] - fsize[i - 1]) >= font_size_delta - 1e-9:
                c = True

        if not c:
            for arr in cats:
                a, b = arr[i], arr[i - 1]
                if _isna_scalar(a) or _isna_scalar(b):
                    continue
                try:
                    differs = bool(a != b)
                except (TypeError, ValueError):
                    differs = a is not b
                if differs:
                    c = True
                    break

        changed[i] = c

    return changed


# ============================================================
# STEP 1: derive one row per line_id
# ============================================================

def _to_line_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate to one row per line_id.

    For each line_id takes:
        y_top    → min across rows  (topmost edge of the line)
        y_bottom → max across rows  (bottommost edge)
        page_number, block_type, text_orientation, shape_id_vertical_grid_line
                 → first value (they are constant within a line_id)
    """
    agg: dict[str, object] = {
        "y_top":    ("y_top",    "min"),
        "y_bottom": ("y_bottom", "max"),
    }
    # x_left → line start (leftmost); font_size → median (robust to a stray glyph)
    if "x_left" in df.columns:
        agg["x_left"] = ("x_left", "min")
    if "font_size" in df.columns:
        agg["font_size"] = ("font_size", "median")
    for col in (
        "page_number", "block_type", "text_orientation",
        "table_id", "struct_ancestors", "struct_ancestor_ids",
        "font_family", "is_bold", "is_italic", "non_stroking_color",
    ):
        if col in df.columns:
            agg[col] = (col, "first")

    line_df = df.groupby("line_id", sort=False).agg(**agg).reset_index()

    if "shape_id_vertical_grid_line" in df.columns:
        vlines = (
            df.groupby("line_id", sort=False)["shape_id_vertical_grid_line"]
            .apply(lambda s: list({v for cell in s if isinstance(cell, list) for v in cell}
                                  or ({s.dropna().iloc[0]} if not s.isna().all() else set())))
            .reset_index()
        )
        line_df = line_df.merge(vlines, on="line_id", how="left")

    return line_df


# ============================================================
# STEP 2–5: gap computation + band assignment (one page at a time)
# ============================================================

def _process_page(
    ltr_df:      pd.DataFrame,
    band_counter: int,
    font_size_split_delta: float = _DEFAULT_FONT_SIZE_SPLIT_DELTA,
) -> tuple[dict[object, float], dict[object, int], int, float, float]:
    """
    Compute line_gap and layout_id for the LTR lines of one page.

    Struct-tree evidence (table_id, heading/list ancestor ids) overrides the
    gap-based decision at each line boundary when present — see the Phase 3
    loop below. Columns absent → those overrides simply never fire.

    Parameters
    ----------
    ltr_df : already sorted by line_id, non-TTB/BTT lines only
    band_counter : running global band counter

    Returns
    -------
    gaps       : {line_id → gap}
    band_ids   : {line_id → band_id}
    band_counter : updated counter
    median_gap, threshold
    """
    line_ids  = ltr_df["line_id"].to_numpy()
    y_top_arr = ltr_df["y_top"].to_numpy(dtype=float)
    y_bot_arr = ltr_df["y_bottom"].to_numpy(dtype=float)
    # NOTE: .fillna("") before astype(str) is required. Under pandas 3 / numpy 2
    # the new `str` dtype keeps missing values as NaN sentinels (astype(str) no
    # longer renders them as the literal "nan"), and NaN != NaN is True — so two
    # lines that both lack a block_type would compare unequal and force a split
    # on every row, defeating the table/heading/list merge below.
    bt_arr    = (
        ltr_df["block_type"].fillna("").astype(str).str.lower().to_numpy()
        if "block_type" in ltr_df.columns
        else np.full(len(ltr_df), "", dtype=object)
    )
    n = len(line_ids)

    table_arr = (
        ltr_df["table_id"].to_numpy(dtype=object)
        if "table_id" in ltr_df.columns
        else np.full(n, None, dtype=object)
    )
    heading_arr, list_arr, para_arr = _struct_group_ids(ltr_df)
    style_changed = _style_change_flags(ltr_df, font_size_split_delta)

    # ── Phase 1: compute gaps ────────────────────────────────────────────────
    raw_gaps   = np.zeros(n, dtype=float)
    page_gaps: list[float] = []
    prev_yb: float | None  = None

    for i in range(n):
        gap = (float(y_top_arr[i]) - prev_yb) if prev_yb is not None else 0.0
        raw_gaps[i] = gap
        page_gaps.append(gap if gap > 0 else 2.0)
        prev_yb = float(y_bot_arr[i])

    # ── Phase 2: adaptive threshold ─────────────────────────────────────────
    if page_gaps:
        median_gap = float(np.median(page_gaps))
        threshold  = max(_MIN_PAGE_GAP_THRESH, _interpolate_gap_multiplier(median_gap) * median_gap)
    else:
        median_gap = 0.0
        threshold  = float("inf")

    # ── Phase 3: assign band IDs ─────────────────────────────────────────────
    band_arr  = np.zeros(n, dtype=int)
    cur_band  = 0
    prev_yb   = None

    prev_table   = None
    prev_heading = None
    prev_list    = None
    prev_para    = None
    prev_bt      = None

    for i in range(n):
        gap = raw_gaps[i]
        yb  = float(y_bot_arr[i])

        if bt_arr[i] == "page_label":
            band_counter += 1
            band_arr[i]   = band_counter
            cur_band       = 0
            prev_yb        = yb
            prev_table     = table_arr[i]
            prev_heading   = heading_arr[i]
            prev_list      = list_arr[i]
            prev_para      = para_arr[i]
            prev_bt        = bt_arr[i]
            continue

        # ── struct-tree override (PDF only; all arrays are None when disabled) ──
        # Priority: block_type change always splits; else table/heading/list/para
        # membership force-merges within a group and force-splits at its edges.
        # Paragraph (P) is the lowest-priority grouping: it only decides when the
        # line is not held together by a table or list — a P inside a <Table> or
        # <L> is governed by those, so its id changing does nothing there.
        # NOTE: use pd.isna (not `is not None`) — a table_id column mixing None
        # with real ids gets upcast to float64, turning missing values into NaN.
        cur_table_na, prev_table_na     = pd.isna(table_arr[i]),   pd.isna(prev_table)
        cur_heading_na, prev_heading_na = pd.isna(heading_arr[i]), pd.isna(prev_heading)
        cur_list_na, prev_list_na       = pd.isna(list_arr[i]),    pd.isna(prev_list)
        cur_para_na, prev_para_na       = pd.isna(para_arr[i]),    pd.isna(prev_para)

        forced_new  = False
        forced_same = False
        if cur_band != 0:
            if bt_arr[i] != prev_bt:
                forced_new = True
            elif not cur_table_na or not prev_table_na:
                forced_same = (not cur_table_na) and table_arr[i] == prev_table
                forced_new  = not forced_same
            elif not cur_heading_na or not prev_heading_na:
                forced_same = (not cur_heading_na) and heading_arr[i] == prev_heading
                forced_new  = not forced_same
            elif not cur_list_na or not prev_list_na:
                forced_same = (not cur_list_na) and list_arr[i] == prev_list
                forced_new  = not forced_same
            elif not cur_para_na or not prev_para_na:
                forced_same = (not cur_para_na) and para_arr[i] == prev_para
                forced_new  = not forced_same

        # Style change is a split-only trigger and never overrides a struct
        # force-merge (it is gated behind `not forced_same`): two lines close
        # enough to merge on gap still split when their style key differs.
        is_col_jump = (gap < -threshold) and (prev_yb is not None)
        if forced_new or (not forced_same and (
            cur_band == 0 or gap > threshold or is_col_jump or style_changed[i]
        )):
            band_counter += 1
            cur_band       = band_counter

        band_arr[i]  = cur_band
        prev_yb      = yb
        prev_table   = table_arr[i]
        prev_heading = heading_arr[i]
        prev_list    = list_arr[i]
        prev_para    = para_arr[i]
        prev_bt      = bt_arr[i]

    gaps_map     = dict(zip(line_ids, raw_gaps))
    band_ids_map = dict(zip(line_ids, band_arr))
    return gaps_map, band_ids_map, band_counter, median_gap, threshold


# ============================================================
# STEP 6: merge bands by shared vertical lines (optional)
# ============================================================

def _merge_bands_by_shared_vertical_lines(
    line_df: pd.DataFrame,
    min_shared_lines: int = 1,
) -> pd.DataFrame:
    """
    For each page, merge bands that share ≥ min_shared_lines vertical grid-line IDs.
    Uses union-find; merged bands adopt the smallest band_id.
    Returns line_df unchanged if shape_id_vertical_grid_line is absent.
    """
    if "shape_id_vertical_grid_line" not in line_df.columns:
        return line_df

    df = line_df.copy()

    for _, page_df in df.groupby("page_number", sort=False):
        page_idx = page_df.index

        band_to_vlines: dict[int, set] = {}
        for idx in page_idx:
            bid = df.at[idx, "layout_id"]
            if pd.isna(bid) or bid < 0:
                continue
            bid    = int(bid)
            vlines = df.at[idx, "shape_id_vertical_grid_line"]
            if vlines is None or (isinstance(vlines, float) and pd.isna(vlines)):
                continue
            if not isinstance(vlines, list):
                vlines = [vlines]
            band_to_vlines.setdefault(bid, set()).update(vlines)

        bids = sorted(band_to_vlines)
        if len(bids) < 2:
            continue

        parent = {b: b for b in bids}

        def find(b: int) -> int:
            while parent[b] != b:
                parent[b] = parent[parent[b]]
                b = parent[b]
            return b

        for i in range(len(bids)):
            for j in range(i + 1, len(bids)):
                b1, b2 = bids[i], bids[j]
                if len(band_to_vlines[b1] & band_to_vlines[b2]) >= min_shared_lines:
                    r1, r2 = find(b1), find(b2)
                    if r1 != r2:
                        parent[max(r1, r2)] = min(r1, r2)

        for idx in page_idx:
            bid = df.at[idx, "layout_id"]
            if pd.isna(bid) or bid < 0:
                continue
            bid = int(bid)
            if bid in parent:
                df.at[idx, "layout_id"] = find(bid)

    return df


# ============================================================
# STEP 6b: pull untagged table rows into one layout (optional)
# ============================================================

def _merge_untagged_table_lines(
    line_df:                pd.DataFrame,
    max_single_cell_bridge: int   = _MAX_SINGLE_CELL_BRIDGE,
    max_table_row_gap:      float = _MAX_TABLE_ROW_GAP,
    bridge_font_tol:        float = _TABLE_BRIDGE_FONT_TOL,
) -> pd.DataFrame:
    """
    Merge the layout_id of consecutive multi-cell lines that form an *untagged*
    table (no struct table_id) so the whole table reads as one layout.

    Per page, walking lines in reading order (line_id):
        - a line with cell_count ≥ 2 opens/extends a table run;
        - up to ``max_single_cell_bridge`` consecutive 1-cell lines may bridge two
          multi-cell segments, but only when another multi-cell line follows and
          each bridging line's font_size is within ``bridge_font_tol`` pt of the
          run's last multi-cell line (leading/trailing 1-cell lines are never
          pulled in — a footnote block below a table stays out);
        - a vertical gap (y_top − prev y_bottom) larger than ``max_table_row_gap``
          pt breaks the run — that usually means a second, separate table.
    Every line in a run of ≥ 2 lines is relabelled to the run's smallest
    layout_id. Rows already inside a tagged table (non-null table_id) are treated
    as hard run boundaries and never merged, so real struct tables are untouched.

    No-op when cell_count is absent.
    """
    if "cell_count" not in line_df.columns:
        return line_df

    has_table = "table_id"  in line_df.columns
    has_font  = "font_size" in line_df.columns

    updates: dict[object, int] = {}   # original index label → new layout_id

    for _, page_df in line_df.groupby("page_number", sort=False):
        pdf   = page_df.sort_values("line_id", kind="mergesort")
        idx   = pdf.index.to_numpy()
        cc    = pdf["cell_count"].fillna(0).to_numpy()
        ytop  = pdf["y_top"].to_numpy(dtype=float)
        ybot  = pdf["y_bottom"].to_numpy(dtype=float)
        lay   = pdf["layout_id"].to_numpy()
        fs    = pdf["font_size"].to_numpy(dtype=float) if has_font  else np.full(len(pdf), np.nan)
        tagged = pdf["table_id"].notna().to_numpy()    if has_table else np.zeros(len(pdf), dtype=bool)
        n = len(pdf)

        def gap(m: int) -> float:                       # gap above local line m (m ≥ 1)
            return float(ytop[m] - ybot[m - 1])

        def is_row(m: int) -> bool:                     # eligible multi-cell (untagged) table row
            return (cc[m] >= 2) and not tagged[m]

        i = 0
        while i < n:
            if not is_row(i):
                i += 1
                continue

            run        = [i]
            last_multi = i
            j          = i + 1
            while j < n:
                if tagged[j] or gap(j) > max_table_row_gap:
                    break
                if cc[j] >= 2:
                    run.append(j)
                    last_multi = j
                    j += 1
                    continue

                # 1-cell run: only bridge if another multi-cell row follows within limit
                k = j
                while k < n and cc[k] < 2 and not tagged[k]:
                    k += 1
                n_single = k - j
                if k < n and is_row(k) and n_single <= max_single_cell_bridge:
                    gaps_ok = all(gap(m) <= max_table_row_gap for m in range(j, k + 1))
                    font_ok = (not has_font) or all(
                        (not np.isnan(fs[m]) and not np.isnan(fs[last_multi])
                         and abs(fs[m] - fs[last_multi]) <= bridge_font_tol)
                        for m in range(j, k)
                    )
                    if gaps_ok and font_ok:
                        run.extend(range(j, k + 1))
                        last_multi = k
                        j = k + 1
                        continue
                break   # trailing / too-many / mismatched 1-cell rows end the run

            if len(run) >= 2:
                target = int(min(lay[m] for m in run))
                for m in run:
                    updates[idx[m]] = target
            i = j

    if not updates:
        return line_df

    out = line_df.copy()
    for ix, val in updates.items():
        out.at[ix, "layout_id"] = val
    return out


# ============================================================
# PUBLIC API
# ============================================================

def assign_layouts(
    df: pd.DataFrame,
    line_level: bool = True,
    merge_by_vertical_lines: bool = False,
    min_shared_lines: int = 1,
    font_size_split_delta: float = _DEFAULT_FONT_SIZE_SPLIT_DELTA,
    merge_untagged_tables: bool = True,
    max_single_cell_bridge: int = _MAX_SINGLE_CELL_BRIDGE,
    max_table_row_gap: float = _MAX_TABLE_ROW_GAP,
    table_bridge_font_tol: float = _TABLE_BRIDGE_FONT_TOL,
) -> pd.DataFrame:
    """
    Assign layout_id to every row of df.

    Accepts any DataFrame that has a `line_id` column.  When multiple rows share
    the same line_id (e.g. words or cells), per-line geometry is derived
    vectorized (min y_top, max y_bottom) before band assignment, and results are
    joined back onto the original rows.

    Assumptions
    -----------
    - line_id encodes reading order (monotonically increasing in the order lines
      should be read).  Column jumps appear as large negative y gaps and
      automatically open a new band.
    - page_number is present and identifies page boundaries.

    Struct-tree refinement (PDF only, automatic)
    ---------------------------------------------
    When table_id / struct_ancestors / struct_ancestor_ids are present, they
    override the gap-based decision at each line boundary:
        - block_type change            → always splits (new layout_id)
        - same table_id                → always merges, regardless of gap;
                                          table start/end always splits
        - same deepest heading (H*) id → always merges; its start/end splits
        - same deepest list (L) id     → always merges; its start/end splits
                                          (only the innermost L per line counts
                                          — outer/ancestor L ids never merge)
        - same deepest paragraph (P) id → always merges; a change in P id splits.
                                          Lowest priority: only consulted when the
                                          line is not held by a table or list, so a
                                          <P> nested in a <Table>/<L> never decides.
    No-op (plain gap-based bands) when these columns are absent, e.g. OCR
    input or a PDF with no struct tree.

    Style refinement (split-only, never overrides struct)
    -----------------------------------------------------
    On top of gap/struct, a change in the per-line style key forces a split even
    when the lines are close enough (gap < page median) to otherwise merge. It is
    a split-only signal — a struct force-merge (same table/heading/list/P) still
    wins. Style key: {font_family, font_size, is_bold, is_italic,
    non_stroking_color}; font_family/is_bold/is_italic are optional (OCR lacks
    them) and absent/NaN cells never fire. font_size splits on
    |Δ| ≥ font_size_split_delta pt; the rest split on any change.
    See _style_change_flags.

    Parameters
    ----------
    df : pd.DataFrame
        Words, cells, or lines.  Required: line_id, page_number, y_top, y_bottom.
        Optional: block_type, text_orientation, shape_id_vertical_grid_line,
        and the style-key columns above.
    font_size_split_delta : float
        Minimum |Δfont_size| (pt) between adjacent lines that triggers a style
        split. Default 1.0; raise to 2.0 if 1 pt proves too sensitive.
    merge_by_vertical_lines : bool
        Merge bands sharing vertical grid-line IDs (union-find).  Use True for
        PDF (reliable line detection).  Leave False for OCR — line detection is
        too noisy and risks collapsing an entire page into one band.
    min_shared_lines : int
        Minimum shared vertical line IDs required to merge two bands.
        Only relevant when merge_by_vertical_lines=True.
    merge_untagged_tables : bool
        After band assignment, pull consecutive multi-cell lines (cell_count ≥ 2)
        that form an untagged table into one layout_id. Needs a cell_count column
        (no-op otherwise); tagged tables (non-null table_id) are left untouched.
        See _merge_untagged_table_lines.
    max_single_cell_bridge : int
        Max consecutive 1-cell lines allowed to bridge two multi-cell table
        segments (default 2). Only relevant when merge_untagged_tables=True.
    max_table_row_gap : float
        A vertical gap (pt) larger than this between adjacent lines breaks the
        table run — likely a separate table (default 30). merge_untagged_tables.
    table_bridge_font_tol : float
        A bridged 1-cell line must be within this many pt of the run's table
        font size to be pulled in (default 2). merge_untagged_tables.
    line_level : bool
        Set True when df already has exactly one row per line_id (e.g. the PDF
        line builder), skipping the _to_line_df aggregation step. Leave False
        for word/cell-level input (e.g. OCR) where multiple rows share a line_id.

    Returns
    -------
    df with added columns (joined by line_id):
        layout_id           int    1-based, monotonically increasing
        line_gap            float  vertical gap above this line (pt)
        median_gap          float  per-page median of positive line gaps
        page_gap_thresh     float  adaptive gap threshold used for this page
    """
    if df.empty:
        return df.assign(
            layout_id=0,
            line_gap=np.nan,
            median_gap=0.0,
            page_gap_thresh=0.0,
        )

    # ── Step 1: one row per line ─────────────────────────────────────────────
    if line_level:
        if df["line_id"].duplicated().any():
            raise ValueError("line_level=True requires a unique line_id per row")
        line_df = df.copy()
    else:
        line_df = _to_line_df(df)

    if "page_number" not in line_df.columns:
        raise ValueError("df must contain a 'page_number' column")

    # ── Step 2–5: per-page band assignment ───────────────────────────────────
    line_df["line_gap"]           = np.nan
    line_df["median_gap"]         = 0.0
    line_df["layout_id"]          = 0
    line_df["page_gap_thresh"]    = 0.0

    vert_mask = (
        line_df["text_orientation"].isin(["TTB", "BTT"])
        if "text_orientation" in line_df.columns
        else pd.Series(False, index=line_df.index)
    )

    band_counter = 0

    for _, page_df in line_df.groupby("page_number", sort=True):
        page_vert_mask = vert_mask.loc[page_df.index]
        vert_idx       = page_df.index[page_vert_mask].tolist()
        ltr_df         = page_df[~page_vert_mask].sort_values("line_id", kind="mergesort")

        if ltr_df.empty:
            median_gap = 0.0
            threshold  = float("inf")
        else:
            gaps_map, band_map, band_counter, median_gap, threshold = _process_page(
                ltr_df, band_counter, font_size_split_delta=font_size_split_delta
            )
            ltr_idx = ltr_df.index.tolist()
            line_df.loc[ltr_idx, "line_gap"]           = [gaps_map[lid] for lid in ltr_df["line_id"]]
            line_df.loc[ltr_idx, "median_gap"]         = median_gap
            line_df.loc[ltr_idx, "layout_id"]          = [band_map[lid] for lid in ltr_df["line_id"]]
            line_df.loc[ltr_idx, "page_gap_thresh"]    = threshold

        for idx in (line_df.loc[vert_idx].sort_values("line_id").index if vert_idx else []):
            band_counter += 1
            line_df.at[idx, "median_gap"]         = median_gap
            line_df.at[idx, "page_gap_thresh"]    = threshold
            line_df.at[idx, "layout_id"]          = band_counter

    # ── Step 6: optional merge by vertical lines ─────────────────────────────
    if merge_by_vertical_lines:
        line_df = _merge_bands_by_shared_vertical_lines(line_df, min_shared_lines=min_shared_lines)

    # ── Step 6b: pull untagged multi-cell table rows into one layout ─────────
    if merge_untagged_tables:
        line_df = _merge_untagged_table_lines(
            line_df,
            max_single_cell_bridge=max_single_cell_bridge,
            max_table_row_gap=max_table_row_gap,
            bridge_font_tol=table_bridge_font_tol,
        )

    # ── Step 7: join back onto input df ─────────────────────────────────────
    band_cols = line_df.set_index("line_id")[
        ["layout_id", "line_gap", "median_gap", "page_gap_thresh"]
    ]
    out = df.copy()
    for col in band_cols.columns:
        out[col] = df["line_id"].map(band_cols[col])

    return out
