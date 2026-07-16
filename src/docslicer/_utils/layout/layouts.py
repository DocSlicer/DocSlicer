"""
Horizontal band (layout) assignment — production rewrite.

Public API:
    df = assign_layouts(df)

Accepts any DataFrame that has a `line_id` column — words, cells, or lines.
When `line_level=False`, rows sharing a `line_id` are first aggregated to one
row per line (vectorized) before geometry is derived; results are joined back.

This module is a clean-room rewrite of ``layouts.py``.  It favours vectorized,
whole-frame operations over the per-page Python loop of the original.  Landed so
far: vectorized gaps (line_gap / median_gap / page_gap_thresh), the per-line
style-change flag, band assignment (layout_id) — the last computed as cumsum of
an adjacent-row "new band" flag, replacing the original per-page loop — and the
untagged-table line merge.  Still to port: layout-type classification.
"""

from __future__ import annotations

import operator
from dataclasses import dataclass

import numpy as np
import pandas as pd

from docslicer._utils.color_utils import (
    OCR_COLOR_DELTA_E_THRESHOLD,
    ciede2000_vec,
    colors_to_lab,
    ocr_colors_match,
)


# ============================================================
# CONFIG
# ============================================================

# Layout-type block_type shortcuts (see assign_layout_types): a layout carrying
# any of these tags is decided outright, without scoring.  Chart wins over
# everything (we do not actively assess charts); the text tags are structural
# roles that are never tabular.
_CHART_BLOCK_TYPES = frozenset({"chart"})
_TEXT_BLOCK_TYPES  = frozenset({
    "heading", "page_label", "vertical_text", "footnote", "block_quote",
})

@dataclass(frozen=True)
class LayoutConfig:
    """
    Tunable knobs for layout assignment.

    Frozen and hashable, so a pipeline can build its own instance and pass it to
    ``assign_layouts(df, config=...)`` without mutating the shared default.
    Construct variants with ``dataclasses.replace(DEFAULT_LAYOUT_CONFIG, ...)``.

    Gap threshold
    -------------
    min_page_gap_thresh
        Floor for the per-page band-gap threshold (pt).  The adaptive threshold
        is ``max(min_page_gap_thresh, gap_multiplier(median_gap) * median_gap)``.
    median_floor_gap
        Non-positive gaps (a page's first line, and column jumps) are counted as
        this many pt when the per-page median gap is taken.  It only floors the
        value fed to the median — the stored ``line_gap`` keeps its true 0 /
        negative value.

    Adaptive multiplier (see ``gap_multiplier``)
    --------------------------------------------
    Tighter line spacing -> higher multiplier (more sensitive splitting); looser
    spacing -> lower multiplier (avoids over-splitting).  Linear ramp:
        median_gap <= mult_low_gap   -> mult_at_low
        median_gap >= mult_high_gap  -> mult_at_high
        in between                   -> linear interpolation
    """
    min_page_gap_thresh: float = 3.5
    median_floor_gap:    float = 2.0

    mult_low_gap:  float = 3.0
    mult_high_gap: float = 10.0
    mult_at_low:   float = 1.60
    mult_at_high:  float = 1.10

    # ── Style-change split (see _style_change_flags) ─────────────────────────
    # A line is flagged when its visual style differs from the previous line on
    # at least ``style_min_changes`` of the ``style_attrs`` that are present.
    # Downstream, that flag is a split-only trigger.  Tune per pipeline:
    #   - PDF (trusted styling): all attrs, any single change (min_changes=1),
    #     font-size delta 1.0 pt — the original layouts.py behaviour.
    #   - OCR prelim pass: no style analysis at all (analyze_style=False) —
    #     bands are decided purely by the vertical-gap trigger.
    #   - OCR second pass: {font_size, is_bold, non_stroking_color}, but require
    #     2 of 3 to change (OCR jitter makes any single attr unreliable) and a
    #     looser 2.0 pt font-size delta.
    # ``font_size`` uses the |Δ| >= delta rule and counts as one attribute; every
    # other attr uses exact-change equality.  A missing (NaN) cell on either
    # side never counts as a change.
    #
    # ``analyze_style`` is the master switch: when False the style-change split is
    # skipped entirely (every line's ``style_change`` is False) and bands fall to
    # the gap trigger alone — the ``style_*`` / colour knobs below are then unused.
    analyze_style:         bool  = True
    style_font_size_delta: float = 1.0
    style_min_changes:     int   = 1
    style_attrs: tuple[str, ...] = (
        "font_family", "font_size", "is_bold", "is_italic", "non_stroking_color",
    )

    # ── Colour comparison for the style-change split ─────────────────────────
    # PDF colours are exact, so ``non_stroking_color`` is compared by exact
    # equality (ocr_color_match=False).  OCR colours jitter even after snapping
    # (#202020 vs #000000 is the *same* ink to a human), so the OCR pipeline sets
    # ocr_color_match=True to compare perceptually: two colours count as changed
    # only when their CIEDE2000 ΔE exceeds ``color_delta_e_threshold``.  This
    # only affects the non_stroking_color comparison; all other attrs are
    # unchanged.  See _style_change_flags / color_utils.ciede2000_vec.
    ocr_color_match:         bool  = False
    color_delta_e_threshold: float = OCR_COLOR_DELTA_E_THRESHOLD

    # ── Untagged-table line merge (see _merge_untagged_table_lines) ──────────
    # After banding, pull consecutive multi-cell lines (cell_count >= 2) that
    # form an *untagged* table (no struct table_id) into one layout_id.  Up to
    # ``max_single_cell_bridge`` one-cell lines may bridge two multi-cell
    # segments, subject to the gap / font / colour guards below.  No-op unless a
    # cell_count column is present; tagged tables (non-null table_id) untouched.
    merge_untagged_tables:      bool  = True
    max_single_cell_bridge:     int   = 2      # consecutive 1-cell lines that may bridge
    max_table_row_gap:          float = 20.0   # pt; a larger gap breaks the run
    max_single_cell_bridge_gap: float = 15.0   # pt; tighter cap around a bridged 1-cell line
    table_bridge_font_tol:      float = 2.0    # pt; bridged line must be within this of table font

    # ── Table-caption ejection (see _eject_table_captions) ───────────────────
    # After banding, split a table layout's *leading* caption lines out into their
    # own layout.  A leading line qualifies when it is single-cell (cell_count
    # <= 1), carries no table_id and no table_grid_id, and is left-aligned within
    # ``caption_max_left_offset`` pt of the table body's leftmost multi-cell edge.
    # The whole df's layout_id is then reindexed (dense, order-preserving).  No-op
    # unless cell_count / x_left are present.
    eject_table_captions:    bool  = True
    caption_max_left_offset: float = 100.0     # pt; leading line within xmin+this is a caption

    # ── Layout-type classification (see assign_layout_types) ─────────────────
    # After banding, classify every layout as 'text' / 'table' / 'chart' and add
    # layout_type + layout_score columns.  ``chart_block_types`` /
    # ``text_block_types`` short-circuit a layout to that verdict without scoring;
    # a scored layout is a table when its summed score exceeds
    # ``table_score_threshold``.  The graded weights themselves are policy-free
    # data in ``_LAYOUT_SCORE_BANDS`` (tune there — no control-flow change).
    classify_types:        bool = True
    chart_block_types:     frozenset[str] = _CHART_BLOCK_TYPES
    text_block_types:      frozenset[str] = _TEXT_BLOCK_TYPES
    table_score_threshold: float = 1.0

    def gap_multiplier(self, median_gap: np.ndarray | pd.Series) -> np.ndarray:
        """
        Vectorized adaptive multiplier for the band-gap threshold — the scalar
        ``_interpolate_gap_multiplier`` of the original, applied elementwise.
        """
        median_gap = np.asarray(median_gap, dtype=float)
        span = self.mult_high_gap - self.mult_low_gap
        t = np.clip((median_gap - self.mult_low_gap) / span, 0.0, 1.0)
        return self.mult_at_low + t * (self.mult_at_high - self.mult_at_low)

    # ── Per-pipeline presets ─────────────────────────────────────────────────
    # These bundle the style/colour knobs for each caller so a pipeline picks one
    # instead of hand-tuning several fields (and getting a combination wrong).
    # They are the "which pipeline is calling" signal; override further with
    # dataclasses.replace(...) if a specific document needs it.

    @classmethod
    def for_pdf(cls) -> "LayoutConfig":
        """Trusted PDF styling: all attrs, any single change, exact colours."""
        return cls()

    @classmethod
    def for_ocr_prelim(cls) -> "LayoutConfig":
        """
        OCR first pass — style (and colour) is too unreliable to trust at all, so
        skip style analysis entirely and let bands fall out of the vertical-gap
        trigger alone.  Layout-type classification and table-caption ejection are
        skipped too: the prelim pass only needs the gap-based banding.
        """
        return cls(
            analyze_style=False,
            classify_types=False,
            eject_table_captions=False,
        )

    @classmethod
    def for_ocr_second_pass(cls) -> "LayoutConfig":
        """
        OCR second pass — font_size is now available.  Require 2 of
        {font_size, is_bold, non_stroking_color} to change (any single one is
        still jitter-prone), a looser 2.0 pt font-size delta, perceptual colour.
        """
        return cls(
            style_attrs=("font_size", "is_bold", "non_stroking_color"),
            style_min_changes=2,
            style_font_size_delta=2.0,
            ocr_color_match=True,
        )


DEFAULT_LAYOUT_CONFIG = LayoutConfig()


# ============================================================
# HELPERS
# ============================================================


def _to_line_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate word/cell-level rows to one row per line_id (vectorized).

    Per line_id:
        y_top    -> min   (topmost edge of the line)
        y_bottom -> max   (bottommost edge)
        x_left   -> min   (line start), when present
        font_size-> median (robust to a stray glyph), when present
        everything else that is constant within a line -> first value

    Reading order is preserved: line_id already encodes it, so the aggregated
    frame is sorted downstream by (page_number, line_id).
    """
    agg: dict[str, tuple[str, str]] = {
        "y_top":    ("y_top",    "min"),
        "y_bottom": ("y_bottom", "max"),
    }
    if "x_left" in df.columns:
        agg["x_left"] = ("x_left", "min")
    if "font_size" in df.columns:
        agg["font_size"] = ("font_size", "median")
    for col in (
        "page_number", "block_type", "text_orientation",
        "table_id", "table_grid_id", "struct_ancestors", "struct_ancestor_ids",
        "font_family", "is_bold", "is_italic", "non_stroking_color",
        "cell_count", "line_score", "shape_id_tr_above", "shape_id_tr_below",
    ):
        if col in df.columns:
            agg[col] = (col, "first")

    return df.groupby("line_id", sort=False).agg(**agg).reset_index()


# ============================================================
# STEP 1: per-line style-change flag (vectorized)
# ============================================================

def _norm_key(series: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """
    Comparable key + present-mask for one style column (vectorized single pass).

    Returns ``(keys, present)`` object/bool arrays aligned to ``series``:
        keys     — sequence cells (list/tuple/ndarray colors) normalized to
                   tuples so plain ``!=`` compares them elementwise; scalars are
                   left as-is.  Value at a missing cell is irrelevant.
        present  — True where the cell is a real value (missing / NaN -> False).

    Sequences (RGB colors) are never treated as NA; only scalar NaN/None is.
    """
    arr = np.array(series.to_numpy(dtype=object), dtype=object)   # writable copy
    n = len(arr)
    present = np.ones(n, dtype=bool)
    for i in range(n):
        v = arr[i]
        if isinstance(v, (list, tuple, np.ndarray)):
            arr[i] = tuple(v)            # hashable, elementwise-comparable
        elif v is None:
            present[i] = False
        else:
            try:
                if pd.isna(v):
                    present[i] = False
            except (TypeError, ValueError):
                pass                     # unusual scalar — treat as present
    return arr, present


def _style_change_flags(
    df: pd.DataFrame,
    config: LayoutConfig = DEFAULT_LAYOUT_CONFIG,
) -> np.ndarray:
    """
    Per line, flag when the visual style differs enough from the previous line.

    ``df`` must already be in reading order (sorted by page_number, line_id).
    Comparison never crosses a page boundary — the first line of every page is
    False.  For each attribute in ``config.style_attrs`` that is present, a
    per-line "changed vs previous line" boolean is computed:

        font_size   -> |Δ| >= config.style_font_size_delta  (both cells present)
        colour¹     -> value changed                        (both cells present)
        other attrs -> value changed                        (both cells present)

        ¹ non_stroking_color: exact equality by default, but when
          config.ocr_color_match is set it is a perceptual compare — changed
          only when CIEDE2000 ΔE > config.color_delta_e_threshold, so OCR scan
          jitter (#202020 vs #000000) is not treated as a change.

    The per-attribute booleans are summed into a change count, and a line is
    flagged when that count >= ``config.style_min_changes``.  A missing (NaN)
    cell on either side never counts as a change, so absent columns and OCR gaps
    simply contribute 0.  Split-only signal — the caller uses it to force a new
    band but never to merge.

    Returns a bool array aligned to ``df``'s row order.
    """
    n = len(df)
    if n < 2 or config.style_min_changes < 1:
        return np.zeros(n, dtype=bool)

    attrs = [a for a in config.style_attrs if a in df.columns]
    if not attrs:
        return np.zeros(n, dtype=bool)

    # Page-boundary mask: no comparison across the first line of each page.
    page         = df["page_number"].to_numpy()
    is_page_start = np.empty(n, dtype=bool)
    is_page_start[0]  = True
    is_page_start[1:] = page[1:] != page[:-1]
    cross_ok = ~is_page_start                     # True where prev line is same page

    change_count = np.zeros(n, dtype=int)

    for attr in attrs:
        if attr == "font_size":
            fs   = df["font_size"].to_numpy(dtype=float)
            prev = np.empty(n, dtype=float)
            prev[0], prev[1:] = np.nan, fs[:-1]
            both    = ~np.isnan(fs) & ~np.isnan(prev) & cross_ok
            changed = both & (np.abs(fs - prev) >= config.style_font_size_delta - 1e-9)
        elif attr == "non_stroking_color" and config.ocr_color_match:
            # Perceptual colour compare: convert each line's colour to CIELAB
            # once, then ΔE against the previous line.  A change counts only when
            # ΔE exceeds the threshold, so scan jitter (#202020 vs #000000) is
            # not a split.  Both cells must be present (missing never fires).
            lab, present     = colors_to_lab(df[attr].to_numpy(dtype=object))
            prev_lab         = np.full_like(lab, np.nan)
            prev_lab[1:]     = lab[:-1]
            prev_present     = np.zeros(n, dtype=bool)
            prev_present[1:] = present[:-1]
            both    = present & prev_present & cross_ok
            delta   = ciede2000_vec(lab, prev_lab)
            changed = both & (delta > config.color_delta_e_threshold)
        else:
            keys, present = _norm_key(df[attr])
            prev_keys        = np.empty(n, dtype=object)
            prev_keys[0]     = None
            prev_keys[1:]    = keys[:-1]
            prev_present     = np.zeros(n, dtype=bool)
            prev_present[1:] = present[:-1]
            both    = present & prev_present & cross_ok
            neq     = np.asarray(keys != prev_keys, dtype=bool)
            changed = both & neq

        change_count += changed.astype(int)

    return change_count >= config.style_min_changes


# ============================================================
# STEP 2: vertical gap + adaptive per-page threshold (vectorized)
# ============================================================

def _assign_gaps(
    line_df: pd.DataFrame,
    config: LayoutConfig = DEFAULT_LAYOUT_CONFIG,
) -> pd.DataFrame:
    """
    Add ``line_gap``, ``median_gap`` and ``page_gap_thresh`` to a line-level df,
    fully vectorized (no per-page Python loop).

    Definitions (per page, lines in reading order = sorted by line_id, vertical
    TTB/BTT lines excluded):

        line_gap        y_top[i] - y_bottom[i-1]; 0.0 for the first line of a
                        page.  Negative for a column jump (reading order moved
                        up to a new column).
        median_gap      per-page median of the gaps, where every non-positive
                        gap (first line + column jumps) is floored to
                        config.median_floor_gap before the median is taken.
        page_gap_thresh max(config.min_page_gap_thresh,
                        config.gap_multiplier(median_gap) * median_gap) — one
                        value per page.

    Vertical (TTB/BTT) lines are not part of any page's gap statistics: their
    line_gap stays NaN, but they still receive their page's median_gap /
    page_gap_thresh so downstream code has a value.  Columns are returned in the
    original row order of ``line_df``.
    """
    out = line_df.copy()
    out["line_gap"]        = np.nan
    out["median_gap"]      = 0.0
    out["page_gap_thresh"] = 0.0

    if out.empty:
        return out

    if "text_orientation" in out.columns:
        vert_mask = out["text_orientation"].isin(["TTB", "BTT"]).to_numpy()
    else:
        vert_mask = np.zeros(len(out), dtype=bool)

    ltr = out.loc[~vert_mask].sort_values(
        ["page_number", "line_id"], kind="mergesort"
    )

    if not ltr.empty:
        page = ltr["page_number"]
        prev_yb = ltr.groupby(page, sort=False)["y_bottom"].shift()
        gap = (ltr["y_top"] - prev_yb).fillna(0.0)   # first line of page -> 0

        # Median uses floored gaps: non-positive (first line / column jump) -> floor.
        floored = gap.where(gap > 0, config.median_floor_gap)
        median_gap = floored.groupby(page, sort=False).transform("median")

        thresh = np.maximum(
            config.min_page_gap_thresh, config.gap_multiplier(median_gap) * median_gap
        )

        out.loc[ltr.index, "line_gap"]        = gap
        out.loc[ltr.index, "median_gap"]      = median_gap
        out.loc[ltr.index, "page_gap_thresh"] = thresh

    # Broadcast each page's median_gap / threshold onto its vertical lines too.
    if vert_mask.any():
        page_stats = (
            out.loc[~vert_mask, ["page_number", "median_gap", "page_gap_thresh"]]
            .drop_duplicates("page_number")
            .set_index("page_number")
        )
        vert_pages = out.loc[vert_mask, "page_number"]
        out.loc[vert_mask, "median_gap"] = (
            vert_pages.map(page_stats["median_gap"]).fillna(0.0).to_numpy()
        )
        out.loc[vert_mask, "page_gap_thresh"] = (
            vert_pages.map(page_stats["page_gap_thresh"]).fillna(0.0).to_numpy()
        )

    return out


def _assign_style_change(
    line_df: pd.DataFrame,
    config: LayoutConfig = DEFAULT_LAYOUT_CONFIG,
) -> pd.DataFrame:
    """
    Add a ``style_change`` bool column to a line-level df.

    Lines are compared in reading order per page (sorted by page_number,
    line_id); vertical TTB/BTT lines are excluded from the comparison and left
    False (they always open their own band anyway).  See ``_style_change_flags``
    for the per-line rule.  Returned in the original row order of ``line_df``.

    When ``config.analyze_style`` is False the split is disabled: every line's
    ``style_change`` stays False and bands fall to the gap trigger alone.
    """
    out = line_df.copy()
    out["style_change"] = False
    if out.empty or not config.analyze_style:
        return out

    if "text_orientation" in out.columns:
        vert_mask = out["text_orientation"].isin(["TTB", "BTT"]).to_numpy()
    else:
        vert_mask = np.zeros(len(out), dtype=bool)

    ltr = out.loc[~vert_mask].sort_values(
        ["page_number", "line_id"], kind="mergesort"
    )
    if not ltr.empty:
        out.loc[ltr.index, "style_change"] = _style_change_flags(ltr, config)

    return out


# ============================================================
# STEP 3: band assignment (layout_id) — vectorized
# ============================================================
#
# The original walked each page line-by-line, carrying prev_* references and a
# running band counter.  Every decision it made for line i, though, is a
# function of only line i and line i-1 (the "previous line" references) plus a
# "segment start" flag — so the whole thing is:
#
#     new_band[i] = segment_start[i]
#                   OR forced_new[i]
#                   OR (not forced_same[i] AND gap_trigger[i])
#     layout_id   = cumsum(new_band)
#
# computed from adjacent rows via shift(), with no Python loop over lines.
# See assign_layouts' docstring for the struct/gap rules this encodes.

_HEADING_TAGS = frozenset({"H", "H1", "H2", "H3", "H4", "H5", "H6"})
_LIST_TAG     = "L"
_PARA_TAG     = "P"


def _deepest_struct_ids(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Per row, the deepest (last-occurring) heading / list / paragraph ancestor id
    from struct_ancestors / struct_ancestor_ids.  Ancestors run root -> leaf, so
    the last matching tag is the innermost.  Arrays of None when the struct
    columns are absent.  (Port of the original ``_struct_group_ids``.)
    """
    n = len(df)
    heading = np.empty(n, dtype=object); heading[:] = None
    lists   = np.empty(n, dtype=object); lists[:]   = None
    paras   = np.empty(n, dtype=object); paras[:]   = None

    if "struct_ancestors" not in df.columns or "struct_ancestor_ids" not in df.columns:
        return heading, lists, paras

    anc_arr = df["struct_ancestors"].to_numpy(dtype=object)
    aid_arr = df["struct_ancestor_ids"].to_numpy(dtype=object)
    for i in range(n):
        ancs = anc_arr[i] if isinstance(anc_arr[i], (list, tuple)) else []
        aids = aid_arr[i] if isinstance(aid_arr[i], (list, tuple)) else []
        for tag, id_ in zip(ancs, aids):
            if tag in _HEADING_TAGS:
                heading[i] = id_
            elif tag == _LIST_TAG:
                lists[i] = id_
            elif tag == _PARA_TAG:
                paras[i] = id_
    return heading, lists, paras


def _shift_prev(arr: np.ndarray, is_page_first: np.ndarray, fill: object) -> np.ndarray:
    """Previous-row value within a page: arr shifted down 1, ``fill`` at each
    page's first row (so no comparison crosses a page boundary)."""
    prev = np.empty(len(arr), dtype=object)
    prev[0]  = fill
    prev[1:] = arr[:-1]
    prev[is_page_first] = fill
    return prev


def _eq_present(cur: np.ndarray, prev: np.ndarray) -> np.ndarray:
    """
    Element-wise equality as a plain bool array, comparing only cells that are
    present on both sides (a missing value on either side -> False).

    ``cur == prev`` on object arrays holding pandas ``pd.NA`` yields ``pd.NA``
    cells, which ``np.asarray(..., dtype=bool)`` cannot cast ("boolean value of
    NA is ambiguous").  Masking to present-on-both cells before the compare keeps
    NA out of the cast entirely (np.nan happened to compare as False, but pd.NA
    from a nullable dtype does not).
    """
    both = ~pd.isna(cur) & ~pd.isna(prev)
    out = np.zeros(len(cur), dtype=bool)
    if both.any():
        out[both] = np.asarray(cur[both] == prev[both], dtype=bool)
    return out


def _band_new_flags(
    df: pd.DataFrame,
    config: LayoutConfig = DEFAULT_LAYOUT_CONFIG,
) -> np.ndarray:
    """
    Boolean "line opens a new band" per row, for an LTR-only frame already sorted
    by (page_number, line_id).  See the module comment above for the formula; the
    struct priority cascade and grid override mirror the original _process_page
    exactly.  Returns a bool array aligned to ``df``'s row order.
    """
    n = len(df)
    if n == 0:
        return np.zeros(0, dtype=bool)

    page = df["page_number"].to_numpy()
    is_page_first = np.empty(n, dtype=bool)
    is_page_first[0]  = True
    is_page_first[1:] = page[1:] != page[:-1]

    # ── block_type (normalized like the original: fillna -> str -> lower) ────
    if "block_type" in df.columns:
        bt = df["block_type"].fillna("").astype(str).str.lower().to_numpy()
    else:
        bt = np.full(n, "", dtype=object)
    prev_bt = _shift_prev(bt, is_page_first, fill="")

    is_page_label      = bt == "page_label"
    prev_is_page_label = _shift_prev(is_page_label, is_page_first, fill=False).astype(bool)

    # ── struct group ids (all None when the columns are absent) ──────────────
    def _col(name: str) -> np.ndarray:
        return (df[name].to_numpy(dtype=object)
                if name in df.columns else np.full(n, None, dtype=object))

    table = _col("table_id")
    grid  = _col("table_grid_id")
    heading, lists, paras = _deepest_struct_ids(df)

    prev_table   = _shift_prev(table,   is_page_first, None)
    prev_grid    = _shift_prev(grid,    is_page_first, None)
    prev_heading = _shift_prev(heading, is_page_first, None)
    prev_list    = _shift_prev(lists,   is_page_first, None)
    prev_para    = _shift_prev(paras,   is_page_first, None)

    # ── struct priority cascade (first match wins): block_type change, then
    #    table / heading / list / para.  `decided` gates lower-priority levels. ─
    bt_change   = bt != prev_bt
    forced_new  = bt_change.copy()
    forced_same = np.zeros(n, dtype=bool)
    decided     = bt_change.copy()

    for cur, prev in ((table, prev_table), (heading, prev_heading),
                      (lists, prev_list), (paras, prev_para)):
        cur_present  = ~pd.isna(cur)
        involved     = (cur_present | ~pd.isna(prev)) & ~decided
        same_here    = involved & cur_present & _eq_present(cur, prev)
        forced_same |= same_here
        forced_new  |= involved & ~same_here
        decided     |= involved

    # table_grid_id: merge-only override — a shared grid id (block_type
    # unchanged) forces a merge regardless of what the cascade decided; it never
    # forces a split on its own.
    grid_merge = (
        (bt == prev_bt)
        & ~pd.isna(grid) & ~pd.isna(prev_grid)
        & _eq_present(grid, prev_grid)
    )
    forced_same = forced_same | grid_merge
    forced_new  = forced_new & ~grid_merge

    # ── gap-based trigger ────────────────────────────────────────────────────
    gap = df["line_gap"].to_numpy(dtype=float)
    thr = df["page_gap_thresh"].to_numpy(dtype=float)
    style_changed = (
        df["style_change"].to_numpy(dtype=bool)
        if "style_change" in df.columns else np.zeros(n, dtype=bool)
    )
    gap_trigger = (gap > thr) | (gap < -thr) | style_changed

    # A segment start always opens a band (and overrides forced_same); otherwise
    # struct forcing wins, then the gap trigger.
    segment_start = is_page_first | is_page_label | prev_is_page_label
    return segment_start | forced_new | (~forced_same & gap_trigger)


def _assign_bands(
    line_df: pd.DataFrame,
    config: LayoutConfig = DEFAULT_LAYOUT_CONFIG,
) -> pd.DataFrame:
    """
    Add a 1-based ``layout_id`` column to a line-level df.

    LTR lines are banded by ``_band_new_flags`` + cumsum in (page, line_id)
    order; each vertical (TTB/BTT) line is its own singleton band, numbered after
    the LTR bands.  Returned in the original row order of ``line_df``.
    """
    out = line_df.copy()
    out["layout_id"] = 0
    if out.empty:
        return out

    if "text_orientation" in out.columns:
        vert_mask = out["text_orientation"].isin(["TTB", "BTT"]).to_numpy()
    else:
        vert_mask = np.zeros(len(out), dtype=bool)

    ltr = out.loc[~vert_mask].sort_values(["page_number", "line_id"], kind="mergesort")
    max_band = 0
    if not ltr.empty:
        new_band = _band_new_flags(ltr, config)
        layout_id = np.cumsum(new_band.astype(np.int64))   # 1-based (first is True)
        out.loc[ltr.index, "layout_id"] = layout_id
        max_band = int(layout_id[-1]) if len(layout_id) else 0

    # Each vertical line: its own band, in reading order, after the LTR bands.
    if vert_mask.any():
        vert = out.loc[vert_mask].sort_values(["page_number", "line_id"], kind="mergesort")
        out.loc[vert.index, "layout_id"] = np.arange(
            max_band + 1, max_band + 1 + len(vert), dtype=np.int64
        )

    return out


# ============================================================
# STEP 4: pull untagged table rows into one layout (per-page loop)
# ============================================================

def _is_na_scalar(v: object) -> bool:
    """True for a missing scalar; sequences (e.g. an RGB colour tuple) are never NA."""
    if isinstance(v, (list, tuple, np.ndarray, dict)):
        return False
    try:
        return bool(pd.isna(v))
    except (TypeError, ValueError):
        return False


def _colors_present_equal(a: object, b: object, config: LayoutConfig) -> bool:
    """
    Both colours present *and* equal.  Exact tuple equality by default; when
    ``config.ocr_color_match`` is set, a perceptual (CIEDE2000) match so OCR scan
    jitter does not break a table bridge.  A missing colour on either side -> False.
    """
    if _is_na_scalar(a) or _is_na_scalar(b):
        return False
    if config.ocr_color_match:
        return ocr_colors_match(a, b, config.color_delta_e_threshold)
    an = tuple(a) if isinstance(a, (list, tuple, np.ndarray)) else a
    bn = tuple(b) if isinstance(b, (list, tuple, np.ndarray)) else b
    return bool(an == bn)


def _merge_untagged_table_lines(
    line_df: pd.DataFrame,
    config: LayoutConfig = DEFAULT_LAYOUT_CONFIG,
) -> pd.DataFrame:
    """
    Merge the layout_id of consecutive multi-cell lines that form an *untagged*
    table (no struct table_id) so the whole table reads as one layout.

    Per page, walking lines in reading order (line_id):
        - a line with cell_count >= 2 opens/extends a table run;
        - up to ``config.max_single_cell_bridge`` consecutive 1-cell lines may
          bridge two multi-cell segments, but only when another multi-cell line
          follows, each bridging line's font_size is within
          ``config.table_bridge_font_tol`` pt of the run's last multi-cell line,
          its non_stroking_color matches (see _colors_present_equal — exact, or
          perceptual under ocr_color_match), and every gap around the bridged
          line(s) is <= ``config.max_single_cell_bridge_gap`` pt;
        - a vertical gap (y_top - prev y_bottom) larger than
          ``config.max_table_row_gap`` pt breaks the run (likely a separate table);
        - a *decreasing* y_top from one line to the next breaks the run and is
          never bridged across: reading order only moves back up the page when
          it jumps to a new column, so the lines straddle a column boundary.
    Every line in a run of >= 2 lines is relabelled to the run's smallest
    layout_id.  Rows already inside a tagged table (non-null table_id) are hard
    run boundaries and never merged.  A block_type change from one line to the
    next also breaks the run and is never bridged across (mirrors the upstream
    "block_type change -> always splits" rule).

    No-op when cell_count is absent.  Faithful port of the original; not
    vectorized (the run/bridge walk is inherently sequential).
    """
    if "cell_count" not in line_df.columns:
        return line_df

    max_single_cell_bridge     = config.max_single_cell_bridge
    max_table_row_gap          = config.max_table_row_gap
    max_single_cell_bridge_gap = config.max_single_cell_bridge_gap
    bridge_font_tol            = config.table_bridge_font_tol

    has_table = "table_id"  in line_df.columns
    has_font  = "font_size" in line_df.columns
    has_color = "non_stroking_color" in line_df.columns
    has_bt    = "block_type" in line_df.columns

    updates: dict[object, int] = {}   # original index label -> new layout_id

    for _, page_df in line_df.groupby("page_number", sort=False):
        pdf   = page_df.sort_values("line_id", kind="mergesort")
        idx   = pdf.index.to_numpy()
        cc    = pdf["cell_count"].fillna(0).to_numpy()
        ytop  = pdf["y_top"].to_numpy(dtype=float)
        ybot  = pdf["y_bottom"].to_numpy(dtype=float)
        lay   = pdf["layout_id"].to_numpy()
        fs    = pdf["font_size"].to_numpy(dtype=float) if has_font  else np.full(len(pdf), np.nan)
        nsc   = pdf["non_stroking_color"].to_numpy(dtype=object) if has_color else np.full(len(pdf), None, dtype=object)
        tagged = pdf["table_id"].notna().to_numpy()    if has_table else np.zeros(len(pdf), dtype=bool)
        # Normalized like _band_new_flags so the block_type comparison matches the
        # upstream split rule (fillna -> str -> lower); "" when the column is absent.
        bt    = (
            pdf["block_type"].fillna("").astype(str).str.lower().to_numpy()
            if has_bt else np.full(len(pdf), "", dtype=object)
        )
        n = len(pdf)

        def gap(m: int) -> float:                       # gap above local line m (m >= 1)
            return float(ytop[m] - ybot[m - 1])

        def col_shift(m: int) -> bool:                  # y_top moved back up => new column
            return bool(ytop[m] < ytop[m - 1])

        def is_row(m: int) -> bool:                     # eligible multi-cell (untagged) table row
            return (cc[m] >= 2) and not tagged[m]

        i = 0
        while i < n:
            if not is_row(i):
                i += 1
                continue

            run        = [i]
            last_multi = i
            run_bt     = bt[i]                           # every run line must share this block_type
            j          = i + 1
            while j < n:
                if tagged[j] or col_shift(j) or gap(j) > max_table_row_gap:
                    break
                # A block_type change ends the run — never merge across it.
                if has_bt and bt[j] != run_bt:
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
                    gaps_ok = all(
                        gap(m) <= max_single_cell_bridge_gap and not col_shift(m)
                        for m in range(j, k + 1)
                    )
                    # No block_type change anywhere in the bridged span (j..k).
                    bt_ok = (not has_bt) or all(bt[m] == run_bt for m in range(j, k + 1))
                    font_ok = (not has_font) or all(
                        (not np.isnan(fs[m]) and not np.isnan(fs[last_multi])
                         and abs(fs[m] - fs[last_multi]) <= bridge_font_tol)
                        for m in range(j, k)
                    )
                    color_ok = (not has_color) or all(
                        _colors_present_equal(nsc[m], nsc[last_multi], config)
                        for m in range(j, k)
                    )
                    if gaps_ok and bt_ok and font_ok and color_ok:
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
# STEP 5: eject leading table captions into their own layout
# ============================================================

def _eject_table_captions(
    line_df: pd.DataFrame,
    config: LayoutConfig = DEFAULT_LAYOUT_CONFIG,
) -> tuple[pd.DataFrame, bool]:
    """
    Split a table layout's leading *caption* lines out into their own layout.

    A caption often shares the table's styling and sits so close to the first
    grid row that the gap/style band pass can't separate it — so it lands inside
    the table's ``layout_id``.  Here, per layout (identified structurally as a
    *table* layout: one that contains at least one multi-cell line), we walk the
    lines in reading order and peel off the *leading run* that is:

        - single-cell        (cell_count <= 1),
        - untagged           (no table_id **and** no table_grid_id), and
        - left-aligned       (x_left <= xmin + config.caption_max_left_offset,
                              where xmin is the leftmost x_left among the layout's
                              multi-cell lines).

    The run stops at the first line that fails any of these (so a table's trailing
    footnotes — which carry the grid's table_grid_id in the example — and its grid
    rows are never touched, and a non-left-aligned leading line is left in place).
    Only leading lines are considered; nothing after the first multi-cell row is
    ejected.

    When at least one line is ejected, every line's ``layout_id`` is reindexed to
    a dense, order-preserving sequence: each ejected caption run becomes a band
    immediately before its table (its lines keep their reading-order position, so
    lexicographic ``(layout_id, is_body)`` ranking places the caption one id ahead
    of the body).  Returns ``(df, changed)``; ``changed`` is False (and the frame
    is returned untouched) when nothing qualifies or the needed columns are
    absent.

    Runs before classification so the freshly-split caption layout is scored on
    its own — a single-cell layout falls to ``layout_type == 'text'`` naturally,
    with no stale table ``block_type`` to clean up.
    """
    needed = {"layout_id", "cell_count", "line_id", "page_number", "x_left"}
    if line_df.empty or not needed.issubset(line_df.columns):
        return line_df, False

    n     = len(line_df)
    cc    = line_df["cell_count"].to_numpy(dtype=float)
    xleft = line_df["x_left"].to_numpy(dtype=float)
    lay   = line_df["layout_id"].to_numpy()
    lid   = line_df["line_id"].to_numpy()

    is_multi  = cc > 1     # NaN -> False
    is_single = cc <= 1    # NaN -> False (a NaN-count line breaks the leading run)

    def _blank(col: str) -> np.ndarray:
        # A missing column means "no such membership anywhere" -> all blank.
        return (line_df[col].isna().to_numpy() if col in line_df.columns
                else np.ones(n, dtype=bool))
    blank_table = _blank("table_id")
    blank_grid  = _blank("table_grid_id")

    tol      = config.caption_max_left_offset
    eject    = np.zeros(n, dtype=bool)
    all_pos  = np.arange(n)

    for lay_id in pd.unique(lay):
        pos = all_pos[lay == lay_id]
        pos = pos[np.argsort(lid[pos], kind="mergesort")]   # reading order
        multi = is_multi[pos]
        if not multi.any():
            continue                                          # not a table layout
        xmin = np.nanmin(xleft[pos[multi]])
        for p in pos:
            # NaN x_left -> comparison False -> run ends (caption must be aligned).
            if (is_single[p] and blank_table[p] and blank_grid[p]
                    and xleft[p] <= xmin + tol):
                eject[p] = True
            else:
                break

    if not eject.any():
        return line_df, False

    # Reindex the whole frame: an ejected caption run sorts just before its table
    # body via the (layout_id, is_body) key; dense-ranking those keys renumbers
    # every layout 1..K in the existing order while inserting the caption bands
    # (and, as a side effect, compacts any gaps left by the untagged-table merge).
    is_body = (~eject).astype(int)
    pairs   = list(zip(lay.tolist(), is_body.tolist()))
    mapping = {key: i + 1 for i, key in enumerate(sorted(set(pairs)))}
    new_layout = np.fromiter((mapping[p] for p in pairs), dtype=np.int64, count=n)

    out = line_df.copy()
    out["layout_id"] = new_layout
    return out, True


# ============================================================
# STEP 6: layout-type classification (text / table / chart per layout)
# ============================================================

# ── Table/text scoring: weights as data ──────────────────────────────────────
# A scored layout accumulates a single signed score — negative pulls toward
# text, positive toward table.  The per-feature weights live here as ordered band
# tables so tuning only ever edits this dict, never the control flow in
# assign_layout_types.  Each rule is (comparator, threshold, points); the rules
# for a feature are tried in order and the FIRST match per layout wins (exactly
# the np.select cascade this replaces), so a value matching no rule — including
# NaN, whose comparisons are all False — contributes 0.  Only the graded numeric
# evidence lives here; the block_type / table_id shortcuts and the final
# table-vs-text threshold are policy and live in LayoutConfig.
_COMPARATORS = {
    ">":  operator.gt, ">=": operator.ge,
    "<":  operator.lt, "<=": operator.le,
    "==": operator.eq,
}

_LAYOUT_SCORE_BANDS: dict[str, tuple[tuple[str, float, float], ...]] = {
    # median cell_count: more cells per line => more table-like.
    "median_cc": (
        (">", 10.0, 10.0), (">", 5.0, 5.0), (">", 2.0, 1.0),
        ("<=", 1.0, -5.0), ("<=", 2.0, -1.0),
    ),
    # distinct bounding rules (shape_id_tr_above/below): dense ruling => table.
    "n_shapes": (
        (">", 15.0, 20.0), (">", 10.0, 10.0), (">", 4.0, 5.0),
        (">", 2.0, 1.0), ("==", 0.0, -2.0),           # 1-2 shapes => 0 (no rule)
    ),
    # median per-line line_score (step-10 word-gap prior): +ve tabular, -ve prose.
    "median_ls": (
        ("<", -8.0, -10.0), ("<", -5.0, -5.0), ("<", -2.0, -2.0), ("<", 0.0, -2.0),
        (">", 8.0, 10.0), (">", 5.0, 5.0), (">", 2.0, 2.0),
    ),
    # line count: a couple of lines is rarely a table; many lines lean table.
    "n_lines": (
        ("==", 1.0, -10.0), ("<", 4.0, -1.0), (">", 10.0, 2.0), (">=", 4.0, 1.0),
    ),
}

# Fail fast on a fat-fingered comparator at import, rather than mis-scoring
# silently at run time (the tables are meant to be edited by hand).
assert all(
    op in _COMPARATORS
    for rules in _LAYOUT_SCORE_BANDS.values()
    for op, _, _ in rules
), "_LAYOUT_SCORE_BANDS: unknown comparator"


def _apply_score_bands(
    values: np.ndarray,
    rules: tuple[tuple[str, float, float], ...],
) -> np.ndarray:
    """
    First-match-wins band lookup over `values` (vectorized, one pass per rule).

    Mirrors np.select: rules are tried in order, each element takes the points of
    the first rule it satisfies, and an element matching none (including NaN,
    whose comparisons are all False) scores 0.  Returns a float array aligned to
    `values`.
    """
    score    = np.zeros(len(values), dtype=float)
    assigned = np.zeros(len(values), dtype=bool)
    for op, thr, pts in rules:
        hit = _COMPARATORS[op](values, thr) & ~assigned
        score[hit] = pts
        assigned |= hit
    return score


def assign_layout_types(
    df: pd.DataFrame,
    layout_col: str = "layout_id",
    config: LayoutConfig = DEFAULT_LAYOUT_CONFIG,
) -> pd.DataFrame:
    """
    Classify every layout as ``'text'``, ``'table'`` or ``'chart'`` and broadcast
    the verdict (plus its raw score) down to line level.

    Pure, vectorized per-layout classification.  A signed score is accumulated
    per layout: negative pulls toward text, positive toward table.  Two columns
    are added to ``df`` (one value per line, constant within a layout):

        layout_type   'text' | 'table' | 'chart'
        layout_score  float — the accumulated score (NaN when a shortcut decided
                      the layout without scoring; see below)

    Decision order (first match wins)
    ---------------------------------
    Shortcuts (no scoring, ``layout_score`` = NaN):
        1. any line block_type in config.chart_block_types   → chart
        2. any nonblank table_id or table_grid_id            → table
        3. any line block_type in config.text_block_types    → text
        4. every line has ≤ 1 cell (cell_count)              → text

    Otherwise (layout has ≥ 1 multi-cell line and no tag above) the score is the
    sum of four band lookups, and ``layout_type`` is ``'table'`` when
    ``layout_score > config.table_score_threshold`` else ``'text'``:

        median cell_count   ≤1:-5  =2:-1  >2:+1  >5:+5  >10:+10
        distinct shapes¹    0:-2   1-2:0  >2:+1  >4:+5  >10:+10  >15:+20
        median line_score²  <-8:-10 <-5:-5 <-2:-2 <0:-2 <=2:0 >2:+2 >5:+5 >8:+10
        line count          <2:-5  <4:-1  ≥4:+1  >10:+2

        ¹ count of distinct non-null ids across shape_id_tr_above +
          shape_id_tr_below (the horizontal rules bounding each line — dense
          ruling is a table tell).
        ² median of per-line line_score (the step-10 word-gap classifier prior).

    The weights live in ``_LAYOUT_SCORE_BANDS`` (edit there to tune — no
    control-flow change needed); the shortcut block_types and the table/text
    threshold live in ``LayoutConfig``.

    Parameters
    ----------
    df : pd.DataFrame
        One row per line, carrying ``layout_col``.  All feature columns are
        optional and degrade gracefully when absent (missing cell_count makes
        every layout single-cell → text; missing line_score / shapes contribute
        0).  Required: ``layout_col``.
    layout_col : str
        Name of the layout id column (default ``'layout_id'``).
    config : LayoutConfig
        Decision policy: which block_types short-circuit to chart / text, and the
        score threshold above which a scored layout is a table.  The graded
        weights are separate — see ``_LAYOUT_SCORE_BANDS``.

    Returns
    -------
    df with two added columns: ``layout_type`` (object) and
    ``layout_score`` (float).
    """
    out = df.copy()
    if out.empty:
        out["layout_type"]  = pd.Series(dtype=object)
        out["layout_score"] = pd.Series(dtype=float)
        return out

    n   = len(out)
    idx = out.index

    # ── Per-line feature extraction (all columns optional) ───────────────────
    if "cell_count" in out.columns:
        cc = out["cell_count"].fillna(0).to_numpy(dtype=float)
    else:
        cc = np.ones(n, dtype=float)   # no cell info → treat as single-cell text

    ls = (
        out["line_score"].to_numpy(dtype=float)
        if "line_score" in out.columns
        else np.full(n, np.nan)
    )

    if "block_type" in out.columns:
        bt            = out["block_type"].fillna("").astype(str).str.strip().str.lower()
        is_chart      = bt.isin(config.chart_block_types).to_numpy()
        is_text_block = bt.isin(config.text_block_types).to_numpy()
    else:
        is_chart      = np.zeros(n, dtype=bool)
        is_text_block = np.zeros(n, dtype=bool)

    has_table = np.zeros(n, dtype=bool)
    if "table_id" in out.columns:
        has_table |= out["table_id"].notna().to_numpy()
    if "table_grid_id" in out.columns:
        has_table |= out["table_grid_id"].notna().to_numpy()

    # ── Per-layout aggregation ───────────────────────────────────────────────
    work = pd.DataFrame({
        "layout_id":     out[layout_col].to_numpy(),
        "cell_count":    cc,
        "line_score":    ls,
        "is_chart":      is_chart,
        "is_text_block": is_text_block,
        "has_table":     has_table,
    })
    agg = work.groupby("layout_id", sort=False).agg(
        n_lines        = ("cell_count",    "size"),
        median_cc      = ("cell_count",    "median"),
        max_cc         = ("cell_count",    "max"),
        median_ls      = ("line_score",    "median"),
        any_chart      = ("is_chart",      "any"),
        any_text_block = ("is_text_block", "any"),
        any_table      = ("has_table",     "any"),
    )

    # Distinct-shape count: union of the two rule-id columns, non-null, per layout.
    # Melt to long form and nunique — stays vectorized (no per-layout Python).
    shape_cols = [c for c in ("shape_id_tr_above", "shape_id_tr_below") if c in out.columns]
    if shape_cols:
        long = pd.DataFrame({
            "layout_id": np.tile(out[layout_col].to_numpy(), len(shape_cols)),
            "shape":     np.concatenate([out[c].to_numpy(dtype=object) for c in shape_cols]),
        })
        long = long[long["shape"].notna()]
        n_shapes = long.groupby("layout_id")["shape"].nunique()
        agg["n_shapes"] = agg.index.map(n_shapes).fillna(0).to_numpy(dtype=float)
    else:
        agg["n_shapes"] = 0.0

    # ── Band scores: weights come from _LAYOUT_SCORE_BANDS, applied by the
    #    generic first-match-wins engine (identical semantics to the previous
    #    np.select cascade — first matching band wins, no match / NaN => 0). ──
    total = np.zeros(len(agg), dtype=float)
    for feature, rules in _LAYOUT_SCORE_BANDS.items():
        total += _apply_score_bands(agg[feature].to_numpy(dtype=float), rules)

    # ── Verdict per layout: shortcuts first, else sign of the score ──────────
    any_chart      = agg["any_chart"].to_numpy(dtype=bool)
    any_table      = agg["any_table"].to_numpy(dtype=bool)
    any_text_block = agg["any_text_block"].to_numpy(dtype=bool)
    single_cell    = agg["max_cc"].to_numpy(dtype=float) <= 1

    scored_type = np.where(total > config.table_score_threshold, "table", "text")
    layout_type = np.select(
        [any_chart,          any_table,          any_text_block, single_cell],
        ["chart",            "table",            "text",         "text"],
        default=scored_type,
    ).astype(object)

    # Score is only meaningful for scored layouts; shortcut verdicts get NaN.
    shortcut     = any_chart | any_table | any_text_block | single_cell
    layout_score = np.where(shortcut, np.nan, total)

    # ── Broadcast back to line level ─────────────────────────────────────────
    type_map  = pd.Series(layout_type,  index=agg.index)
    score_map = pd.Series(layout_score, index=agg.index)
    out["layout_type"]  = out[layout_col].map(type_map).to_numpy()
    out["layout_score"] = pd.Series(out[layout_col].map(score_map).to_numpy(), index=idx)

    # Struct-tagged tables already carry block_type == "table" from step_06.
    # Untagged (layout-detected) tables only become known as tables here, so
    # promote block_type too — downstream consumers (block merger) key off
    # block_type alone and shouldn't need to know how the table was detected.
    # A layout can still be verdict "table" while containing a stray line that
    # already has a deliberate block_type (toc, exhibits, chart, heading, ...)
    # — any_table wins over any_text_block above — so only fill lines that
    # have no block_type yet; never overwrite an existing one.
    if "block_type" not in out.columns:
        out["block_type"] = None
    bt_is_blank = out["block_type"].isna() | (out["block_type"].astype(str).str.strip() == "")
    out.loc[(out["layout_type"] == "table") & bt_is_blank, "block_type"] = "table"

    return out


# ============================================================
# PUBLIC API
# ============================================================

def assign_layouts(
    df: pd.DataFrame,
    line_level: bool = True,
    config: LayoutConfig = DEFAULT_LAYOUT_CONFIG,
) -> pd.DataFrame:
    """
    Assign layout metadata to every row of ``df``.

    Accepts any DataFrame with a ``line_id`` column.  With ``line_level=False``,
    rows sharing a line_id are first aggregated to one row per line (see
    ``_to_line_df``); with ``line_level=True`` the frame is assumed to already
    have exactly one row per line_id.

    ``config`` is a frozen ``LayoutConfig`` — pass a pipeline-specific instance
    (built via ``dataclasses.replace(DEFAULT_LAYOUT_CONFIG, ...)``) to override
    the gap-threshold knobs; defaults match the original ``layouts.py``.

    Assumptions
    -----------
    - line_id encodes reading order (monotonically increasing).
    - page_number is present and identifies page boundaries.

    Returns
    -------
    df with added columns (joined back by line_id):
        line_gap         float  vertical gap above this line (pt)
        median_gap       float  per-page median of (floored) line gaps
        page_gap_thresh  float  adaptive per-page gap threshold
        style_change     bool   line's style differs enough from the one above
                                (split-only trigger; see _style_change_flags)
        layout_id        int    1-based band id, monotonically increasing in
                                reading order (see _assign_bands); consecutive
                                untagged multi-cell table lines share one id when
                                config.merge_untagged_tables (see
                                _merge_untagged_table_lines)
        layout_type      object 'text' | 'table' | 'chart' per layout, when
                                config.classify_types (see assign_layout_types)
        layout_score     float  signed table/text score, when
                                config.classify_types (NaN for shortcut verdicts)
    """
    if df.empty:
        empty = df.assign(
            line_gap=np.nan, median_gap=0.0, page_gap_thresh=0.0,
            style_change=False, layout_id=0,
        )
        if config.classify_types:
            empty = empty.assign(layout_type=None, layout_score=np.nan)
        return empty

    # ── Step 0: one row per line ─────────────────────────────────────────────
    if line_level:
        if df["line_id"].duplicated().any():
            raise ValueError("line_level=True requires a unique line_id per row")
        line_df = df.copy()
    else:
        line_df = _to_line_df(df)

    if "page_number" not in line_df.columns:
        raise ValueError("df must contain a 'page_number' column")

    # ── Step 1: vectorized gaps + per-page threshold ─────────────────────────
    line_df = _assign_gaps(line_df, config=config)

    # ── Step 2: vectorized per-line style-change flag ────────────────────────
    line_df = _assign_style_change(line_df, config=config)

    # ── Step 3: vectorized band assignment (layout_id) ───────────────────────
    line_df = _assign_bands(line_df, config=config)

    # ── Step 4: pull untagged multi-cell table rows into one layout ──────────
    if config.merge_untagged_tables:
        line_df = _merge_untagged_table_lines(line_df, config=config)

    # ── Step 5: eject leading table captions, reindexing layout_id if any ────
    #   Runs before classification so a split-off caption is scored on its own.
    if config.eject_table_captions:
        line_df, _ = _eject_table_captions(line_df, config=config)

    # ── Step 6: classify each layout as text / table / chart ─────────────────
    join_cols = ["line_gap", "median_gap", "page_gap_thresh", "style_change", "layout_id"]
    if config.classify_types:
        line_df = assign_layout_types(line_df, layout_col="layout_id", config=config)
        join_cols += ["layout_type", "layout_score"]

    # ── Join back onto the input rows by line_id ─────────────────────────────
    band_cols = line_df.set_index("line_id")[join_cols]
    out = df.copy()
    for col in join_cols:
        out[col] = df["line_id"].map(band_cols[col])

    return out
