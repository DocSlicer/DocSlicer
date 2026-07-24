# SPDX-FileCopyrightText: 2026 Market Framer Inc.
# SPDX-License-Identifier: AGPL-3.0-only
"""
Word-to-shape relationship detection: hyperlinks, background rects, table
grid-cell containment, underlines / strikethroughs, and nearest table rules.

df_words + df_shapes → df_words + relationship fields.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ._utils.line_classification import classify_line, line_gap_stats

__all__ = [
    "add_word_relationships",
    "add_link_relationships",
    "add_rect_relationships",
    "add_grid_cell_relationships",
    "add_horizontal_line_relationships",
    "add_table_rule_relationships",
    "score_horizontal_lines",
    "classify_horizontal_lines",
    "LineKpiConfig",
    "LINE_KPI_CONFIG",
    "LineClassConfig",
    "LINE_CLASS_CONFIG",
    "SCORE_CLASSES",
]

# ================================================================================
# Public API
# ================================================================================

def add_word_relationships(
    df_words: pd.DataFrame,
    df_links: pd.DataFrame | None = None,
    df_shapes: pd.DataFrame | None = None,
    df_grid_cells: pd.DataFrame | None = None,
    min_link_overlap_ratio: float = 0.5,
    min_rect_overlap_ratio: float = 0.5,
    grid_contain_tol: float = 1.0,
    min_tr_x_overlap_ratio: float = 0.75,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Annotate df_words with links, background rects, grid-cell containment,
    horizontal-line (underline / strikethrough) relationships, and nearest
    table-rule above/below in one call. Each input is optional — omitting one just
    skips that annotation (the default columns are still added).

    This also runs the full horizontal-line pipeline on df_shapes internally
    (score_horizontal_lines -> classify_horizontal_lines), so the returned
    df_shapes carries the hl_ KPI/score columns and the underline/strikethrough
    shape_role updates. Returns ``(df_words, df_shapes)`` — the enriched df_shapes
    must be captured to keep those columns downstream.
    """
    # Enrich df_shapes first: classify horizontal lines (underline / strikethrough
    # / table_rule / separator) and stash the touched-word ids, so the word merge
    # below is a pure lookup. The enriched df_shapes is returned to the caller.
    if df_shapes is not None and not df_shapes.empty:
        df_shapes = score_horizontal_lines(df_shapes, df_words)
        df_shapes = classify_horizontal_lines(df_shapes)

    df_words = add_link_relationships(df_words, df_links, min_link_overlap_ratio)
    df_words = add_rect_relationships(df_words, df_shapes, min_rect_overlap_ratio)
    df_words = add_grid_cell_relationships(df_words, df_grid_cells, grid_contain_tol)
    df_words = add_horizontal_line_relationships(df_words, df_shapes)
    df_words = add_table_rule_relationships(df_words, df_shapes, min_tr_x_overlap_ratio)
    return df_words, df_shapes


# ================================================================================
# SHARED HELPERS
# ================================================================================

def _best_overlap_per_word(
    w_xl: np.ndarray, w_xr: np.ndarray,
    w_yt: np.ndarray, w_yb: np.ndarray,
    b_xl: np.ndarray, b_xr: np.ndarray,
    b_yt: np.ndarray, b_yb: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Best-overlapping box-B for each box-A (overlap as a fraction of box-A area),
    computed over the full A x B grid in one broadcast. Returns
    ``(best_ratio, best_b_pos)``; an A with no positive overlap anywhere gets
    ratio 0.0 (its best_b_pos is then meaningless — gate on the ratio).
    """
    inter_w = (np.minimum(w_xr[:, None], b_xr[None, :])
               - np.maximum(w_xl[:, None], b_xl[None, :]))
    inter_h = (np.minimum(w_yb[:, None], b_yb[None, :])
               - np.maximum(w_yt[:, None], b_yt[None, :]))
    inter_area = np.clip(inter_w, 0.0, None) * np.clip(inter_h, 0.0, None)
    area_a = np.maximum((w_xr - w_xl) * (w_yb - w_yt), 1e-12)

    ratios = inter_area / area_a[:, None]
    best_pos = np.argmax(ratios, axis=1)
    best = np.take_along_axis(ratios, best_pos[:, None], axis=1)[:, 0]
    return best, best_pos


# ================================================================================
# LINK RELATIONSHIPS
# ================================================================================

def add_link_relationships(
    df_words: pd.DataFrame,
    df_links: pd.DataFrame | None,
    min_overlap_ratio: float = 0.5,
) -> pd.DataFrame:
    """Attach the best-overlapping hyperlink to each word."""
    df_words = df_words.copy()
    df_words["has_link"]  = False
    df_words["link_url"]  = None
    df_words["link_dest"] = None
    df_words["link_type"] = None

    if df_words.empty or df_links is None or df_links.empty:
        return df_words

    idx_arr = df_words.index.to_numpy()
    w_xl = df_words["x_left"].to_numpy(float)
    w_xr = df_words["x_right"].to_numpy(float)
    w_yt = df_words["y_top"].to_numpy(float)
    w_yb = df_words["y_bottom"].to_numpy(float)
    word_pages = df_words.groupby("page_number", sort=False).indices
    link_pages = df_links.groupby("page_number", sort=False).indices

    l_xl = df_links["x_left"].to_numpy(float)
    l_xr = df_links["x_right"].to_numpy(float)
    l_yt = df_links["y_top"].to_numpy(float)
    l_yb = df_links["y_bottom"].to_numpy(float)
    l_type = df_links["link_type"].to_numpy()
    l_url  = df_links["link_url"].to_numpy() if "link_url" in df_links.columns else None
    l_dest = df_links["link_dest"].to_numpy() if "link_dest" in df_links.columns else None

    for page_num, wpos in word_pages.items():
        lpos = link_pages.get(page_num)
        if lpos is None:
            continue

        best_ratios, best_link_pos = _best_overlap_per_word(
            w_xl[wpos], w_xr[wpos], w_yt[wpos], w_yb[wpos],
            l_xl[lpos], l_xr[lpos], l_yt[lpos], l_yb[lpos],
        )
        matched = (best_ratios >= min_overlap_ratio) & (best_ratios > 0)
        if not matched.any():
            continue

        target_word_idxs = idx_arr[wpos[matched]]
        source_link_pos  = lpos[best_link_pos[matched]]

        df_words.loc[target_word_idxs, "has_link"]  = True
        df_words.loc[target_word_idxs, "link_type"] = l_type[source_link_pos]
        if l_url is not None:
            df_words.loc[target_word_idxs, "link_url"]  = l_url[source_link_pos]
        if l_dest is not None:
            df_words.loc[target_word_idxs, "link_dest"] = l_dest[source_link_pos]

    return df_words


# ================================================================================
# RECT RELATIONSHIPS
# ================================================================================

_RECT_ROLE_EXCLUDE = {"page_background", "background_band"}


def add_rect_relationships(
    df_words: pd.DataFrame,
    df_shapes: pd.DataFrame | None,
    min_overlap_ratio: float = 0.80,
) -> pd.DataFrame:
    """Attach the best-overlapping background rect to each word."""
    df_words = df_words.copy()
    df_words["inside_rect_shape"]             = False
    df_words["background_non_stroking_color"] = None
    df_words["background_stroking_color"]     = None
    df_words["shape_id_container"]            = None

    if df_words.empty or df_shapes is None or df_shapes.empty:
        return df_words

    rects = df_shapes[df_shapes["shape_type"] == "rect"]
    if "shape_role" in rects.columns:
        rects = rects[~rects["shape_role"].isin(_RECT_ROLE_EXCLUDE)]
    if rects.empty:
        return df_words

    idx_arr = df_words.index.to_numpy()
    w_xl = df_words["x_left"].to_numpy(float)
    w_xr = df_words["x_right"].to_numpy(float)
    w_yt = df_words["y_top"].to_numpy(float)
    w_yb = df_words["y_bottom"].to_numpy(float)
    word_pages = df_words.groupby("page_number", sort=False).indices
    rect_pages = rects.groupby("page_number", sort=False).indices

    r_xl = rects["x_left"].to_numpy(float)
    r_xr = rects["x_right"].to_numpy(float)
    r_yt = rects["y_top"].to_numpy(float)
    r_yb = rects["y_bottom"].to_numpy(float)
    r_ns  = (rects["non_stroking_color"].to_numpy()
             if "non_stroking_color" in rects.columns else None)
    r_st  = (rects["stroking_color"].to_numpy()
             if "stroking_color" in rects.columns else None)
    r_sid = rects["shape_id"].to_numpy() if "shape_id" in rects.columns else None

    for page_num, wpos in word_pages.items():
        rpos = rect_pages.get(page_num)
        if rpos is None:
            continue

        best_ratios, best_rect_pos = _best_overlap_per_word(
            w_xl[wpos], w_xr[wpos], w_yt[wpos], w_yb[wpos],
            r_xl[rpos], r_xr[rpos], r_yt[rpos], r_yb[rpos],
        )
        matched = (best_ratios >= min_overlap_ratio) & (best_ratios > 0)
        if not matched.any():
            continue

        target_word_idxs = idx_arr[wpos[matched]]
        matched_rect_pos = rpos[best_rect_pos[matched]]

        df_words.loc[target_word_idxs, "inside_rect_shape"] = True
        if r_ns is not None:
            df_words.loc[target_word_idxs, "background_non_stroking_color"] = r_ns[matched_rect_pos]
        if r_st is not None:
            df_words.loc[target_word_idxs, "background_stroking_color"] = r_st[matched_rect_pos]
        if r_sid is not None:
            df_words.loc[target_word_idxs, "shape_id_container"] = r_sid[matched_rect_pos]

    return df_words


# ================================================================================
# GRID-CELL CONTAINMENT
# ================================================================================

def add_grid_cell_relationships(
    df_words: pd.DataFrame,
    df_grid_cells: pd.DataFrame | None,
    contain_tol: float = 1.0,
) -> pd.DataFrame:
    """
    Tag each word with the reconstructed table grid cell whose bbox fully
    contains it (grid_cell_id + table_grid_id). Words outside every grid cell
    (body text, a table caption above the grid, ...) get NA.

    Grid cells within a table tile the plane without overlap, so a contained
    word matches exactly one cell. ``contain_tol`` allows a small spill past
    the cell edges (absorbs glyph box / grid-line rounding).
    """
    df_words = df_words.copy()
    df_words["grid_cell_id"]  = pd.array([pd.NA] * len(df_words), dtype="Int64")
    df_words["table_grid_id"] = pd.array([pd.NA] * len(df_words), dtype="Int64")

    if df_words.empty or df_grid_cells is None or df_grid_cells.empty:
        return df_words

    g = pd.DataFrame({
        "page_number":  df_grid_cells["page_number"].to_numpy(),
        "grid_cell_id": df_grid_cells["grid_cell_id"].to_numpy(),
        "table_grid_id": df_grid_cells["table_grid_id"].to_numpy()
        if "table_grid_id" in df_grid_cells.columns else np.nan,
        "g_x_left":   df_grid_cells["x_left"].to_numpy(float),
        "g_x_right":  df_grid_cells["x_right"].to_numpy(float),
        "g_y_top":    df_grid_cells["y_top"].to_numpy(float),
        "g_y_bottom": df_grid_cells["y_bottom"].to_numpy(float),
    })

    w = pd.DataFrame({
        "_row": np.arange(len(df_words)),
        "page_number": df_words["page_number"].to_numpy(),
        "w_x_left":   df_words["x_left"].to_numpy(float),
        "w_x_right":  df_words["x_right"].to_numpy(float),
        "w_y_top":    df_words["y_top"].to_numpy(float),
        "w_y_bottom": df_words["y_bottom"].to_numpy(float),
    })

    pairs = w.merge(g, on="page_number", how="inner")
    if pairs.empty:
        return df_words

    inside = (
        (pairs["w_x_left"]   >= pairs["g_x_left"]   - contain_tol)
        & (pairs["w_x_right"]  <= pairs["g_x_right"]  + contain_tol)
        & (pairs["w_y_top"]    >= pairs["g_y_top"]    - contain_tol)
        & (pairs["w_y_bottom"] <= pairs["g_y_bottom"] + contain_tol)
    ).to_numpy()

    hit = pairs[inside].drop_duplicates("_row", keep="first")
    if hit.empty:
        return df_words

    rows = hit["_row"].to_numpy()
    df_words.iloc[rows, df_words.columns.get_loc("grid_cell_id")]  = hit["grid_cell_id"].to_numpy()
    df_words.iloc[rows, df_words.columns.get_loc("table_grid_id")] = hit["table_grid_id"].to_numpy()
    return df_words


# ================================================================================
# HORIZONTAL-LINE WORD ANNOTATIONS  (underline / strikethrough)
# ================================================================================

def add_horizontal_line_relationships(
    df_words: pd.DataFrame,
    df_shapes: pd.DataFrame | None,
) -> pd.DataFrame:
    """
    Fold classified horizontal-line insights back onto the words each line touches.

    Reuses the word sets the line scorer already computed (hl_run_word_ids for the
    underlined text run, hl_strike_word_ids for struck words), so this step does no
    geometry of its own — the line's classification (shape_role == "underline" /
    "strikethrough") is the gate. Run AFTER score_horizontal_lines +
    classify_horizontal_lines have populated shape_role and the id columns.

    Columns added to df_words:
        is_underlined          bool  — a line classified "underline" hugs this word
        shape_id_underline     obj   — that line's shape_id (last wins on overlap)
        is_strikethrough       bool  — a line classified "strikethrough" crosses it
        strikethrough_ratio    float — 1.0 when struck, else 0.0
        shape_id_strikethrough obj   — that line's shape_id (last wins on overlap)
    """
    df_words = df_words.copy()
    df_words["is_underlined"]          = False
    df_words["shape_id_underline"]     = None
    df_words["is_strikethrough"]       = False
    df_words["strikethrough_ratio"]    = 0.0
    df_words["shape_id_strikethrough"] = None

    if (df_words.empty or df_shapes is None or df_shapes.empty
            or "shape_role" not in df_shapes.columns):
        return df_words

    has_shape_id = "shape_id" in df_shapes.columns

    def _apply(role: str, ids_col: str, flag_col: str, id_col: str) -> None:
        if ids_col not in df_shapes.columns:
            return
        lines = df_shapes[df_shapes["shape_role"] == role]
        if lines.empty:
            return
        # Fold every line's word ids into one word_id -> shape_id map first (a
        # later line overwrites an earlier one, preserving the documented
        # last-wins-on-overlap), then write each column once.
        shape_ids = (lines["shape_id"] if has_shape_id
                     else pd.Series(None, index=lines.index, dtype=object))
        assign: dict = {}
        for ids, sid in zip(lines[ids_col], shape_ids):
            if not isinstance(ids, (list, tuple, np.ndarray)) or len(ids) == 0:
                continue
            for wid in ids:
                assign[wid] = sid
        if not assign:
            return
        target = df_words.index.intersection(pd.Index(list(assign)))
        if target.empty:
            return
        df_words.loc[target, flag_col] = True
        if has_shape_id:
            df_words.loc[target, id_col] = [assign[w] for w in target]

    _apply("underline", "hl_run_word_ids", "is_underlined", "shape_id_underline")
    _apply("strikethrough", "hl_strike_word_ids", "is_strikethrough",
           "shape_id_strikethrough")

    df_words.loc[df_words["is_strikethrough"], "strikethrough_ratio"] = 1.0
    return df_words


# Horizontal-line roles that block a word<->table_rule search: an explicit
# separator, or an undetermined ("other") line the classifier couldn't resolve.
_TR_WALL_ROLES = ("separator", "other")


def add_table_rule_relationships(
    df_words: pd.DataFrame,
    df_shapes: pd.DataFrame | None,
    min_x_overlap_ratio: float = 0.75,
) -> pd.DataFrame:
    """
    Tag each word with the nearest table-rule line above and below it (by vertical
    distance), restricted to rules that horizontally back the word.

    A word only considers a table rule (shape_role == "table_rule") if the rule
    overlaps the word's x-span by at least ``min_x_overlap_ratio`` of the word's
    width (e.g. a word at x [80, 110] needs >= 0.75 * 30 = 22.5 pt of overlap).
    There is no y-distance limit and no contact requirement: intervening words
    between the word and the rule are fine — the nearest qualifying rule on each
    side wins. A rule level with the word's box (its y-center inside the glyph box)
    is neither above nor below, so it is skipped.

    A search can't cross a barrier line: the nearest horizontal line (shape_type ==
    "line", shape_orientation == "horizontal") whose shape_role is in
    ``_TR_WALL_ROLES`` (a "separator", or an undetermined "other" line) and whose
    x-span passes over the word acts as a wall on each side. Rects and vertical
    lines are excluded even when their shape_role is still the unclassified
    "other" default — only horizontal lines are eligible walls. A table rule
    beyond that wall (farther from the word than the barrier) is excluded, so a
    word directly under a full-width divider gets no shape_id_tr_below from rules
    living below that divider.

    Columns added to df_words:
        shape_id_tr_above   obj — shape_id of the nearest qualifying rule above
        shape_id_tr_below   obj — shape_id of the nearest qualifying rule below
    Words with no qualifying rule on a side get None there.
    """
    df_words = df_words.copy()
    df_words["shape_id_tr_above"] = None
    df_words["shape_id_tr_below"] = None

    if (df_words.empty or df_shapes is None or df_shapes.empty
            or "shape_role" not in df_shapes.columns):
        return df_words

    rules = df_shapes[df_shapes["shape_role"] == "table_rule"]
    if rules.empty:
        return df_words
    has_shape_id = "shape_id" in rules.columns
    wall_mask = df_shapes["shape_role"].isin(_TR_WALL_ROLES)
    if "shape_type" in df_shapes.columns:
        wall_mask &= df_shapes["shape_type"].astype("string") == "line"
    if "shape_orientation" in df_shapes.columns:
        wall_mask &= df_shapes["shape_orientation"].astype("string") == "horizontal"
    walls = df_shapes[wall_mask]

    idx_arr = df_words.index.to_numpy()
    a_xl = df_words["x_left"].to_numpy(float)
    a_xr = df_words["x_right"].to_numpy(float)
    a_yt = df_words["y_top"].to_numpy(float)
    a_yb = df_words["y_bottom"].to_numpy(float)
    word_pages = df_words.groupby("page_number", sort=False).indices
    rule_pages = rules.groupby("page_number", sort=False).indices
    wall_pages = (walls.groupby("page_number", sort=False).indices
                  if not walls.empty else {})

    r_xl_all = rules["x_left"].to_numpy(float)
    r_xr_all = rules["x_right"].to_numpy(float)
    r_yc_all = 0.5 * (rules["y_top"].to_numpy(float)
                      + rules["y_bottom"].to_numpy(float))
    r_ids_all = (rules["shape_id"].to_numpy() if has_shape_id
                 else rules.index.to_numpy())

    if not walls.empty:
        s_xl_all = walls["x_left"].to_numpy(float)
        s_xr_all = walls["x_right"].to_numpy(float)
        s_yc_all = 0.5 * (walls["y_top"].to_numpy(float)
                          + walls["y_bottom"].to_numpy(float))

    for page_num, wpos in word_pages.items():
        rpos = rule_pages.get(page_num)
        if rpos is None:
            continue

        word_idxs = idx_arr[wpos]
        w_xl = a_xl[wpos]
        w_xr = a_xr[wpos]
        w_yt = a_yt[wpos]
        w_yb = a_yb[wpos]
        w_width  = np.maximum(w_xr - w_xl, 1e-6)
        w_center = 0.5 * (w_yt + w_yb)
        w_cx     = 0.5 * (w_xl + w_xr)

        r_xl = r_xl_all[rpos]
        r_xr = r_xr_all[rpos]
        r_yc = r_yc_all[rpos]
        r_ids = r_ids_all[rpos]

        # Barrier walls: the nearest wall line (separator / undetermined "other")
        # that spans over the word's x-center caps how far the rule search may reach
        # on each side. No wall on a side => an open wall (+/-inf).
        wall_above = np.full(len(word_idxs), -np.inf)
        wall_below = np.full(len(word_idxs), np.inf)
        spos = wall_pages.get(page_num)
        if spos is not None:
            s_xl = s_xl_all[spos]
            s_xr = s_xr_all[spos]
            s_yc = s_yc_all[spos]
            spans = (s_xl[None, :] <= w_cx[:, None]) & (s_xr[None, :] >= w_cx[:, None])
            sep_above = spans & (s_yc[None, :] <= w_yt[:, None])
            sep_below = spans & (s_yc[None, :] >= w_yb[:, None])
            wall_above = np.where(sep_above, s_yc[None, :], -np.inf).max(axis=1)
            wall_below = np.where(sep_below, s_yc[None, :],  np.inf).min(axis=1)

        # x overlap of each (word, rule) pair, as an absolute pt amount.
        overlap = (np.minimum(w_xr[:, None], r_xr[None, :])
                   - np.maximum(w_xl[:, None], r_xl[None, :]))
        x_ok = np.clip(overlap, 0.0, None) >= (min_x_overlap_ratio * w_width[:, None])

        # Above: rule center at/above the word's top, not past the separator wall.
        # Below: at/below its bottom, not past the wall.
        above = x_ok & (r_yc[None, :] <= w_yt[:, None]) & (r_yc[None, :] >= wall_above[:, None])
        below = x_ok & (r_yc[None, :] >= w_yb[:, None]) & (r_yc[None, :] <= wall_below[:, None])

        dist    = np.abs(r_yc[None, :] - w_center[:, None])
        d_above = np.where(above, dist, np.inf)
        d_below = np.where(below, dist, np.inf)

        has_above = np.isfinite(d_above).any(axis=1)
        has_below = np.isfinite(d_below).any(axis=1)

        if has_above.any():
            pick = np.argmin(d_above, axis=1)[has_above]
            df_words.loc[word_idxs[has_above], "shape_id_tr_above"] = r_ids[pick]
        if has_below.any():
            pick = np.argmin(d_below, axis=1)[has_below]
            df_words.loc[word_idxs[has_below], "shape_id_tr_below"] = r_ids[pick]

    return df_words


# ================================================================================
# HORIZONTAL-LINE KPIs  (underline vs. hrule — stage 1: feature extraction)
# ================================================================================

# Diagnostic pass over df_shapes. For every *candidate* horizontal line (shape_type
# == "line", horizontal, not already shape_role == "table_grid" or "box") it computes a set
# of KPI columns — shape-only and word-relative — WITHOUT deciding anything. The
# score / classification is stage 2 (classify_horizontal_lines), kept separate so
# the KPI columns stay inspectable on their own when tuning against a new corpus.
# Non-candidate rows get NA/0 so the columns are safe to carry on df_shapes.
#
# Columns added (all prefixed hl_):
#   shape-only
#     hl_is_candidate    bool   — line was scored (horizontal, non-grid line)
#     hl_width_pct       float  — line width / page_width  (0..1)
#     hl_n_segments      Int64  — len(raw_shape_ids): raw strokes merged into it
#     hl_rect_relation   str    — "none" | "inside" (fully in a rect) | "crosses"
#     hl_repeat_count    Int64  — identical lines (same x_left/x_right) on the page
#     hl_y_aligned_count Int64  — candidate lines sharing this line's y (same row rule
#                                 across columns); counts self, so 1 == none aligned
#   word-relative (from the "top set": aligned words hugging the line from above)
#     hl_top_n_words     Int64  — words in the top set
#     hl_run_x_left      float  — left edge of the underlined text run
#     hl_run_x_right     float  — right edge of the text run
#     hl_run_width       float  — run_x_right - run_x_left
#     hl_width_coverage  float  — overlap(line, run) / line_len  (how backed by text)
#     hl_line_over_run   float  — line_len / run_width  (>1 => line overhangs text)
#     hl_top_content     str    — "text" | "table" | "undetermined" (gap+content prior)
#     hl_top_is_bimodal  bool   — top set has a space/gutter gap valley (table-ish)
#     hl_top_jump_ratio  float  — largest sorted-gap jump in the top set
#     hl_gap_above       float  — line_y_top  - nearest-above word y_bottom (hug)
#     hl_gap_below       float  — nearest-below word y_top - line_y_bottom  (air below)
#   strikethrough (words the line passes vertically through, near their y_center)
#     hl_strike_n_words  Int64  — words struck through
#     hl_strike_coverage float  — struck-word overlap width / line_len
#     hl_strike_center_dev float — mean |line_y_center - struck word y_center|


@dataclass(frozen=True)
class LineKpiConfig:
    # A word belongs to a line's "top set" if its y_bottom sits within this many pt
    # of the nearest-above word's y_bottom (absorbs OCR baseline jitter / aligned
    # words that are a hair off).
    top_set_jitter: float = 5.0
    # A word may dip this far below the line's top edge and still count as "above"
    # (glyph descenders / the rule drawn a hair into the text box).
    above_overlap_tol: float = 4.0
    # A word only joins the top / below sets if it horizontally overlaps the line's
    # x-span (plus this slack). Keeps a multi-column row from pulling the adjacent
    # column's words across the gutter into the set. Small slack absorbs a rule
    # drawn a hair short of its first/last glyph.
    x_overlap_tol: float = 0.5
    # Strikethrough: a word is "struck" when the line passes through its box and
    # sits within this many pt of the word's y_center. Mirrors
    # shape_marry.UnderlineConfig.strike_center_tol.
    strike_center_tol: float = 2.0
    # Two candidate lines are "the same line repeated" when both their x_left and
    # x_right agree within this many pt.
    repeat_tol: float = 1.0
    # Two candidate lines are "on the same row rule" (y-aligned) when their
    # y_centers agree within this many pt. Feeds hl_y_aligned_count — a small tol
    # keeps only lines that are visually flush (table row separators, header rules).
    y_align_tol: float = 2.0
    # Slack when testing "line fully inside a rect".
    rect_contain_tol: float = 1.0


LINE_KPI_CONFIG = LineKpiConfig()


# The float / Int64 / object columns this pass writes, with their non-candidate
# defaults. Kept in one place so init and the "no work" early-returns stay in sync.
_HL_FLOAT_COLS = (
    "hl_width_pct", "hl_run_x_left", "hl_run_x_right", "hl_run_width",
    "hl_width_coverage", "hl_line_over_run", "hl_top_jump_ratio",
    "hl_gap_above", "hl_gap_below",
    "hl_strike_coverage", "hl_strike_center_dev",
)
_HL_INT_COLS = ("hl_n_segments", "hl_repeat_count", "hl_y_aligned_count",
                "hl_top_n_words", "hl_strike_n_words")


def _init_hl_columns(df: pd.DataFrame) -> None:
    """Add every hl_ column with its non-candidate default, in place."""
    n = len(df)
    df["hl_is_candidate"] = np.zeros(n, dtype=bool)
    for c in _HL_FLOAT_COLS:
        df[c] = np.full(n, np.nan)
    for c in _HL_INT_COLS:
        df[c] = pd.array([pd.NA] * n, dtype="Int64")
    df["hl_rect_relation"] = pd.array([pd.NA] * n, dtype="string")
    df["hl_top_content"] = pd.array([pd.NA] * n, dtype="string")
    df["hl_top_is_bimodal"] = pd.array([pd.NA] * n, dtype="boolean")
    # Word ids the line touches, captured here where the masks already exist so the
    # word-merge step (add_horizontal_line_relationships) needs no geometry of its
    # own. None on non-candidate / no-match rows; a list of df_words index values
    # otherwise. hl_run_word_ids = the underlined text run (top set);
    # hl_strike_word_ids = words the line strikes through.
    for c in ("hl_run_word_ids", "hl_strike_word_ids"):
        df[c] = pd.Series([None] * n, index=df.index, dtype=object)


def _candidate_line_mask(df_shapes: pd.DataFrame) -> np.ndarray:
    """Horizontal 'line' shapes that are not already a table grid rule or box edge."""
    mask = np.ones(len(df_shapes), dtype=bool)
    if "shape_type" in df_shapes.columns:
        mask &= (df_shapes["shape_type"].astype("string") == "line").to_numpy()
    if "shape_orientation" in df_shapes.columns:
        mask &= (df_shapes["shape_orientation"].astype("string") == "horizontal").to_numpy()
    if "shape_role" in df_shapes.columns:
        mask &= (~df_shapes["shape_role"].astype("string").isin(["table_grid", "box"])).to_numpy()
    return mask


def score_horizontal_lines(
    df_shapes: pd.DataFrame,
    df_words: pd.DataFrame,
    config: LineKpiConfig = LINE_KPI_CONFIG,
) -> pd.DataFrame:
    """
    Annotate df_shapes with per-line KPI columns (see the section header) for every
    candidate horizontal line. Returns a copy; leaves the classification to stage 2.

    Shape-only KPIs are vectorised across each page's candidate lines. The
    word-relative KPIs loop over candidate lines (a handful per page) and use numpy
    masks on that page's word arrays, so the whole pass stays cheap.
    """
    df_shapes = df_shapes.copy()
    _init_hl_columns(df_shapes)

    if df_shapes.empty:
        return df_shapes

    cand = _candidate_line_mask(df_shapes)
    if not cand.any():
        return df_shapes

    df_shapes.loc[cand, "hl_is_candidate"] = True

    # --- shape-only, vectorised across all candidates ------------------------
    ci = np.flatnonzero(cand)                      # positional indices of candidates
    s = df_shapes.iloc[ci]

    x_left = s["x_left"].to_numpy(float)
    x_right = s["x_right"].to_numpy(float)
    line_len = np.maximum(x_right - x_left, 1e-6)
    y_top = s["y_top"].to_numpy(float)
    y_bottom = s["y_bottom"].to_numpy(float)
    page = s["page_number"].to_numpy()

    if "page_width" in s.columns:
        pw = s["page_width"].to_numpy(float)
        df_shapes.iloc[ci, df_shapes.columns.get_loc("hl_width_pct")] = \
            (x_right - x_left) / np.where(pw > 0, pw, np.nan)

    if "raw_shape_ids" in s.columns:
        nseg = s["raw_shape_ids"].map(
            lambda v: len(v) if isinstance(v, (list, tuple, np.ndarray)) else pd.NA
        )
        df_shapes.iloc[ci, df_shapes.columns.get_loc("hl_n_segments")] = \
            nseg.to_numpy()

    # Repeat count: group candidates by (page, rounded x_left, rounded x_right).
    tol = max(config.repeat_tol, 1e-6)
    key = pd.DataFrame({
        "p": page,
        "l": np.round(x_left / tol),
        "r": np.round(x_right / tol),
    })
    repeat = key.groupby(["p", "l", "r"])["p"].transform("size").to_numpy()
    df_shapes.iloc[ci, df_shapes.columns.get_loc("hl_repeat_count")] = repeat

    # Y-aligned count: candidate lines on the page flush at the same y (a row rule
    # split across columns, like the 4 short lines above a table header). Single-
    # linkage cluster on y_center within each page — sort, then start a new group
    # whenever the gap to the previous line exceeds y_align_tol, so there are no
    # arbitrary bin edges. Count includes self, so 1 means no other line is aligned.
    ytol = max(config.y_align_tol, 1e-6)
    y_center = 0.5 * (y_top + y_bottom)
    order = np.lexsort((y_center, page))            # by page, then y_center
    p_sorted = page[order]
    y_sorted = y_center[order]
    new_group = np.empty(len(order), dtype=bool)
    new_group[0] = True
    new_group[1:] = (p_sorted[1:] != p_sorted[:-1]) | (np.diff(y_sorted) > ytol)
    group_id = np.cumsum(new_group) - 1
    counts = np.bincount(group_id)
    y_aligned = np.empty(len(order), dtype=np.int64)
    y_aligned[order] = counts[group_id]             # scatter back to input order
    df_shapes.iloc[ci, df_shapes.columns.get_loc("hl_y_aligned_count")] = y_aligned

    # Rect relation: fully-inside a background rect (0) vs. crossing its edge (+2).
    rel = _line_rect_relation(df_shapes, ci, x_left, x_right, y_top, y_bottom, page,
                              config)
    df_shapes.iloc[ci, df_shapes.columns.get_loc("hl_rect_relation")] = rel

    # --- word-relative KPIs --------------------------------------------------
    words_ok = (df_words is not None and not df_words.empty
                and {"page_number", "x_left", "x_right", "y_top", "y_bottom"}
                .issubset(df_words.columns))
    if not words_ok:
        return df_shapes

    has_fs = "font_size" in df_words.columns
    has_text = "text" in df_words.columns

    # Per-page word arrays, built once.
    by_page: dict = {}
    for pg, grp in df_words.groupby("page_number", sort=False):
        by_page[pg] = {
            "idx": grp.index.to_numpy(),
            "xl": grp["x_left"].to_numpy(float),
            "xr": grp["x_right"].to_numpy(float),
            "yt": grp["y_top"].to_numpy(float),
            "yb": grp["y_bottom"].to_numpy(float),
            "fs": grp["font_size"].to_numpy(float) if has_fs else None,
            "txt": grp["text"].astype(str).to_numpy() if has_text else None,
        }

    # Per-line results accumulate in plain arrays and are folded onto df_shapes
    # in one pass after the loop — a per-cell .iat write on this wide frame goes
    # through the block manager every time and dominates the runtime otherwise.
    n_cand = len(ci)
    o_strike_n   = np.full(n_cand, np.nan)
    o_strike_ids = np.full(n_cand, None, dtype=object)
    o_strike_cov = np.full(n_cand, np.nan)
    o_strike_dev = np.full(n_cand, np.nan)
    o_gap_below  = np.full(n_cand, np.nan)
    o_gap_above  = np.full(n_cand, np.nan)
    o_top_n      = np.full(n_cand, np.nan)
    o_run_ids    = np.full(n_cand, None, dtype=object)
    o_run_xl     = np.full(n_cand, np.nan)
    o_run_xr     = np.full(n_cand, np.nan)
    o_run_w      = np.full(n_cand, np.nan)
    o_width_cov  = np.full(n_cand, np.nan)
    o_over_run   = np.full(n_cand, np.nan)
    o_jump       = np.full(n_cand, np.nan)
    o_bimodal    = np.full(n_cand, None, dtype=object)
    o_content    = np.full(n_cand, None, dtype=object)

    for k in range(len(ci)):
        pg = page[k]
        wp = by_page.get(pg)
        if wp is None:
            continue
        lyt, lyb = y_top[k], y_bottom[k]
        lxl, lxr, llen = x_left[k], x_right[k], line_len[k]

        # Only words in the line's x-band count — a rule decorates its own column,
        # so words across the gutter must not leak into the top / below sets.
        xtol = config.x_overlap_tol
        in_band = (wp["xr"] >= lxl - xtol) & (wp["xl"] <= lxr + xtol)

        # Strikethrough: words the line passes vertically through, near their
        # center (line box strictly inside the glyph box, y within strike tol).
        lyc = 0.5 * (lyt + lyb)
        w_yc = 0.5 * (wp["yt"] + wp["yb"])
        struck = (in_band & (wp["yt"] < lyt) & (wp["yb"] > lyb)
                  & (np.abs(w_yc - lyc) <= config.strike_center_tol))
        n_struck = int(struck.sum())
        o_strike_n[k] = n_struck
        if n_struck:
            o_strike_ids[k] = wp["idx"][struck].tolist()
            s_l = float(wp["xl"][struck].min())
            s_r = float(wp["xr"][struck].max())
            s_overlap = max(0.0, min(lxr, s_r) - max(lxl, s_l))
            o_strike_cov[k] = s_overlap / llen
            o_strike_dev[k] = float(np.abs(w_yc[struck] - lyc).mean())

        # Nearest word BELOW the line (single word; no set needed).
        below = in_band & (wp["yt"] >= lyb - config.above_overlap_tol)
        if below.any():
            o_gap_below[k] = float(wp["yt"][below].min() - lyb)

        # Top set: words whose y_bottom hugs the line from above. Anchor on the
        # nearest-above word's y_bottom, then widen by the jitter tolerance.
        above = in_band & (wp["yb"] <= lyt + config.above_overlap_tol)
        if not above.any():
            continue
        yb_star = wp["yb"][above].max()
        o_gap_above[k] = float(lyt - yb_star)

        top = above & (wp["yb"] >= yb_star - config.top_set_jitter)
        n_top = int(top.sum())
        o_top_n[k] = n_top
        if n_top == 0:
            continue

        o_run_ids[k] = wp["idx"][top].tolist()
        run_l = float(wp["xl"][top].min())
        run_r = float(wp["xr"][top].max())
        run_w = max(run_r - run_l, 1e-6)
        overlap = max(0.0, min(lxr, run_r) - max(lxl, run_l))
        o_run_xl[k] = run_l
        o_run_xr[k] = run_r
        o_run_w[k] = run_r - run_l
        o_width_cov[k] = overlap / llen
        o_over_run[k] = llen / run_w

        # Text-vs-table prior on the top set: the same full gap-stats +
        # classification pass the cell builder runs per line, applied here to
        # the words above this rule. Needs font_size for gap stats and text
        # for the content classifier.
        if has_fs and n_top >= 2:
            order = np.argsort(wp["xl"][top], kind="stable")
            gxl = wp["xl"][top][order]
            gxr = wp["xr"][top][order]
            gfs = wp["fs"][top][order]
            gs = line_gap_stats(gxl, gxr, gfs)
            o_jump[k] = float(gs["jump_ratio"])
            o_bimodal[k] = bool(gs["is_bimodal"])
            if has_text:
                texts = list(wp["txt"][top][order])
                label, _ = classify_line(texts, gs)
                o_content[k] = label

    # Scatter the per-candidate arrays back onto the full-length columns.
    n_all = len(df_shapes)

    def _scatter_float(col_name: str, vals: np.ndarray) -> None:
        full = np.full(n_all, np.nan)
        full[ci] = vals
        df_shapes[col_name] = full

    def _scatter_obj(col_name: str, vals: np.ndarray, dtype: str | None = None) -> None:
        full = np.full(n_all, None, dtype=object)
        full[ci] = vals
        if dtype is None:
            df_shapes[col_name] = pd.Series(full, index=df_shapes.index, dtype=object)
        else:
            df_shapes[col_name] = pd.array(full, dtype=dtype)

    def _scatter_int(col_name: str, vals: np.ndarray) -> None:
        full = np.full(n_all, np.nan)
        full[ci] = vals
        df_shapes[col_name] = pd.array(full, dtype="Int64")

    _scatter_int("hl_strike_n_words", o_strike_n)
    _scatter_obj("hl_strike_word_ids", o_strike_ids)
    _scatter_float("hl_strike_coverage", o_strike_cov)
    _scatter_float("hl_strike_center_dev", o_strike_dev)
    _scatter_float("hl_gap_below", o_gap_below)
    _scatter_float("hl_gap_above", o_gap_above)
    _scatter_int("hl_top_n_words", o_top_n)
    _scatter_obj("hl_run_word_ids", o_run_ids)
    _scatter_float("hl_run_x_left", o_run_xl)
    _scatter_float("hl_run_x_right", o_run_xr)
    _scatter_float("hl_run_width", o_run_w)
    _scatter_float("hl_width_coverage", o_width_cov)
    _scatter_float("hl_line_over_run", o_over_run)
    _scatter_float("hl_top_jump_ratio", o_jump)
    _scatter_obj("hl_top_is_bimodal", o_bimodal, dtype="boolean")
    _scatter_obj("hl_top_content", o_content, dtype="string")

    return df_shapes


def _line_rect_relation(
    df_shapes: pd.DataFrame,
    ci: np.ndarray,
    x_left: np.ndarray, x_right: np.ndarray,
    y_top: np.ndarray, y_bottom: np.ndarray,
    page: np.ndarray,
    config: LineKpiConfig,
) -> np.ndarray:
    """
    Per candidate line, classify its relation to the (non-background) rects on its
    page: "inside" if some rect fully contains it, else "crosses" if it cuts across
    a rect's interior, else "none". Returned as an object array aligned to ci.
    """
    out = np.array(["none"] * len(ci), dtype=object)
    if "shape_type" not in df_shapes.columns:
        return out

    rects = df_shapes[df_shapes["shape_type"].astype("string") == "rect"]
    if "shape_role" in rects.columns:
        rects = rects[~rects["shape_role"].isin(_RECT_ROLE_EXCLUDE)]
    if rects.empty:
        return out

    tol = config.rect_contain_tol
    rp = rects["page_number"].to_numpy()
    rxl = rects["x_left"].to_numpy(float)
    rxr = rects["x_right"].to_numpy(float)
    ryt = rects["y_top"].to_numpy(float)
    ryb = rects["y_bottom"].to_numpy(float)

    for k in range(len(ci)):
        same = rp == page[k]
        if not same.any():
            continue
        cx_l, cx_r = x_left[k], x_right[k]
        cy = (y_top[k] + y_bottom[k]) / 2.0
        rl, rr = rxl[same], rxr[same]
        rt, rb = ryt[same], ryb[same]

        y_in = (cy >= rt - tol) & (cy <= rb + tol)          # line sits in rect's band
        x_overlap = (cx_l <= rr) & (cx_r >= rl)             # any horizontal overlap
        contained = (cx_l >= rl - tol) & (cx_r <= rr + tol) & y_in

        if contained.any():
            out[k] = "inside"
        elif (y_in & x_overlap).any():
            out[k] = "crosses"

    return out


# ================================================================================
# HORIZONTAL-LINE CLASSIFICATION  (stage 2: score the KPIs into 4 buckets)
# ================================================================================
#
# Consumes the hl_ KPI columns from score_horizontal_lines and turns them into a
# label. Unlike the gutter scorer (one yes/no accumulator), each candidate line
# carries FOUR running scores — one per class — and every KPI votes across them:
# it can add to one bucket while deducting from another (a wide line is evidence
# for table_rule / separator and against underline). The label is the arg-max
# class, unless the top score is below `min_score` or ties the runner-up within
# `min_margin`, in which case the line is `other` (undetermined).
#
#   table_rule     a rule inside/under a table row (spans the row, repeats, grid)
#   underline      short rule hugging a text run from just below (or inside) it
#   strikethrough  rule passing through the y-center of a word run
#   separator      long standalone divider between blocks (not tabular)
#   other          none of the above won clearly
#
# The weights below are heuristics tuned against real documents. To adapt them
# to a new corpus, edit the numbers in _NUMERIC_RULES / _CATEGORICAL_RULES —
# no control-flow changes needed.

SCORE_CLASSES = ("table_rule", "underline", "strikethrough", "separator")


@dataclass(frozen=True)
class LineClassConfig:
    # The winning class must reach this score, and beat the runner-up by this
    # margin, or the line is labelled `other`.
    min_score: float = 1.0
    min_margin: float = 1.0
    undetermined_label: str = "other"
    # Strikethrough override: a line whose struck-word y-center deviation is
    # within this tolerance AND whose coverage exceeds this threshold is always
    # classified as strikethrough, bypassing the additive score entirely (see
    # the strike_override block in classify_horizontal_lines).
    strike_override_center_tol: float = 2.0
    strike_override_min_coverage: float = 0.98


LINE_CLASS_CONFIG = LineClassConfig()


# Numeric banding: column -> tuple of (lo, hi, {class: points}), lo inclusive /
# hi exclusive, open ends via -inf/inf. A NaN value matches NO band (NaN
# comparisons are False), so a KPI that is NA for a line simply contributes 0 —
# e.g. a strikethrough has no top set, so all the top-set KPIs stay neutral for it.
_NEG_INF, _INF = float("-inf"), float("inf")
# Sentinel for hard-gating a class out of the argmax entirely. Must be finite
# (unlike _NEG_INF) so it survives a round-trip through a float64 DB column,
# while staying far below any realistic combined additive score.
_HARD_GATE = -1e6

# NOTE: bands are lo <= v < hi (lo inclusive, hi exclusive)

_NUMERIC_RULES: dict[str, tuple] = {
    # --- Width as a fraction of page width -------------------------------------
    # Wide => table_rule / separator; narrow => underline / strikethrough.
    "hl_width_pct": (
        (0.70, _INF,     {"underline": -1, "table_rule": +2, "separator": +2}),
        (0.40, 0.70,     {"underline": +0, "table_rule": +1, "separator": +1}),
        (0.20, 0.40,     {}),
        (0.10, 0.20,     {"underline": +2, "table_rule": -2, "separator": -2}),
        (0.01, 0.10,     {"underline": +3, "table_rule": -5, "separator": -10}),
        (_NEG_INF, 0.01, {"underline": +5, "table_rule": -10, "separator": -20}),
    ),
    # --- How much of the line is backed by the text run above ------------------
    # line_len / run_width: ~1 => the rule fits its text exactly (underline or a
    # tight row rule); >>1 => it overhangs into margins => separator.
    "hl_width_coverage": (
        (_NEG_INF, 0.95, {"underline": -5, "table_rule": +2, "separator": +2}),
    ),
    # Vertical hug above: a negative gap means the line sits inside the glyph box,
    # the underline signature — the deeper in, the stronger. Only clearly-inside
    # lines vote underline here; a line at/below the box bottom is left neutral for
    # other KPIs to judge. A big gap means the line floats well below its text.
    "hl_gap_above": (
        (_NEG_INF, -2.0, {"underline": +3, "table_rule": -3, "separator": -50}),
        (-2.0, -1.5,     {"underline": +2, "table_rule": -1, "separator": -50}),
        (-1.5, -0.5,     {"underline": +1, "table_rule": +0, "separator": -20}),
        (-0.5, 0,        {"underline": +0, "table_rule": +1, "separator": -10}),
        (0, 1.0,         {"underline": -1, "table_rule": +5, "separator": -5}),
        (1.0, 3.0,       {"underline": -5, "table_rule": +1, "separator": +0}),
        (3.0, 8.0,       {"underline": -10, "table_rule": +0, "separator": +2}),
        (8.0, _INF,      {"underline": -20, "table_rule": -2, "separator": +3}),
    ),
    # Air below the line: a lot of it reads like a block divider.
    "hl_gap_below": (
        (0, 2.0,         {"table_rule": +2, "separator": -2}),
        (8.0, 100.0,     {"separator": +2}), # No points deducted for table (last table rule can have >>8pt distance to next word)
        (100.0, _INF,    {"separator": +5}),
    ),
    # Many identical rules on the page => table rows.
    "hl_repeat_count": ( 
        (0, 1.0,         {"underline": +0, "table_rule": -5, "separator": +5}),
        (3.0, 5.0,       {"underline": +0, "table_rule": +2, "separator": -1}),
        (5.0, 8.0,       {"underline": -2, "table_rule": +5, "separator": -2}),
        (8.0, 15.0,      {"underline": -3, "table_rule": +8, "separator": -10}),
        (15.0, _INF,     {"underline": -5, "table_rule": +10, "separator": -20}), # Don't go too negative for underline: large blocks of justified underlined text
    ),
    # Lines flush at the same y across columns (a row rule split into segments) =>
    # table evidence only. Count includes self: 2 aligned => +1, 3 => +2, 4 => +3,
    # 5+ holds at +4. table_rule only — never votes for/against the other classes.
    "hl_y_aligned_count": (
        (2.0, 3.0,       {"table_rule": +1}),
        (3.0, 4.0,       {"table_rule": +2}),
        (4.0, 5.0,       {"table_rule": +3}),
        (5.0, _INF,      {"table_rule": +4}),
    ),
    # Many raw strokes (# of raw_shape_ids in the dict) merged in => grid-built rule.
    "hl_n_segments": (
        (2.0, 5.0,       {"underline": -2, "table_rule": +2, "separator": -2}),
        (5.0, 8.0,       {"underline": -5, "table_rule": +5, "separator": -5}),
        (8.0, _INF,      {"underline": -10, "table_rule": +10, "separator": -10}),
    ),
}

# Categorical / boolean columns: column -> {value: {class: points}}.
_CATEGORICAL_RULES: dict[str, dict] = {
    # Content of the run above the line: prose above => underline, tabular => rule.
    # None means blank/NA -- no word above the line at all (not "text"/"table",
    # which require a top set) -- which reads as a block divider.
    "hl_top_content": {
        "text":          {"underline": +3, "table_rule": -2, "separator": +1},
        "table":         {"underline": -5, "table_rule": +2},
        None:            {"separator": +5},
    },
    # A space/gutter valley above means columns (table); no valley means a single
    # prose run (underline).
    "hl_top_is_bimodal": {
        False:           {"underline": +2, "table_rule": -1},
        True:            {"underline": -2, "table_rule": +2},
    },
    # Crossing a shaded band reads more like a table rule than an underline.
    "hl_rect_relation": {
        "crosses":       {"table_rule": +2, "underline": -2, "separator": -1},
    },
}


def classify_horizontal_lines(
    df_shapes: pd.DataFrame,
    config: LineClassConfig = LINE_CLASS_CONFIG,
) -> pd.DataFrame:
    """
    Score the hl_ KPI columns into per-class totals and a single label, then fold
    a decisive label onto shape_role. Run AFTER score_horizontal_lines. Returns a
    copy with columns:
        hl_score_table_rule, hl_score_underline, hl_score_strikethrough,
        hl_score_separator   (float, NaN on non-candidate rows)
        hl_class             (str: one of SCORE_CLASSES or "other"; NA otherwise)
        shape_role            updated in place for rows where hl_is_candidate is
                               True, shape_role is still the default "other" (a
                               line already tagged table_grid / box by
                               process_shapes is left alone), and hl_class is
                               decisive (one of SCORE_CLASSES, i.e. not "other").

    Fully vectorised over the candidate lines. Weights live in _NUMERIC_RULES /
    _CATEGORICAL_RULES — edit those to tune; this function is just the engine.
    """
    df = df_shapes.copy()
    n = len(df)
    for cls in SCORE_CLASSES:
        df[f"hl_score_{cls}"] = np.full(n, np.nan)
    df["hl_class"] = pd.array([pd.NA] * n, dtype="string")

    if "hl_is_candidate" not in df.columns:
        return df
    cand = df["hl_is_candidate"].fillna(False).to_numpy(bool)
    if not cand.any():
        return df

    idx = np.flatnonzero(cand)
    m = len(idx)
    sub = df.iloc[idx]
    scores = {cls: np.zeros(m) for cls in SCORE_CLASSES}

    for col_name, bands in _NUMERIC_RULES.items():
        if col_name not in sub.columns:
            continue
        v = pd.to_numeric(sub[col_name], errors="coerce").to_numpy(float)
        for lo, hi, deltas in bands:
            band_mask = (v >= lo) & (v < hi)
            for cls, pts in deltas.items():
                scores[cls][band_mask] += pts

    for col_name, mapping in _CATEGORICAL_RULES.items():
        if col_name not in sub.columns:
            continue
        col_vals = sub[col_name]
        for value, deltas in mapping.items():
            if value is None:
                # None is the blank/NA sentinel: value is missing, not a real
                # category (e.g. hl_top_content has no top set to classify).
                val_mask = col_vals.isna().to_numpy()
            else:
                # Series comparison keeps NA as NA; fillna(False) drops
                # non-matches (and NA rows) cleanly for string/boolean dtypes.
                val_mask = (col_vals == value).fillna(False).to_numpy(bool)
            for cls, pts in deltas.items():
                scores[cls][val_mask] += pts

    # Hard gate: strikethrough requires the line to actually cross >=1 word
    # (through its y-center). Without a crossing there is no strikethrough, no
    # matter how the additive weights land — so disqualify it outright. Guards
    # against a narrow whitespace line winning strikethrough by default.
    if "hl_strike_n_words" in sub.columns:
        n_struck = pd.to_numeric(sub["hl_strike_n_words"], errors="coerce").fillna(0).to_numpy()
        scores["strikethrough"][n_struck < 1] = _HARD_GATE

    # Arg-max class with a floor + margin gate.
    mat = np.vstack([scores[c] for c in SCORE_CLASSES])        # (n_classes, m)
    order = np.argsort(-mat, axis=0, kind="stable")
    top_i = order[0]
    top = mat[top_i, np.arange(m)]
    second = mat[order[1], np.arange(m)]
    decided = (top >= config.min_score) & ((top - second) >= config.min_margin)
    class_arr = np.array(SCORE_CLASSES, dtype=object)
    labels = np.where(decided, class_arr[top_i], config.undetermined_label)

    # Hard override: a line that crosses its struck words dead-center with
    # near-total coverage is unambiguously a strikethrough, no matter how the
    # additive score / argmax landed (e.g. a wide table-rule-shaped KPI
    # profile could otherwise outscore it). This runs as a separate decision
    # framework from the additive score, not another rule folded into it.
    if "hl_strike_center_dev" in sub.columns and "hl_strike_coverage" in sub.columns:
        center_dev = pd.to_numeric(sub["hl_strike_center_dev"], errors="coerce").to_numpy(float)
        coverage   = pd.to_numeric(sub["hl_strike_coverage"], errors="coerce").to_numpy(float)
        strike_override = (
            (center_dev < config.strike_override_center_tol)
            & (coverage > config.strike_override_min_coverage)
        )
        labels = np.where(strike_override, "strikethrough", labels)

    for cls in SCORE_CLASSES:
        full = np.full(n, np.nan)
        full[idx] = scores[cls]
        df[f"hl_score_{cls}"] = full
    label_full = np.array([pd.NA] * n, dtype=object)
    label_full[idx] = labels
    df["hl_class"] = pd.array(label_full, dtype="string")

    # Fold the decisive label onto shape_role. Gated on hl_is_candidate (not
    # just idx/cand) and shape_role == "other" so this can never touch a row
    # already tagged table_grid / box by process_shapes, even if hl_class was
    # somehow set on it -- an explicit safeguard, not just redundant with idx.
    if "shape_role" in df.columns:
        decisive = (
            df["hl_is_candidate"].fillna(False).astype(bool)
            & df["hl_class"].isin(SCORE_CLASSES)
            & (df["shape_role"] == "other")
        )
        df.loc[decisive, "shape_role"] = df.loc[decisive, "hl_class"]

    return df


