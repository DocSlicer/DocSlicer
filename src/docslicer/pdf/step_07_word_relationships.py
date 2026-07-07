"""
step_5_word_relationships.py

Word-to-shape relationship detection: links, background rects, vertical grid
lines, and horizontal grid lines / underlines.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# ================================================================================
# Public API
# ================================================================================

def add_word_relationships(
    df_words: pd.DataFrame,
    df_links: pd.DataFrame = None,
    df_shapes: pd.DataFrame = None,
    df_grid_cells: pd.DataFrame = None,
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

def _bbox_overlap_ratio(
    x_left_a: np.ndarray, x_right_a: np.ndarray,
    y_top_a:  np.ndarray, y_bottom_a: np.ndarray,
    x_left_b: float, x_right_b: float,
    y_top_b:  float, y_bottom_b: float,
) -> np.ndarray:
    """Overlap of each box-A with a single box-B, as a fraction of box-A area."""
    xi_l = np.maximum(x_left_a,  x_left_b)
    xi_r = np.minimum(x_right_a, x_right_b)
    yi_t = np.maximum(y_top_a,   y_top_b)
    yi_b = np.minimum(y_bottom_a, y_bottom_b)

    has_overlap = (xi_l < xi_r) & (yi_t < yi_b)
    inter_area  = np.where(has_overlap, (xi_r - xi_l) * (yi_b - yi_t), 0.0)
    area_a      = (x_right_a - x_left_a) * (y_bottom_a - y_top_a)

    return np.where(area_a > 0, inter_area / area_a, 0.0)


# ================================================================================
# LINK RELATIONSHIPS
# ================================================================================

def add_link_relationships(
    df_words: pd.DataFrame,
    df_links: pd.DataFrame,
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

    for page_num in df_words["page_number"].unique():
        page_links = df_links[df_links["page_number"] == page_num]
        if page_links.empty:
            continue

        word_idxs = df_words.index[df_words["page_number"] == page_num].to_numpy()
        x_left    = df_words.loc[word_idxs, "x_left"].to_numpy()
        x_right   = df_words.loc[word_idxs, "x_right"].to_numpy()
        y_top     = df_words.loc[word_idxs, "y_top"].to_numpy()
        y_bottom  = df_words.loc[word_idxs, "y_bottom"].to_numpy()

        best_ratios   = np.zeros(len(word_idxs))
        best_link_pos = np.full(len(word_idxs), -1, dtype=np.int64)

        for pos, (_, link) in enumerate(page_links.iterrows()):
            ratios = _bbox_overlap_ratio(
                x_left, x_right, y_top, y_bottom,
                link["x_left"], link["x_right"], link["y_top"], link["y_bottom"],
            )
            better = ratios > best_ratios
            best_ratios   = np.where(better, ratios, best_ratios)
            best_link_pos = np.where(better, pos,    best_link_pos)

        matched = (best_ratios >= min_overlap_ratio) & (best_link_pos >= 0)
        if not matched.any():
            continue

        target_word_idxs = word_idxs[matched]
        source_link_rows = page_links.iloc[best_link_pos[matched]]

        df_words.loc[target_word_idxs, "has_link"]  = True
        df_words.loc[target_word_idxs, "link_type"] = source_link_rows["link_type"].values
        if "link_url" in df_links.columns:
            df_words.loc[target_word_idxs, "link_url"]  = source_link_rows["link_url"].values
        if "link_dest" in df_links.columns:
            df_words.loc[target_word_idxs, "link_dest"] = source_link_rows["link_dest"].values

    return df_words


# ================================================================================
# RECT RELATIONSHIPS
# ================================================================================

_RECT_ROLE_EXCLUDE = {"page_background", "background_band"}


def add_rect_relationships(
    df_words: pd.DataFrame,
    df_shapes: pd.DataFrame,
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

    for page_num in df_words["page_number"].unique():
        page_rects = rects[rects["page_number"] == page_num]
        if page_rects.empty:
            continue

        word_idxs = df_words.index[df_words["page_number"] == page_num].to_numpy()
        x_left    = df_words.loc[word_idxs, "x_left"].to_numpy()
        x_right   = df_words.loc[word_idxs, "x_right"].to_numpy()
        y_top     = df_words.loc[word_idxs, "y_top"].to_numpy()
        y_bottom  = df_words.loc[word_idxs, "y_bottom"].to_numpy()

        best_ratios  = np.zeros(len(word_idxs))
        best_rect_pos = np.full(len(word_idxs), -1, dtype=np.int64)

        for pos, (_, rect) in enumerate(page_rects.iterrows()):
            ratios = _bbox_overlap_ratio(
                x_left, x_right, y_top, y_bottom,
                rect["x_left"], rect["x_right"], rect["y_top"], rect["y_bottom"],
            )
            better = ratios > best_ratios
            best_ratios   = np.where(better, ratios, best_ratios)
            best_rect_pos = np.where(better, pos,    best_rect_pos)

        matched = (best_ratios >= min_overlap_ratio) & (best_rect_pos >= 0)
        if not matched.any():
            continue

        target_word_idxs = word_idxs[matched]
        matched_rects     = page_rects.iloc[best_rect_pos[matched]]

        df_words.loc[target_word_idxs, "inside_rect_shape"] = True
        if "non_stroking_color" in page_rects.columns:
            df_words.loc[target_word_idxs, "background_non_stroking_color"] = matched_rects["non_stroking_color"].values
        if "stroking_color" in page_rects.columns:
            df_words.loc[target_word_idxs, "background_stroking_color"] = matched_rects["stroking_color"].values
        if "shape_id" in page_rects.columns:
            df_words.loc[target_word_idxs, "shape_id_container"] = matched_rects["shape_id"].values

    return df_words


# ================================================================================
# GRID-CELL CONTAINMENT
# ================================================================================

def add_grid_cell_relationships(
    df_words: pd.DataFrame,
    df_grid_cells: pd.DataFrame,
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
    df_shapes: pd.DataFrame,
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
        for _, line in lines.iterrows():
            ids = line[ids_col]
            if not isinstance(ids, (list, tuple, np.ndarray)) or len(ids) == 0:
                continue
            target = df_words.index.intersection(pd.Index(ids))
            if target.empty:
                continue
            df_words.loc[target, flag_col] = True
            if has_shape_id:
                df_words.loc[target, id_col] = line["shape_id"]

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
    df_shapes: pd.DataFrame,
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

    A search can't cross a barrier line: the nearest line whose shape_role is in
    ``_TR_WALL_ROLES`` (a "separator", or an undetermined "other" line) and whose
    x-span passes over the word acts as a wall on each side. A table rule beyond
    that wall (farther from the word than the barrier) is excluded, so a word
    directly under a full-width divider gets no shape_id_tr_below from rules living
    below that divider.

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
    walls = df_shapes[df_shapes["shape_role"].isin(_TR_WALL_ROLES)]

    for page_num in df_words["page_number"].unique():
        page_rules = rules[rules["page_number"] == page_num]
        if page_rules.empty:
            continue

        word_idxs = df_words.index[df_words["page_number"] == page_num].to_numpy()
        w_xl = df_words.loc[word_idxs, "x_left"].to_numpy(float)
        w_xr = df_words.loc[word_idxs, "x_right"].to_numpy(float)
        w_yt = df_words.loc[word_idxs, "y_top"].to_numpy(float)
        w_yb = df_words.loc[word_idxs, "y_bottom"].to_numpy(float)
        w_width  = np.maximum(w_xr - w_xl, 1e-6)
        w_center = 0.5 * (w_yt + w_yb)
        w_cx     = 0.5 * (w_xl + w_xr)

        r_xl = page_rules["x_left"].to_numpy(float)
        r_xr = page_rules["x_right"].to_numpy(float)
        r_yc = 0.5 * (page_rules["y_top"].to_numpy(float)
                      + page_rules["y_bottom"].to_numpy(float))
        r_ids = (page_rules["shape_id"].to_numpy() if has_shape_id
                 else page_rules.index.to_numpy())

        # Barrier walls: the nearest wall line (separator / undetermined "other")
        # that spans over the word's x-center caps how far the rule search may reach
        # on each side. No wall on a side => an open wall (+/-inf).
        wall_above = np.full(len(word_idxs), -np.inf)
        wall_below = np.full(len(word_idxs), np.inf)
        page_walls = walls[walls["page_number"] == page_num]
        if not page_walls.empty:
            s_xl = page_walls["x_left"].to_numpy(float)
            s_xr = page_walls["x_right"].to_numpy(float)
            s_yc = 0.5 * (page_walls["y_top"].to_numpy(float)
                          + page_walls["y_bottom"].to_numpy(float))
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
# score / classification is stage 2 and is intentionally NOT done here: we want to
# eyeball these columns on real docs first, then fit thresholds (like the gutter
# scorer). Non-candidate rows get NA/0 so the columns are safe to carry on df_shapes.
#
# Columns added (all prefixed hl_):
#   shape-only
#     hl_is_candidate    bool   — line was scored (horizontal, non-grid line)
#     hl_width_pct       float  — line width / page_width  (0..1)
#     hl_n_segments      Int64  — len(raw_shape_ids): raw strokes merged into it
#     hl_rect_relation   str    — "none" | "inside" (fully in a rect) | "crosses"
#     hl_repeat_count    Int64  — identical lines (same x_left/x_right) on the page
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
    # words that are a hair off). May be too wide — this is the knob to tune.
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
_HL_INT_COLS = ("hl_n_segments", "hl_repeat_count", "hl_top_n_words",
                "hl_strike_n_words")


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

    from docslicer.pdf.step_10_cell_builder import _line_gap_stats, classify_line

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

    col = df_shapes.columns.get_loc
    for k in range(len(ci)):
        pg = page[k]
        wp = by_page.get(pg)
        if wp is None:
            continue
        row = ci[k]
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
        df_shapes.iat[row, col("hl_strike_n_words")] = n_struck
        if n_struck:
            df_shapes.iat[row, col("hl_strike_word_ids")] = wp["idx"][struck].tolist()
            s_l = float(wp["xl"][struck].min())
            s_r = float(wp["xr"][struck].max())
            s_overlap = max(0.0, min(lxr, s_r) - max(lxl, s_l))
            df_shapes.iat[row, col("hl_strike_coverage")] = s_overlap / llen
            df_shapes.iat[row, col("hl_strike_center_dev")] = float(
                np.abs(w_yc[struck] - lyc).mean())

        # Nearest word BELOW the line (single word; no set needed).
        below = in_band & (wp["yt"] >= lyb - config.above_overlap_tol)
        if below.any():
            df_shapes.iat[row, col("hl_gap_below")] = float(
                wp["yt"][below].min() - lyb)

        # Top set: words whose y_bottom hugs the line from above. Anchor on the
        # nearest-above word's y_bottom, then widen by the jitter tolerance.
        above = in_band & (wp["yb"] <= lyt + config.above_overlap_tol)
        if not above.any():
            continue
        yb_star = wp["yb"][above].max()
        df_shapes.iat[row, col("hl_gap_above")] = float(lyt - yb_star)

        top = above & (wp["yb"] >= yb_star - config.top_set_jitter)
        n_top = int(top.sum())
        df_shapes.iat[row, col("hl_top_n_words")] = n_top
        if n_top == 0:
            continue

        df_shapes.iat[row, col("hl_run_word_ids")] = wp["idx"][top].tolist()
        run_l = float(wp["xl"][top].min())
        run_r = float(wp["xr"][top].max())
        run_w = max(run_r - run_l, 1e-6)
        overlap = max(0.0, min(lxr, run_r) - max(lxl, run_l))
        df_shapes.iat[row, col("hl_run_x_left")] = run_l
        df_shapes.iat[row, col("hl_run_x_right")] = run_r
        df_shapes.iat[row, col("hl_run_width")] = run_r - run_l
        df_shapes.iat[row, col("hl_width_coverage")] = overlap / llen
        df_shapes.iat[row, col("hl_line_over_run")] = llen / run_w

        # Text-vs-table prior on the top set (borrowed, kept cheap). Needs
        # font_size for gap stats and text for the content classifier.
        if has_fs and n_top >= 2:
            order = np.argsort(wp["xl"][top], kind="stable")
            gxl = wp["xl"][top][order]
            gxr = wp["xr"][top][order]
            gfs = wp["fs"][top][order]
            gs = _line_gap_stats(gxl, gxr, gfs)
            df_shapes.iat[row, col("hl_top_jump_ratio")] = float(gs["jump_ratio"])
            df_shapes.iat[row, col("hl_top_is_bimodal")] = bool(gs["is_bimodal"])
            if has_text:
                texts = list(wp["txt"][top][order])
                label, _ = classify_line(texts, gs)
                df_shapes.iat[row, col("hl_top_content")] = label

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
# The weights below are a STARTING POINT, meant to be edited. Only the two rules
# you specified (hl_width_pct, hl_width_coverage) are "locked"; the rest are DRAFT
# seeds so the pass produces a full label column to eyeball. Tune by editing the
# numbers in _NUMERIC_RULES / _CATEGORICAL_RULES — no control-flow changes needed.

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

_NUMERIC_RULES: dict[str, tuple] = {
    # --- LOCKED: width as a fraction of page width -----------------------------
    # Wide => table_rule / separator; narrow => underline / strikethrough.
    "hl_width_pct": (
        (0.70, _INF,     {"underline": -1, "table_rule": +2, "separator": +2}),
        (0.40, 0.70,     {"underline": +0, "table_rule": +1, "separator": +1}),
        (0.20, 0.40,     {}),
        (0.10, 0.20,     {"underline": +2, "table_rule": -2, "separator": -2}),
        (0.01, 0.10,     {"underline": +3, "table_rule": -5, "separator": -10}),
        (_NEG_INF, 0.01, {"underline": +5, "table_rule": -10, "separator": -20}),
    ),
    # --- LOCKED: how much of the line is backed by the text run above ----------
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
        (8.0, 15.0,      {"underline": -3, "table_rule": +5, "separator": -10}),
        (15.0, _INF,     {"underline": -5, "table_rule": +10, "separator": -20}), # Don't go too negative for underline: large blocks of justified underlined text
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


