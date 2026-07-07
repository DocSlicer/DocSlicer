"""
shape_relationships.py

Word-to-shape relationship detection: links, background rects, vertical grid
lines, and horizontal grid lines / underlines.

There is no df_cells anymore — everything here operates directly on df_words
(one row per word, its own bounding box). Each function matches a word's own
bbox against df_links / df_shapes and annotates that word's row. No grouping
or aggregation step is involved.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


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
# CONVENIENCE ENTRY POINT
# ================================================================================

def add_word_relationships(
    df_words: pd.DataFrame,
    df_links: pd.DataFrame = None,
    df_shapes: pd.DataFrame = None,
    df_grid_cells: pd.DataFrame = None,
    min_link_overlap_ratio: float = 0.5,
    min_rect_overlap_ratio: float = 0.5,
    grid_contain_tol: float = 1.0,
) -> pd.DataFrame:
    """
    Annotate df_words with links, background rects, and grid-cell containment
    in one call. Each input is optional — omitting one just skips that
    annotation (the corresponding default columns are still added).
    """
    df_words = add_link_relationships(df_words, df_links, min_link_overlap_ratio)
    df_words = add_rect_relationships(df_words, df_shapes, min_rect_overlap_ratio)
    df_words = add_grid_cell_relationships(df_words, df_grid_cells, grid_contain_tol)
    return df_words


# ================================================================================
# VERTICAL GRID-LINE RELATIONSHIPS
# ================================================================================

