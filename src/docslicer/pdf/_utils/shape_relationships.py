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

def add_rect_relationships(
    df_words: pd.DataFrame,
    df_shapes: pd.DataFrame,
) -> pd.DataFrame:
    """Mark words that fall entirely inside a rect shape."""
    df_words = df_words.copy()
    df_words["inside_rect_shape"]             = False
    df_words["background_non_stroking_color"] = None
    df_words["background_stroking_color"]     = None
    df_words["shape_id_container"]            = None

    if df_words.empty or df_shapes is None or df_shapes.empty:
        return df_words

    rects = df_shapes[df_shapes["shape_type"] == "rect"]
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

        rx_l = page_rects["x_left"].to_numpy()
        rx_r = page_rects["x_right"].to_numpy()
        ry_t = page_rects["y_top"].to_numpy()
        ry_b = page_rects["y_bottom"].to_numpy()

        # (W, R) boolean matrix: True where word w is entirely inside rect r
        inside_2d = (
            (x_left[:, None]   >= rx_l) &
            (x_right[:, None]  <= rx_r) &
            (y_top[:, None]    >= ry_t) &
            (y_bottom[:, None] <= ry_b)
        )

        has_any = inside_2d.any(axis=1)
        if not has_any.any():
            continue

        # argmax gives the index of the first True along axis=1 (first matching rect wins)
        first_rect_pos   = inside_2d.argmax(axis=1)
        target_word_idxs = word_idxs[has_any]
        matched_rects     = page_rects.iloc[first_rect_pos[has_any]]

        df_words.loc[target_word_idxs, "inside_rect_shape"] = True
        if "non_stroking_color" in page_rects.columns:
            df_words.loc[target_word_idxs, "background_non_stroking_color"] = matched_rects["non_stroking_color"].values
        if "stroking_color" in page_rects.columns:
            df_words.loc[target_word_idxs, "background_stroking_color"] = matched_rects["stroking_color"].values
        if "shape_id" in page_rects.columns:
            df_words.loc[target_word_idxs, "shape_id_container"] = matched_rects["shape_id"].values

    return df_words


# ================================================================================
# VERTICAL GRID-LINE RELATIONSHIPS
# ================================================================================

def add_vertical_line_relationships(
    df_words: pd.DataFrame,
    df_shapes: pd.DataFrame,
) -> pd.DataFrame:
    """
    Flag words whose vertical center falls within any vertical line's y-range.

    Adds:
      - has_vertical_grid_line: bool
      - shape_id_vertical_grid_line: list[int] | None
    """
    df_words = df_words.copy()
    df_words["has_vertical_grid_line"]      = False
    df_words["shape_id_vertical_grid_line"] = None

    if df_words.empty or df_shapes is None or df_shapes.empty:
        return df_words

    shapes = df_shapes
    if "page_number" not in shapes.columns and "page_num" in shapes.columns:
        shapes = shapes.rename(columns={"page_num": "page_number"})

    v_lines = shapes[
        (shapes.get("shape_type") == "line") &
        (shapes.get("shape_orientation") == "vertical")
    ]
    if v_lines.empty:
        return df_words

    y_top    = df_words["y_top"].astype(float).to_numpy()
    y_bottom = df_words["y_bottom"].astype(float).to_numpy()
    center_y = (y_top + y_bottom) / 2.0

    for page in df_words["page_number"].unique():
        page_lines = v_lines[v_lines["page_number"] == page]
        if page_lines.empty:
            continue

        word_idxs  = np.where((df_words["page_number"] == page).to_numpy())[0]
        cy_page    = center_y[word_idxs]
        line_y_top = page_lines["y_top"].to_numpy(dtype=float)
        line_y_bot = page_lines["y_bottom"].to_numpy(dtype=float)
        line_ids   = page_lines["shape_id"].to_numpy(dtype=int)

        matches_per_word: list[list[int]] = [[] for _ in range(len(word_idxs))]

        for j in range(len(line_ids)):
            in_range = (cy_page >= line_y_top[j]) & (cy_page <= line_y_bot[j])
            if not in_range.any():
                continue
            lid = int(line_ids[j])
            for pos in np.where(in_range)[0]:
                matches_per_word[pos].append(lid)

        hit_positions = [i for i, m in enumerate(matches_per_word) if m]
        if not hit_positions:
            continue

        hit_global = word_idxs[hit_positions]
        df_words.iloc[hit_global, df_words.columns.get_loc("has_vertical_grid_line")] = True
        for gi, li in zip(hit_global, hit_positions):
            df_words.iat[gi, df_words.columns.get_loc("shape_id_vertical_grid_line")] = matches_per_word[li]

    return df_words


# ================================================================================
# HORIZONTAL GRID-LINE / UNDERLINE RELATIONSHIPS
# ================================================================================

_UNDERLINE_COVERAGE_THRESHOLD  = 95.0
_UNDERLINE_SEPARATOR_GAP       = 10.0
_UNDERLINE_X_OVERLAP_EPS       = 1e-4

# A line covering >= this fraction of the page content width is always a grid
# line, regardless of the per-word 1.5x test.
_GRID_LINE_PAGE_COVERAGE    = 0.98
_GRID_LINE_PAGE_WIDTH_FLOOR = 400.0  # minimum denominator (pt)


def _union_coverage(line_left: float, line_right: float, segments: list[tuple[float, float]]) -> float:
    """Percent of line width covered by the union of segments."""
    line_width = line_right - line_left
    if not segments or line_width <= 0:
        return 0.0

    intervals = []
    for sx_l, sx_r in segments:
        seg_left  = max(line_left,  sx_l)
        seg_right = min(line_right, sx_r)
        if seg_right > seg_left:
            intervals.append((seg_left, seg_right))

    if not intervals:
        return 0.0

    intervals.sort(key=lambda t: t[0])
    merged = []
    cur_start, cur_end = intervals[0]
    for start, end in intervals[1:]:
        if start <= cur_end:
            cur_end = max(cur_end, end)
        else:
            merged.append((cur_start, cur_end))
            cur_start, cur_end = start, end
    merged.append((cur_start, cur_end))

    covered = sum(e - s for s, e in merged)
    return 100.0 * covered / line_width


def _x_overlap_width(
    cx_l: float, cx_r: float,
    ax_l_arr: np.ndarray, ax_r_arr: np.ndarray,
) -> np.ndarray:
    """Width of x-overlap between a candidate word and each already-assigned segment."""
    return np.minimum(cx_r, ax_r_arr) - np.maximum(cx_l, ax_l_arr)


def add_horizontal_line_relationships(
    df_words: pd.DataFrame,
    df_shapes: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Assign underline and horizontal grid-line shapes to words.

    A horizontal line whose width is <= 1.5x the matched word's width is treated
    as a text underline (is_underlined / shape_id_underline). A wider line is a
    table grid line (has_horizontal_grid_line / shape_id_horizontal_grid_line).

    Uses horizontal_band_id when present (propagated by the line builder) to widen
    underline matches to same-band words above the line; falls back to matching
    only the closest row of words when it's absent.

    Returns (df_words_out, df_shapes_out) — df_shapes_out carries an added
    line_role column ("underline" | "separator").
    """
    df_words = df_words.copy()
    df_words["is_underlined"]               = False
    df_words["shape_id_underline"]          = pd.Series(pd.NA, index=df_words.index, dtype="Int64")
    df_words["has_horizontal_grid_line"]      = False
    df_words["shape_id_horizontal_grid_line"] = pd.Series(pd.NA, index=df_words.index, dtype="Int64")

    if df_words.empty or df_shapes is None or df_shapes.empty:
        return df_words, df_shapes

    shapes = df_shapes.copy()
    shapes["line_role"] = pd.NA

    line_mask = (
        (shapes.get("shape_type") == "line")
        & (shapes.get("shape_orientation") == "horizontal")
    )
    lines_all = shapes[line_mask].copy()
    if lines_all.empty:
        return df_words, shapes

    for col in ["y_top", "y_bottom", "x_left", "x_right"]:
        df_words[col] = df_words[col].astype(float, copy=False)
        lines_all[col] = lines_all[col].astype(float, copy=False)

    lines_all = lines_all.sort_values(["page_number", "y_top", "x_left"])

    for (page,), lines_page in lines_all.groupby(["page_number"]):
        page_mask = df_words["page_number"] == page
        if not page_mask.any():
            shapes.loc[lines_page.index, "line_role"] = "separator"
            continue

        words_page = df_words.loc[page_mask]

        w_id    = words_page["word_id"].to_numpy()
        w_y_top = words_page["y_top"].to_numpy()
        w_y_bot = words_page["y_bottom"].to_numpy()
        w_x_l   = words_page["x_left"].to_numpy()
        w_x_r   = words_page["x_right"].to_numpy()
        w_band  = words_page["horizontal_band_id"].to_numpy() if "horizontal_band_id" in words_page.columns else None

        page_content_width = max(
            _GRID_LINE_PAGE_WIDTH_FLOOR,
            float(w_x_r.max()) - float(w_x_l.min()),
        )

        id_to_global_idx = dict(zip(w_id, words_page.index))
        id_to_arr_idx    = dict(zip(w_id, range(len(w_id))))

        for line_idx, line in lines_page.iterrows():
            line_id = int(line["shape_id"])
            ly_top  = float(line["y_top"])
            lx_l    = float(line["x_left"])
            lx_r    = float(line["x_right"])

            under   = ly_top >= w_y_bot
            through = (ly_top >= w_y_top) & (ly_top <= w_y_bot)
            x_overlap_with_line = (w_x_r > lx_l + _UNDERLINE_X_OVERLAP_EPS) & (w_x_l < lx_r - _UNDERLINE_X_OVERLAP_EPS)
            eligible_mask = (under | through) & x_overlap_with_line

            if not np.any(eligible_mask):
                shapes.at[line_idx, "line_role"] = "separator"
                continue

            eligible_idx = np.where(eligible_mask)[0]
            gaps         = np.abs(w_y_bot[eligible_idx] - ly_top)
            min_gap      = float(gaps.min())
            closest_idx  = eligible_idx[gaps == min_gap]

            if min_gap > _UNDERLINE_SEPARATOR_GAP:
                shapes.at[line_idx, "line_role"] = "separator"
                continue

            closest_ids   = w_id[closest_idx]
            closest_bands = w_band[closest_idx] if w_band is not None else None

            segments     = [(w_x_l[i], w_x_r[i]) for i in closest_idx]
            coverage_pct = _union_coverage(lx_l, lx_r, segments)

            band_words_above_idx = np.array([], dtype=int)
            if w_band is not None:
                seed_bands       = np.unique(closest_bands)
                min_seed_word_id = int(closest_ids.min())

                band_mask = (
                    np.isin(w_band, seed_bands)
                    & (w_id < min_seed_word_id)
                    & x_overlap_with_line
                )
                band_idx = np.where(band_mask)[0]

                if band_idx.size:
                    earlier_lines = lines_page[lines_page["y_top"] < ly_top]
                    if earlier_lines.empty:
                        band_words_above_idx = band_idx
                    else:
                        el_x_l   = earlier_lines["x_left"].to_numpy()
                        el_x_r   = earlier_lines["x_right"].to_numpy()
                        el_y_top = earlier_lines["y_top"].to_numpy()

                        keep = []
                        for bi in band_idx:
                            mask_x = (w_x_l[bi] >= el_x_l) & (w_x_r[bi] <= el_x_r)
                            mask_y = (w_y_top[bi] <= el_y_top)
                            if not np.any(mask_x & mask_y):
                                keep.append(bi)
                        band_words_above_idx = np.array(keep, dtype=int)

            assigned_ids      = set(int(x) for x in closest_ids)
            assigned_segments = segments.copy()
            shapes.at[line_idx, "line_role"] = "underline"

            if band_words_above_idx.size > 0 and coverage_pct < _UNDERLINE_COVERAGE_THRESHOLD:
                cand_idx    = band_words_above_idx
                gap_to_line = np.abs(w_y_bot[cand_idx] - ly_top)
                cand_idx    = cand_idx[np.argsort(gap_to_line)]

                for bi in cand_idx:
                    wx_l = w_x_l[bi]
                    wx_r = w_x_r[bi]

                    if assigned_segments:
                        ax_l_arr = np.array([seg[0] for seg in assigned_segments])
                        ax_r_arr = np.array([seg[1] for seg in assigned_segments])
                        overlaps = _x_overlap_width(wx_l, wx_r, ax_l_arr, ax_r_arr)
                        if np.any(overlaps > _UNDERLINE_X_OVERLAP_EPS):
                            continue

                    assigned_segments.append((wx_l, wx_r))
                    assigned_ids.add(int(w_id[bi]))
                    coverage_pct = _union_coverage(lx_l, lx_r, assigned_segments)
                    if coverage_pct >= _UNDERLINE_COVERAGE_THRESHOLD:
                        break

            line_width         = lx_r - lx_l
            is_full_width_line = line_width >= _GRID_LINE_PAGE_COVERAGE * page_content_width
            for wid_val in assigned_ids:
                g_idx   = id_to_global_idx.get(wid_val)
                arr_idx = id_to_arr_idx.get(wid_val)
                if g_idx is None or arr_idx is None:
                    continue
                word_width   = float(w_x_r[arr_idx] - w_x_l[arr_idx])
                is_underline = (not is_full_width_line) and (line_width <= 1.5 * word_width)
                if is_underline:
                    df_words.at[g_idx, "is_underlined"] = True
                    if pd.isna(df_words.at[g_idx, "shape_id_underline"]):
                        df_words.at[g_idx, "shape_id_underline"] = line_id
                else:
                    df_words.at[g_idx, "has_horizontal_grid_line"] = True
                    if pd.isna(df_words.at[g_idx, "shape_id_horizontal_grid_line"]):
                        df_words.at[g_idx, "shape_id_horizontal_grid_line"] = line_id

    return df_words, shapes
