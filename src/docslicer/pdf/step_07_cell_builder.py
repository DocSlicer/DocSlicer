"""
step_06_cell_builder.py

Words → Cells.

Pipeline:
  1. Reading-order pre-sort + assign_line_id (gutter-zone-aware)
  2. Cell ID assignment – gap-based horizontal merging within each line
  3. Cell aggregation  – aggregate_hierarchical (words → cell rows)
  4. Link relationships     (vectorized)
  5. Rect relationships     (vectorized)
  6. Vertical-line relationships (vectorized)
  7. Cell underlines

Public API:
    df_cells, df_words = build_cells(df_words, df_shapes, df_links)
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import pandas as pd

from .._utils.line_merger import assign_line_id
from .._utils.hierarchical_aggregator import aggregate_hierarchical, build_standard_agg_spec
from .._utils.text_utils import is_bullet_token


# ================================================================================
# CONFIG
# ================================================================================

def _interpolate_font_gap(
    font_size: Optional[float],
    gap_at_low: float = 6.0,
    gap_at_high: float = 10.0,
    low_size: float = 10.0,
    high_size: float = 24.0,
) -> float:
    """Adaptive cell-merge gap threshold based on font size."""
    if font_size is None or font_size <= 0 or np.isnan(font_size):
        return gap_at_low
    if font_size <= low_size:
        return gap_at_low
    if font_size >= high_size:
        return gap_at_high
    t = (font_size - low_size) / (high_size - low_size)
    return gap_at_low + t * (gap_at_high - gap_at_low)


_BULLET_MERGE_ENABLED   = True
_BULLET_MERGE_MAX_GAP   = 30.0   # max gap for bullet → text merge
_DOLLAR_MERGE_MAX_GAP   = 60.0   # max gap for "$" → number merge
_SENTENCE_MERGE_MAX_GAP = 10.0   # wider tolerance for justified paragraph text


# Grammar-glue stopwords for sentence detection
_STOPWORDS = {
    "the", "and", "of", "to", "in", "for", "with", "as", "on", "by", "from", "at",
    "into", "among", "including", "that", "which", "who", "whose", "its",
    "is", "are", "was", "were", "be", "been", "being",
}
_PUNCT_CHARS  = ".,;:"
_STRIP_CHARS  = ".,;:()[]{}\"'"


# ================================================================================
# HELPERS
# ================================================================================

def _is_numeric_like(text: str) -> bool:
    if not text:
        return False
    text = str(text).strip()
    return bool(text) and all(ch.isdigit() or ch in ",.()-+% " for ch in text)


def _is_bullet_token(text: str) -> bool:
    return is_bullet_token(text)


# ================================================================================
# SENTENCE DETECTION
# ================================================================================

def is_sentence_like_line(words_df_line: pd.DataFrame) -> tuple[bool, int]:
    """
    Heuristic: is this line a paragraph sentence rather than a table row?

    Returns (is_sentence_like, score).
    """
    if words_df_line is None or words_df_line.empty:
        return False, 0

    n = len(words_df_line)
    if n < 5:
        return False, 0

    alpha_tokens   = int(words_df_line.get("alpha_word_count",   pd.Series([0])).sum())
    digit_chars    = int(words_df_line.get("digit_count",        pd.Series([0])).sum())
    capitalized    = int(words_df_line.get("capitalized_word_count", pd.Series([0])).sum())
    total_chars    = int(words_df_line.get("char_count",         pd.Series([0])).sum())

    alpha_ratio   = alpha_tokens / max(n, 1)
    numeric_ratio = digit_chars  / max(total_chars, 1)

    # Vectorized stopword and punctuation detection
    text_series = words_df_line["text"].astype(str)
    t_norm = text_series.str.lower().str.strip(_STRIP_CHARS)
    stop_hits = int(t_norm.isin(_STOPWORDS).sum())

    if "alpha_count" in words_df_line.columns:
        alpha_ok = words_df_line["alpha_count"].gt(0)
    else:
        alpha_ok = text_series.str.contains(r"[a-zA-Z]", regex=True, na=False)
    has_punct = bool((alpha_ok & text_series.str.contains(r"[.,;:]", regex=True, na=False)).any())

    score = 0
    if n >= 8:
        score += 2
    elif n >= 6:
        score += 1

    if stop_hits >= 2:
        score += 2
    elif stop_hits == 1:
        score += 1

    if has_punct:
        score += 1

    if alpha_ratio >= 0.75:
        score += 2
    elif alpha_ratio >= 0.60:
        score += 1

    if numeric_ratio >= 0.35:
        score -= 3
    elif numeric_ratio >= 0.20:
        score -= 2
    elif numeric_ratio >= 0.12:
        score -= 1

    if alpha_tokens >= 4 and capitalized <= alpha_tokens * 0.5:
        score += 1

    if capitalized >= 4 and stop_hits == 0:
        score -= 2

    return score >= 4, score


# ================================================================================
# CELL ID ASSIGNMENT
# ================================================================================

def _assign_cell_ids(df: pd.DataFrame) -> pd.DataFrame:
    """
    Assign a global cell_id to df based on horizontal gaps within each line_id.

    Rules per line_id (left→right):
      - If sentence-like: use _SENTENCE_MERGE_MAX_GAP
      - Else: font-size-interpolated threshold
      - Bullet/dollar special merges override threshold

    All per-word flags and per-line sentence scores are computed in a single
    vectorized pass before the loop, so no pandas calls happen inside it.
    """
    if df.empty:
        df = df.copy()
        df["cell_id"] = pd.Series(dtype="int64")
        df["is_sentence_like"] = pd.Series(dtype="bool")
        df["sentence_score"] = pd.Series(dtype="int32")
        return df

    df = df.sort_values(
        ["page_number", "line_id", "x_left", "y_top"],
        kind="mergesort",
    ).reset_index(drop=True)

    line_arr      = df["line_id"].to_numpy(dtype=np.int64)
    x_left_arr    = df["x_left"].to_numpy(dtype=float)
    x_right_arr   = df["x_right"].to_numpy(dtype=float)
    font_size_arr = df["font_size"].to_numpy(dtype=float)
    text_arr      = df["text"].astype(str).to_numpy()

    # ---- Pre-compute per-word flags (one vectorized pass over all words) ----
    stripped      = pd.Series(text_arr).str.strip()
    t_norm        = stripped.str.lower().str.strip(_STRIP_CHARS)

    # np.frompyfunc vectorises Python callables without a Python-level loop
    bullet_flags  = (
        np.frompyfunc(_is_bullet_token, 1, 1)(stripped.to_numpy()).astype(bool)
        if _BULLET_MERGE_ENABLED else
        np.zeros(len(text_arr), dtype=bool)
    )
    dollar_flags  = (stripped == "$").to_numpy()
    numeric_flags = np.frompyfunc(_is_numeric_like, 1, 1)(stripped.to_numpy()).astype(bool)
    is_stopword_arr = t_norm.isin(_STOPWORDS).to_numpy()

    if "alpha_count" in df.columns:
        alpha_ok = df["alpha_count"].gt(0).to_numpy()
    else:
        alpha_ok = stripped.str.contains(r"[a-zA-Z]", na=False).to_numpy()
    has_alpha_punct = alpha_ok & stripped.str.contains(r"[.,;:]", na=False).to_numpy()

    # ---- Pre-compute per-line sentence scores (single groupby over all lines) ----
    df["_sw"]  = is_stopword_arr
    df["_hap"] = has_alpha_punct

    agg_dict: dict = {"text": "count", "_sw": "sum", "_hap": "any"}
    for col in ("alpha_word_count", "digit_count", "capitalized_word_count", "char_count"):
        if col in df.columns:
            agg_dict[col] = "sum"

    ls = df.groupby("line_id", sort=False).agg(agg_dict)
    df.drop(columns=["_sw", "_hap"], inplace=True)

    n_col    = ls["text"]
    atok_col = ls.get("alpha_word_count",       pd.Series(0, index=ls.index))
    dch_col  = ls.get("digit_count",            pd.Series(0, index=ls.index))
    cap_col  = ls.get("capitalized_word_count", pd.Series(0, index=ls.index))
    tch_col  = ls.get("char_count",             pd.Series(1, index=ls.index))
    sw_col   = ls["_sw"]
    hp_col   = ls["_hap"]

    alpha_ratio   = atok_col / n_col.clip(lower=1)
    numeric_ratio = dch_col  / tch_col.clip(lower=1)

    score = pd.Series(0, index=ls.index, dtype=np.int32)
    score += np.where(n_col >= 8, 2, np.where(n_col >= 6, 1, 0))
    score += np.where(sw_col >= 2, 2, np.where(sw_col == 1, 1, 0))
    score += hp_col.astype(int)
    score += np.where(alpha_ratio >= 0.75, 2, np.where(alpha_ratio >= 0.60, 1, 0))
    score -= np.where(numeric_ratio >= 0.35, 3,
              np.where(numeric_ratio >= 0.20, 2,
              np.where(numeric_ratio >= 0.12, 1, 0)))
    score += ((atok_col >= 4) & (cap_col <= atok_col * 0.5)).astype(int)
    score -= ((cap_col >= 4) & (sw_col == 0)).astype(int) * 2

    # Lines with < 5 words are never sentence-like (mirrors is_sentence_like_line)
    is_sentence_map: dict = ((score >= 4) & (n_col >= 5)).to_dict()
    score_map:       dict = score.to_dict()

    # ---- Main loop: pure array lookups, no pandas calls inside ----
    n = len(df)
    cell_id_arr        = np.empty(n, dtype=np.int64)
    is_sentence_arr    = np.empty(n, dtype=bool)
    sentence_score_arr = np.empty(n, dtype=np.int32)

    next_cell_id = 1
    i = 0

    while i < n:
        current_line = line_arr[i]
        j = i + 1
        while j < n and line_arr[j] == current_line:
            j += 1

        is_sentence             = is_sentence_map.get(current_line, False)
        is_sentence_arr[i:j]    = is_sentence
        sentence_score_arr[i:j] = score_map.get(current_line, 0)

        if is_sentence:
            line_threshold = _SENTENCE_MERGE_MAX_GAP
        else:
            valid_fonts = font_size_arr[i:j]
            valid_fonts = valid_fonts[valid_fonts > 0]
            median_font = float(np.median(valid_fonts)) if valid_fonts.size > 0 else None
            line_threshold = _interpolate_font_gap(median_font)

        current_cell   = next_cell_id
        cell_id_arr[i] = current_cell

        for k in range(i, j - 1):
            gap = x_left_arr[k + 1] - x_right_arr[k]

            if gap < 0:
                next_cell_id += 1
                current_cell = next_cell_id
                cell_id_arr[k + 1] = current_cell
                continue

            if (gap <= line_threshold
                    or (bullet_flags[k] and bool(text_arr[k + 1].strip()) and gap <= _BULLET_MERGE_MAX_GAP)
                    or (dollar_flags[k] and numeric_flags[k + 1] and gap <= _DOLLAR_MERGE_MAX_GAP)):
                cell_id_arr[k + 1] = current_cell
            else:
                next_cell_id += 1
                current_cell = next_cell_id
                cell_id_arr[k + 1] = current_cell

        next_cell_id += 1
        i = j

    df["cell_id"]          = cell_id_arr
    df["is_sentence_like"] = is_sentence_arr
    df["sentence_score"]   = sentence_score_arr
    return df


# ================================================================================
# CELL AGGREGATION  (words → cells)
# ================================================================================

def _build_cells_df(df_words: pd.DataFrame) -> pd.DataFrame:
    """Aggregate words into cells via the shared hierarchical aggregator."""

    agg_spec = build_standard_agg_spec(
        identity_cols=[
            "page_number",
            "page_width",
            "page_height",
            "line_id",
            "reading_column",
            "gutter_id_left",
            "gutter_id_right",
            "is_sentence_like",
            "sentence_score",
        ],
        include_geometry=True,
        include_style=True,
        include_counts=True,
        include_metadata=False,   # no links/rects yet at word level
        include_hierarchy=False,
        include_table=False,
        extra_agg={
            "text":    lambda s: " ".join(t for t in s.astype(str) if t.strip()),
            "word_id": list,
        },
    )

    df_cells = aggregate_hierarchical(
        df_words,
        group_col="cell_id",
        agg_spec=agg_spec,
        rename_count_col={"word_id": "word_ids"},
    )

    return df_cells


# ================================================================================
# SHAPE & LINK RELATIONSHIPS  (vectorized)
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


def _add_link_relationships(
    df_cells: pd.DataFrame,
    df_links: pd.DataFrame,
    min_overlap_ratio: float = 0.5,
) -> pd.DataFrame:
    """Attach the best-overlapping hyperlink to each cell (vectorized per page)."""
    df_cells["has_link"]  = False
    df_cells["link_url"]  = None
    df_cells["link_dest"] = None
    df_cells["link_type"] = None

    if df_links.empty:
        return df_cells

    for page_num in df_cells["page_number"].unique():
        page_links = df_links[df_links["page_number"] == page_num]
        if page_links.empty:
            continue

        cell_idxs  = df_cells.index[df_cells["page_number"] == page_num].to_numpy()
        x_left     = df_cells.loc[cell_idxs, "x_left"].to_numpy()
        x_right    = df_cells.loc[cell_idxs, "x_right"].to_numpy()
        y_top      = df_cells.loc[cell_idxs, "y_top"].to_numpy()
        y_bottom   = df_cells.loc[cell_idxs, "y_bottom"].to_numpy()

        best_ratios   = np.zeros(len(cell_idxs))
        best_link_pos = np.full(len(cell_idxs), -1, dtype=np.int64)

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

        target_cell_idxs = cell_idxs[matched]
        source_link_rows = page_links.iloc[best_link_pos[matched]]

        df_cells.loc[target_cell_idxs, "has_link"]  = True
        df_cells.loc[target_cell_idxs, "link_type"] = source_link_rows["link_type"].values
        if "link_url" in df_links.columns:
            df_cells.loc[target_cell_idxs, "link_url"]  = source_link_rows["link_url"].values
        if "link_dest" in df_links.columns:
            df_cells.loc[target_cell_idxs, "link_dest"] = source_link_rows["link_dest"].values

    return df_cells


def _add_rect_relationships(
    df_cells: pd.DataFrame,
    df_shapes: pd.DataFrame,
) -> pd.DataFrame:
    """Mark cells that fall entirely inside a rect shape (vectorized per page)."""
    df_cells["inside_rect_shape"]             = False
    df_cells["background_non_stroking_color"] = None
    df_cells["background_stroking_color"]     = None
    df_cells["shape_id_container"]            = None

    if df_shapes.empty:
        return df_cells

    rects = df_shapes[df_shapes["shape_type"] == "rect"]
    if rects.empty:
        return df_cells

    for page_num in df_cells["page_number"].unique():
        page_rects = rects[rects["page_number"] == page_num]
        if page_rects.empty:
            continue

        cell_idxs = df_cells.index[df_cells["page_number"] == page_num].to_numpy()
        x_left    = df_cells.loc[cell_idxs, "x_left"].to_numpy()
        x_right   = df_cells.loc[cell_idxs, "x_right"].to_numpy()
        y_top     = df_cells.loc[cell_idxs, "y_top"].to_numpy()
        y_bottom  = df_cells.loc[cell_idxs, "y_bottom"].to_numpy()

        rx_l = page_rects["x_left"].to_numpy()
        rx_r = page_rects["x_right"].to_numpy()
        ry_t = page_rects["y_top"].to_numpy()
        ry_b = page_rects["y_bottom"].to_numpy()

        # (C, R) boolean matrix: True where cell c is entirely inside rect r
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
        first_rect_pos  = inside_2d.argmax(axis=1)
        target_cell_idxs = cell_idxs[has_any]
        matched_rects    = page_rects.iloc[first_rect_pos[has_any]]

        df_cells.loc[target_cell_idxs, "inside_rect_shape"] = True
        if "non_stroking_color" in page_rects.columns:
            df_cells.loc[target_cell_idxs, "background_non_stroking_color"] = matched_rects["non_stroking_color"].values
        if "stroking_color" in page_rects.columns:
            df_cells.loc[target_cell_idxs, "background_stroking_color"] = matched_rects["stroking_color"].values
        if "shape_id" in page_rects.columns:
            df_cells.loc[target_cell_idxs, "shape_id_container"] = matched_rects["shape_id"].values

    return df_cells


def add_vertical_line_relationships_to_cells(
    cells_df: pd.DataFrame,
    shapes_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Flag cells whose vertical center falls within any vertical line's y-range.

    Adds:
      - has_vertical_line: bool
      - shape_id_vertical_line: list[int] | None
    """
    cells_df["has_vertical_line"]       = False
    cells_df["shape_id_vertical_line"]  = None

    if shapes_df.empty:
        return cells_df

    shapes = shapes_df
    if "page_number" not in shapes.columns and "page_num" in shapes.columns:
        shapes = shapes.rename(columns={"page_num": "page_number"})

    v_lines = shapes[
        (shapes.get("shape_type") == "line") &
        (shapes.get("orientation") == "vertical")
    ]

    if v_lines.empty:
        return cells_df

    cells_df["y_top"]    = cells_df["y_top"].astype(float)
    cells_df["y_bottom"] = cells_df["y_bottom"].astype(float)

    center_y = (cells_df["y_top"].to_numpy() + cells_df["y_bottom"].to_numpy()) / 2.0

    for page in cells_df["page_number"].unique():
        page_lines = v_lines[v_lines["page_number"] == page]
        if page_lines.empty:
            continue

        cell_idxs    = np.where((cells_df["page_number"] == page).to_numpy())[0]
        cy_page      = center_y[cell_idxs]
        line_y_top   = page_lines["y_top"].to_numpy(dtype=float)
        line_y_bot   = page_lines["y_bottom"].to_numpy(dtype=float)
        line_ids     = page_lines["shape_id"].to_numpy(dtype=int)

        matches_per_cell: list[list[int]] = [[] for _ in range(len(cell_idxs))]

        for j in range(len(line_ids)):
            in_range = (cy_page >= line_y_top[j]) & (cy_page <= line_y_bot[j])
            if not in_range.any():
                continue
            lid = int(line_ids[j])
            for pos in np.where(in_range)[0]:
                matches_per_cell[pos].append(lid)

        hit_positions = [i for i, m in enumerate(matches_per_cell) if m]
        if not hit_positions:
            continue

        hit_global = cell_idxs[hit_positions]
        cells_df.loc[hit_global, "has_vertical_line"] = True
        for gi, li in zip(hit_global, hit_positions):
            cells_df.at[gi, "shape_id_vertical_line"] = matches_per_cell[li]

    return cells_df


# ================================================================================
# UNDERLINE RELATIONSHIPS  (inlined from _utils/cell_underline_assessor.py)
# ================================================================================

_UNDERLINE_COVERAGE_THRESHOLD  = 95.0
_UNDERLINE_SEPARATOR_GAP       = 10.0
_UNDERLINE_X_OVERLAP_EPS       = 1e-4


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


def _x_overlap_width(cx_l, cx_r, ax_l_arr, ax_r_arr) -> np.ndarray:
    return np.minimum(cx_r, ax_r_arr) - np.maximum(cx_l, ax_l_arr)


def assign_cell_underlines(
    cells_df: pd.DataFrame,
    shapes_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Assign underline shapes to cells.

    cells_df is mutated in-place (no copy).
    shapes_df is copied; the copy is returned with line_role assigned.
    """
    cells  = cells_df          # mutate in-place — no copy needed
    shapes = shapes_df.copy()  # shapes IS modified and returned separately

    cells["is_underlined"]      = False
    cells["shape_id_underline"] = pd.Series(pd.NA, index=cells.index, dtype="Int64")

    if "line_role" not in shapes.columns:
        shapes["line_role"] = pd.NA
    else:
        shapes["line_role"] = pd.NA

    for col in ["y_top", "y_bottom", "x_left", "x_right"]:
        if col in cells.columns:
            cells[col] = cells[col].astype(float, copy=False)
        if col in shapes.columns:
            shapes[col] = shapes[col].astype(float, copy=False)

    line_mask = (
        (shapes.get("shape_type") == "line")
        & (shapes.get("shape_orientation") == "horizontal")
    )
    lines_all = shapes[line_mask].copy()
    if lines_all.empty:
        return cells, shapes

    lines_all = lines_all.sort_values(["page_number", "y_top", "x_left"])

    for (page,), lines_page in lines_all.groupby(["page_number"]):
        page_mask = cells["page_number"] == page
        if not page_mask.any():
            shapes.loc[lines_page.index, "line_role"] = "separator"
            continue

        cells_page = cells.loc[page_mask]

        c_id    = cells_page["cell_id"].to_numpy()
        c_y_top = cells_page["y_top"].to_numpy()
        c_y_bot = cells_page["y_bottom"].to_numpy()
        c_x_l   = cells_page["x_left"].to_numpy()
        c_x_r   = cells_page["x_right"].to_numpy()
        c_band  = cells_page["horizontal_band_id"].to_numpy() if "horizontal_band_id" in cells_page.columns else None

        id_to_global_idx = dict(zip(c_id, cells_page.index))

        for line_idx, line in lines_page.iterrows():
            line_id = int(line["shape_id"])
            ly_top  = float(line["y_top"])
            lx_l    = float(line["x_left"])
            lx_r    = float(line["x_right"])

            under   = ly_top >= c_y_bot
            through = (ly_top >= c_y_top) & (ly_top <= c_y_bot)
            x_overlap_with_line = (c_x_r > lx_l + _UNDERLINE_X_OVERLAP_EPS) & (c_x_l < lx_r - _UNDERLINE_X_OVERLAP_EPS)
            eligible_mask = (under | through) & x_overlap_with_line

            if not np.any(eligible_mask):
                shapes.at[line_idx, "line_role"] = "separator"
                continue

            eligible_idx = np.where(eligible_mask)[0]
            gaps         = np.abs(c_y_bot[eligible_idx] - ly_top)
            min_gap      = float(gaps.min())
            closest_idx  = eligible_idx[gaps == min_gap]

            if min_gap > _UNDERLINE_SEPARATOR_GAP:
                shapes.at[line_idx, "line_role"] = "separator"
                continue

            closest_ids   = c_id[closest_idx]
            closest_bands = c_band[closest_idx] if c_band is not None else None

            segments     = [(c_x_l[i], c_x_r[i]) for i in closest_idx]
            coverage_pct = _union_coverage(lx_l, lx_r, segments)

            band_cells_above_idx = np.array([], dtype=int)
            if c_band is not None:
                seed_bands       = np.unique(closest_bands)
                min_seed_cell_id = int(closest_ids.min())

                band_mask = (
                    np.isin(c_band, seed_bands)
                    & (c_id < min_seed_cell_id)
                    & x_overlap_with_line
                )
                band_idx = np.where(band_mask)[0]

                if band_idx.size:
                    earlier_lines = lines_page[lines_page["y_top"] < ly_top]
                    if earlier_lines.empty:
                        band_cells_above_idx = band_idx
                    else:
                        el_x_l   = earlier_lines["x_left"].to_numpy()
                        el_x_r   = earlier_lines["x_right"].to_numpy()
                        el_y_top = earlier_lines["y_top"].to_numpy()

                        keep = []
                        for bi in band_idx:
                            mask_x = (c_x_l[bi] >= el_x_l) & (c_x_r[bi] <= el_x_r)
                            mask_y = (c_y_top[bi] <= el_y_top)
                            if not np.any(mask_x & mask_y):
                                keep.append(bi)
                        band_cells_above_idx = np.array(keep, dtype=int)

            assigned_ids      = set(int(x) for x in closest_ids)
            assigned_segments = segments.copy()
            shapes.at[line_idx, "line_role"] = "underline"

            if band_cells_above_idx.size > 0 and coverage_pct < _UNDERLINE_COVERAGE_THRESHOLD:
                cand_idx    = band_cells_above_idx
                gap_to_line = np.abs(c_y_bot[cand_idx] - ly_top)
                cand_idx    = cand_idx[np.argsort(gap_to_line)]

                for bi in cand_idx:
                    cx_l = c_x_l[bi]
                    cx_r = c_x_r[bi]

                    if assigned_segments:
                        ax_l_arr  = np.array([seg[0] for seg in assigned_segments])
                        ax_r_arr  = np.array([seg[1] for seg in assigned_segments])
                        overlaps  = _x_overlap_width(cx_l, cx_r, ax_l_arr, ax_r_arr)
                        if np.any(overlaps > _UNDERLINE_X_OVERLAP_EPS):
                            continue

                    assigned_segments.append((cx_l, cx_r))
                    assigned_ids.add(int(c_id[bi]))
                    coverage_pct = _union_coverage(lx_l, lx_r, assigned_segments)
                    if coverage_pct >= _UNDERLINE_COVERAGE_THRESHOLD:
                        break

            for cid_val in assigned_ids:
                g_idx = id_to_global_idx.get(cid_val)
                if g_idx is None:
                    continue
                cells.at[g_idx, "is_underlined"] = True
                if pd.isna(cells.at[g_idx, "shape_id_underline"]):
                    cells.at[g_idx, "shape_id_underline"] = line_id

    return cells, shapes


# ================================================================================
# READING ORDER PRE-SORT
# ================================================================================

def _sort_reading_order(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sort words into correct reading order for mixed single/multi-column pages.

    Problem with a naive (page, y_top) sort:
        A page may have singlecol content, then a 2-col section, then more singlecol.
        Grouping all col-1 words first, then col-2, pushes col-2 below the trailing
        singlecol content — violating reading order.

    Solution — compute a zone_y per word, then sort by (zone_y, reading_column, y_top, x_left):
        • Singlecol words (reading_column=1, no gutter_id_right): zone_y = y_top (their own)
        • Multicol words: zone_y = min(y_top) of col-1 words that share the same gutter zone

    For 3-column layouts the zone is propagated via the gutter chain:
        col1 has gutter_id_right=G1 → zone_y[G1] = min y_top of col-1 words
        col2 has gutter_id_left=G1, gutter_id_right=G2 → zone_y[G2] = zone_y[G1]
        col3 has gutter_id_left=G2 → gets zone_y[G2] = zone_y[G1]
    """
    if df.empty:
        return df

    has_right = "gutter_id_right" in df.columns
    has_left  = "gutter_id_left"  in df.columns
    has_rc    = "reading_column"  in df.columns

    if not has_right or df["gutter_id_right"].isna().all():
        return df.sort_values(
            ["page_number", "y_top", "x_left"], kind="mergesort"
        ).reset_index(drop=True)

    df = df.copy()
    rc = df["reading_column"].fillna(1).astype(int) if has_rc else pd.Series(1, index=df.index)

    df["_zone_y"] = df["y_top"].astype(float)
    df["_rc"]     = rc

    for page in df["page_number"].unique():
        pm = df["page_number"] == page

        col1_gutter_mask = pm & df["gutter_id_right"].notna() & (rc == 1)
        if not col1_gutter_mask.any():
            continue

        zone_y: dict = (
            df.loc[col1_gutter_mask]
            .groupby("gutter_id_right")["y_top"]
            .min()
            .to_dict()
        )

        # Propagate zone_y through gutter chains for 3+ column layouts.
        # Reduce to unique (G_left, G_right) pairs — avoids iterating over every word.
        if has_left:
            chain_mask = pm & df["gutter_id_left"].notna() & df["gutter_id_right"].notna()
            if chain_mask.any():
                chain_pairs = (
                    df.loc[chain_mask, ["gutter_id_left", "gutter_id_right"]]
                    .drop_duplicates()
                )
                chain_pairs = chain_pairs[chain_pairs["gutter_id_left"].isin(zone_y)]
                if not chain_pairs.empty:
                    chain_pairs = chain_pairs.copy()
                    chain_pairs["_inherited"] = chain_pairs["gutter_id_left"].map(zone_y)
                    min_inherited = chain_pairs.groupby("gutter_id_right")["_inherited"].min()
                    for G_right, inherited in min_inherited.items():
                        zone_y[G_right] = min(inherited, zone_y.get(G_right, inherited))

        for G, zy in zone_y.items():
            right_mask = pm & (df["gutter_id_right"] == G)
            if right_mask.any():
                df.loc[right_mask, "_zone_y"] = zy

            if has_left:
                left_mask = pm & (df["gutter_id_left"] == G)
                if left_mask.any():
                    df.loc[left_mask, "_zone_y"] = zy

    result = (
        df.sort_values(
            ["page_number", "_zone_y", "_rc", "y_top", "x_left"],
            kind="mergesort",
        )
        .drop(columns=["_zone_y", "_rc"])
        .reset_index(drop=True)
    )
    return result


# ================================================================================
# PUBLIC API
# ================================================================================

def build_cells(
    df_words: pd.DataFrame,
    df_shapes: Optional[pd.DataFrame] = None,
    df_links:  Optional[pd.DataFrame] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build cell-level DataFrame from word-level input.

    Parameters
    ----------
    df_words : pd.DataFrame
        Word-level data. Expected columns include reading_column, gutter_id_left,
        gutter_id_right (from step_05_gutter_extractor) plus standard word fields.
    df_shapes : pd.DataFrame, optional
    df_links  : pd.DataFrame, optional

    Returns
    -------
    df_cells     : one row per cell
    df_words_out : input words augmented with line_id, cell_id
    """
    if df_words is None or df_words.empty:
        return pd.DataFrame(), pd.DataFrame()

    df = df_words.copy()

    # Exclude line numbers from cell building — flagged by step_05_line_number_detector.
    # Stash them so they can be returned in df_words_out for debug visibility.
    if "line_number_flag" in df.columns:
        df_line_numbers = df[df["line_number_flag"]].copy()
        df = df[~df["line_number_flag"]].reset_index(drop=True)
    else:
        df_line_numbers = pd.DataFrame()

    # Ensure reading_column exists (default 1 = single-column)
    if "reading_column" not in df.columns:
        df["reading_column"] = 1
    df["reading_column"] = df["reading_column"].fillna(1).astype(int)

    # Separate vertical text; it bypasses cell-building
    if "text_orientation" in df.columns:
        vert_mask = df["text_orientation"].isin(["TTB", "BTT"])
    else:
        vert_mask = pd.Series(False, index=df.index)

    df_vert  = df[vert_mask].copy()
    df_horiz = df[~vert_mask].copy()

    # --- Step 1: Sort into reading order, then assign line IDs ---
    df_horiz = _sort_reading_order(df_horiz)
    df_horiz = assign_line_id(df_horiz)

    # --- Step 2: Cell ID assignment ---
    df_horiz = _assign_cell_ids(df_horiz)

    # --- Step 3: Aggregate words → cells ---
    df_cells = _build_cells_df(df_horiz)

    # Steps 4–7 mutate df_cells in-place (no intermediate copies).

    # --- Step 4: Link relationships ---
    if df_links is not None and not df_links.empty:
        _add_link_relationships(df_cells, df_links)

    # --- Step 5: Rect relationships ---
    if df_shapes is not None and not df_shapes.empty:
        _add_rect_relationships(df_cells, df_shapes)

    # --- Step 6: Vertical line relationships ---
    if df_shapes is not None and not df_shapes.empty:
        add_vertical_line_relationships_to_cells(df_cells, df_shapes)

    # --- Step 7: Underline relationships ---
    if df_shapes is not None and not df_shapes.empty:
        df_cells, df_shapes = assign_cell_underlines(df_cells, df_shapes)

    # Recombine LTR + vertical words (vertical words carry no cell/line IDs)
    # Also restore line-number words so df_words_out remains complete for debug.
    df_words_out = (
        pd.concat([df_horiz, df_vert, df_line_numbers], ignore_index=True)
        .sort_values(["page_number", "y_top", "x_left"], kind="mergesort")
        .reset_index(drop=True)
    )

    return df_cells, df_words_out
