import pandas as pd
import numpy as np

_COVERAGE_THRESHOLD = 95.0      # percent
_SEPARATOR_GAP_THRESHOLD = 10.0
_X_OVERLAP_EPS = 1e-4


def _union_coverage(line_left: float, line_right: float, segments: list[tuple[float, float]]) -> float:
    """
    Percent of line width covered by union of segments.
    segments: list of (x_left, x_right) for assigned cells.
    """
    line_width = line_right - line_left
    if not segments or line_width <= 0:
        return 0.0

    intervals = []
    for (sx_l, sx_r) in segments:
        seg_left = max(line_left, sx_l)
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

    covered_width = sum(end - start for start, end in merged)
    return 100.0 * covered_width / line_width


def _x_overlap_width(cx_l, cx_r, ax_l_arr, ax_r_arr) -> np.ndarray:
    """
    Vectorized overlap with all assigned segments.
    Returns width of overlap for each assigned segment.
    """
    return np.minimum(cx_r, ax_r_arr) - np.maximum(cx_l, ax_l_arr)


def assign_cell_underlines(
    cells_df: pd.DataFrame,
    shapes_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Optimized underline assigner.

    cells_out:
      - is_underlined : bool
      - shape_id_underline : Int64 (NA if not underlined)

    shapes_out:
      - line_role : 'underline', 'separator', or <NA> for non-line shapes
    """
    cells = cells_df.copy()
    shapes = shapes_df.copy()

    # --- init cell annotations ---
    cells["is_underlined"] = False
    cells["shape_id_underline"] = pd.Series(
        pd.NA, index=cells.index, dtype="Int64"
    )

    # --- init shape annotations ---
    if "line_role" not in shapes.columns:
        shapes["line_role"] = pd.NA
    else:
        shapes["line_role"] = pd.NA

    # Pre-cast numeric columns ONCE
    for col in ["y_top", "y_bottom", "x_left", "x_right"]:
        if col in cells.columns:
            cells[col] = cells[col].astype(float, copy=False)
        if col in shapes.columns:
            shapes[col] = shapes[col].astype(float, copy=False)

    # select horizontal line shapes
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

        # numpy views for speed
        c_id = cells_page["cell_id"].to_numpy()
        c_y_top = cells_page["y_top"].to_numpy()
        c_y_bot = cells_page["y_bottom"].to_numpy()
        c_x_l = cells_page["x_left"].to_numpy()
        c_x_r = cells_page["x_right"].to_numpy()
        c_band = cells_page["horizontal_band_id"].to_numpy() if "horizontal_band_id" in cells_page.columns else None

        # map from cell_id -> global index in `cells`
        id_to_global_idx = dict(zip(c_id, cells_page.index))

        # Iterate lines on this page
        for line_idx, line in lines_page.iterrows():
            line_id = int(line["shape_id"])
            ly_top = float(line["y_top"])
            lx_l = float(line["x_left"])
            lx_r = float(line["x_right"])

            # ===== 1) eligibility: vertical + horizontal =====
            # Vertical: under or through text
            under = ly_top >= c_y_bot
            through = (ly_top >= c_y_top) & (ly_top <= c_y_bot)

            # Horizontal: MUST overlap the line in X
            # (cell_x_right > line_x_left) AND (cell_x_left < line_x_right)
            x_overlap_with_line = (c_x_r > lx_l + _X_OVERLAP_EPS) & (c_x_l < lx_r - _X_OVERLAP_EPS)

            eligible_mask = (under | through) & x_overlap_with_line

            if not np.any(eligible_mask):
                shapes.at[line_idx, "line_role"] = "separator"
                continue

            # ===== 2) closest cells by |ly_top - y_bottom| =====
            eligible_idx = np.where(eligible_mask)[0]
            gaps = np.abs(c_y_bot[eligible_idx] - ly_top)
            min_gap = float(gaps.min())
            closest_mask_local = (gaps == min_gap)
            closest_idx = eligible_idx[closest_mask_local]

            if min_gap > _SEPARATOR_GAP_THRESHOLD:
                shapes.at[line_idx, "line_role"] = "separator"
                continue

            closest_ids = c_id[closest_idx]
            closest_bands = c_band[closest_idx] if c_band is not None else None

            # initial segments (for coverage)
            segments = [(c_x_l[i], c_x_r[i]) for i in closest_idx]
            coverage_pct = _union_coverage(lx_l, lx_r, segments)

            # ===== 3) band cells ABOVE & pruned by earlier lines =====
            band_cells_above_idx = np.array([], dtype=int)
            if c_band is not None:
                seed_bands = np.unique(closest_bands)
                min_seed_cell_id = int(closest_ids.min())

                # same band(s), above closest row(s), AND overlapping this line in X
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
                        el_x_l = earlier_lines["x_left"].to_numpy()
                        el_x_r = earlier_lines["x_right"].to_numpy()
                        el_y_top = earlier_lines["y_top"].to_numpy()

                        keep = []
                        for bi in band_idx:
                            cx_l = c_x_l[bi]
                            cx_r = c_x_r[bi]
                            cy_top = c_y_top[bi]

                            mask_x = (cx_l >= el_x_l) & (cx_r <= el_x_r)
                            mask_y = (cy_top <= el_y_top)
                            if not np.any(mask_x & mask_y):
                                keep.append(bi)
                        band_cells_above_idx = np.array(keep, dtype=int)

            # ===== 4) decide exit mode / expansion =====
            assigned_ids = set(int(x) for x in closest_ids)
            assigned_segments = segments.copy()

            # 4a) Alone in band (no other above candidates) OR
            # 4b) enough coverage already
            if band_cells_above_idx.size == 0 or coverage_pct >= _COVERAGE_THRESHOLD:
                shapes.at[line_idx, "line_role"] = "underline"
            else:
                # 4c) expansion
                shapes.at[line_idx, "line_role"] = "underline"

                cand_idx = band_cells_above_idx
                gap_to_line = np.abs(c_y_bot[cand_idx] - ly_top)
                order = np.argsort(gap_to_line)
                cand_idx = cand_idx[order]

                for bi in cand_idx:
                    cx_l = c_x_l[bi]
                    cx_r = c_x_r[bi]

                    if assigned_segments:
                        ax_l_arr = np.array([seg[0] for seg in assigned_segments])
                        ax_r_arr = np.array([seg[1] for seg in assigned_segments])
                        overlaps_arr = _x_overlap_width(cx_l, cx_r, ax_l_arr, ax_r_arr)
                        if np.any(overlaps_arr > _X_OVERLAP_EPS):
                            continue

                    assigned_segments.append((cx_l, cx_r))
                    assigned_ids.add(int(c_id[bi]))
                    coverage_pct = _union_coverage(lx_l, lx_r, assigned_segments)
                    if coverage_pct >= _COVERAGE_THRESHOLD:
                        break

            # ===== 5) write assignments to cells =====
            if shapes.at[line_idx, "line_role"] == "underline" and assigned_ids:
                for cid_val in assigned_ids:
                    g_idx = id_to_global_idx.get(cid_val)
                    if g_idx is None:
                        continue
                    cells.at[g_idx, "is_underlined"] = True
                    if pd.isna(cells.at[g_idx, "shape_id_underline"]):
                        cells.at[g_idx, "shape_id_underline"] = line_id

    return cells, shapes
