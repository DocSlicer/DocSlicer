"""
Gutter-based reading order: fallback for sources without native stream order.

Used by the OCR pipeline (sole method) and the PDF pipeline (when text_object_id
is absent).

Public API:
    df_words = assign_reading_order(df_words, df_shapes)
    df_vert  = assign_vertical_line_ids(df_vert, line_id_offset)
"""

from __future__ import annotations

import pandas as pd

from .line_merger import assign_line_id


# ================================================================================
# GUTTER-BASED SORT
# ================================================================================

def _sort_by_gutters(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sort words into correct reading order for mixed single/multi-column pages.

    For singlecol pages a plain (y_top, x_left) sort is sufficient.
    For multicol pages, words in the same gutter zone must be grouped together
    before sorting by y so col-2 text doesn't end up below trailing singlecol text.

    zone_y per word:
      - Singlecol words       → zone_y = y_top  (their own)
      - Col-1 multicol words  → zone_y = min(y_top) of their gutter zone
      - Col-2+ words          → zone_y inherited from col-1 via gutter chain
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

        # Propagate zone_y through 3+ column chains
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

    return (
        df.sort_values(
            ["page_number", "_zone_y", "_rc", "y_top", "x_left"],
            kind="mergesort",
        )
        .drop(columns=["_zone_y", "_rc"])
        .reset_index(drop=True)
    )


# ================================================================================
# VERTICAL TEXT  (TTB / BTT)
# ================================================================================

def assign_vertical_line_ids(df_vert: pd.DataFrame, line_id_offset: int) -> pd.DataFrame:
    """
    Assign line_ids to vertical (TTB/BTT) words using a coordinate-swap trick.

    Words sharing the same x-band (a column of rotated text) become a "line".
    Swap x↔y so assign_line_id groups by x-proximity, then swap back.
    All IDs are offset above the horizontal line_id ceiling.
    """
    if df_vert.empty:
        df_vert = df_vert.copy()
        df_vert["line_id"] = pd.Series(dtype="Int64")
        return df_vert

    df = df_vert.copy()

    orig_xl = df["x_left"].to_numpy(dtype=float)
    orig_xr = df["x_right"].to_numpy(dtype=float)
    orig_yt = df["y_top"].to_numpy(dtype=float)
    orig_yb = df["y_bottom"].to_numpy(dtype=float)

    df["y_top"]    = orig_xl
    df["y_bottom"] = orig_xr
    df["x_left"]   = orig_yt
    df["x_right"]  = orig_yb

    # BTT: negate x so ascending sort yields bottom→top order
    df["_x_left_pre_neg"]  = df["x_left"]
    df["_x_right_pre_neg"] = df["x_right"]
    if "text_orientation" in df.columns:
        btt_mask = df["text_orientation"] == "BTT"
        if btt_mask.any():
            btt_xl = df.loc[btt_mask, "x_left"].to_numpy(dtype=float)
            btt_xr = df.loc[btt_mask, "x_right"].to_numpy(dtype=float)
            df.loc[btt_mask, "x_left"]  = -btt_xr
            df.loc[btt_mask, "x_right"] = -btt_xl

    df = df.sort_values(["page_number", "y_top", "x_left"], kind="mergesort").reset_index(drop=True)
    df = assign_line_id(df)

    # Restore coordinates
    df["x_left"]  = df["_x_left_pre_neg"]
    df["x_right"] = df["_x_right_pre_neg"]
    df.drop(columns=["_x_left_pre_neg", "_x_right_pre_neg"], inplace=True)

    cur_xl = df["x_left"].to_numpy(dtype=float)
    cur_xr = df["x_right"].to_numpy(dtype=float)
    cur_yt = df["y_top"].to_numpy(dtype=float)
    cur_yb = df["y_bottom"].to_numpy(dtype=float)
    df["x_left"]   = cur_yt
    df["x_right"]  = cur_yb
    df["y_top"]    = cur_xl
    df["y_bottom"] = cur_xr

    df["line_id"] = df["line_id"] + line_id_offset
    return df


# ================================================================================
# PUBLIC API
# ================================================================================

def assign_reading_order(
    df_words: pd.DataFrame,
    df_shapes: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Assign line_id to every word using gutter-aware spatial sorting.

    Runs gutter detection first if gutter columns are not already present,
    then sorts via _sort_by_gutters and assigns line_ids sequentially.

    Parameters
    ----------
    df_words  : word-level DataFrame with at least page_number, x_left, y_top.
    df_shapes : shape DataFrame passed to gutter detection when needed.

    Returns
    -------
    df_words with line_id (and center_bucket) populated.
    """
    if df_words is None or df_words.empty:
        out = (df_words.copy() if df_words is not None else pd.DataFrame())
        out["line_id"] = pd.Series(dtype="Int64")
        return out

    df = df_words.copy()

    gutter_cols_present = (
        "gutter_id_right" in df.columns
        and df["gutter_id_right"].notna().any()
    )
    if not gutter_cols_present:
        from .gutter_detector import detect_and_annotate_gutters
        df, _, _ = detect_and_annotate_gutters(
            df, df_shapes if df_shapes is not None else pd.DataFrame()
        )

    # Split horizontal (LTR/default) vs vertical (TTB/BTT)
    if "text_orientation" in df.columns:
        vert_mask = df["text_orientation"].isin(["TTB", "BTT"])
    else:
        vert_mask = pd.Series(False, index=df.index)

    pages = sorted(df["page_number"].unique())
    horiz_pages: list[pd.DataFrame] = []
    vert_pages:  list[pd.DataFrame] = []
    running_line = 0

    for page_num in pages:
        page_mask = df["page_number"] == page_num
        df_h = df[page_mask & ~vert_mask].copy()
        df_v = df[page_mask & vert_mask].copy()

        if not df_h.empty:
            df_h = _sort_by_gutters(df_h)
            df_h = assign_line_id(df_h)
            df_h["line_id"] = df_h["line_id"] + running_line
            running_line = int(df_h["line_id"].max())

        horiz_pages.append(df_h)

        if not df_v.empty:
            df_v = assign_vertical_line_ids(df_v, line_id_offset=running_line)
            running_line = int(df_v["line_id"].max())

        vert_pages.append(df_v)

    all_parts = horiz_pages + vert_pages
    non_empty = [p for p in all_parts if not p.empty]
    if not non_empty:
        df["line_id"] = pd.Series(dtype="Int64")
        return df

    return (
        pd.concat(non_empty, ignore_index=True)
        .sort_values(["page_number", "y_top", "x_left"], kind="mergesort")
        .reset_index(drop=True)
    )
