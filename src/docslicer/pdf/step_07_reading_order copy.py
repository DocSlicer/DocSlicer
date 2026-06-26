"""
step_07_reading_order.py

Assign line_id to every word in df_words.

Strategy:
  1. If struct_group_id is present (native PDFs with structure tree):
       - Collapse words into a struct-group-level DataFrame.
       - Sort struct groups and call assign_line_id on them.
       - Map line_id back to individual words via struct_group_id.
       - Return (df_words, df_struct_groups).
  2. Otherwise, fall back to the word-level strategy:
       a. Check whether text_object_id is populated (PDF byte-stream order, most accurate).
       b. If YES  → sort horizontal words by (page_number, text_object_id), call assign_line_id.
       c. If NO   → run gutter detection (if not already done), sort via gutter-aware heuristic,
                    call assign_line_id.
       d. Vertical words (TTB / BTT) are processed separately after horizontal words and receive
          line_ids offset above the highest horizontal line_id so they sort to the end of each page.
       - Return (df_words, None).

Public API:
    df_words, df_struct_groups = assign_reading_order(df_words, df_shapes)
"""

from __future__ import annotations

import pandas as pd

from .._utils.line_merger import assign_line_id


# ================================================================================
# GUTTER-BASED SORT  (fallback for PDFs without text_object_id)
# ================================================================================

def _sort_by_gutters(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sort words into correct reading order for mixed single/multi-column pages.

    For singlecol pages a plain (y_top, x_left) sort is sufficient.
    For multicol pages, words in the same gutter zone must be grouped together
    before sorting by y so col-2 text doesn't end up below trailing singlecol text.

    zone_y per word:
      - Singlecol words                   → zone_y = y_top  (their own)
      - Col-1 multicol words              → zone_y = min(y_top) of their gutter zone
      - Col-2+ words                      → zone_y inherited from col-1 via gutter chain
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
# VERTICAL TEXT  (TTB / BTT)
# ================================================================================

def _assign_vertical_line_ids(df_vert: pd.DataFrame, line_id_offset: int) -> pd.DataFrame:
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
# STRUCT GROUP AGGREGATION  (native PDF path)
# ================================================================================

def _build_struct_groups_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse word-level df into one row per (page_number, struct_group_id).

    The bounding box spans all words in the group.  text_object_id takes the
    minimum so groups sort in PDF stream order when sorted by that column.
    """
    agg_dict: dict = {
        "y_top":    ("y_top",    "min"),
        "y_bottom": ("y_bottom", "max"),
        "x_left":   ("x_left",  "min"),
        "x_right":  ("x_right", "max"),
    }
    if "text_object_id" in df.columns:
        agg_dict["text_object_id"] = ("text_object_id", "min")
    if "text_orientation" in df.columns:
        agg_dict["text_orientation"] = ("text_orientation", "first")

    return (
        df.groupby(["page_number", "struct_group_id"], sort=False)
        .agg(**agg_dict)
        .reset_index()
    )


# ================================================================================
# PUBLIC API
# ================================================================================

def assign_reading_order(
    df_words: pd.DataFrame,
    df_shapes: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """
    Add line_id to every word in df_words.

    When struct_group_id is present (native PDFs), line_id is assigned at the
    struct-group level and mapped back to individual words.  This correctly
    handles struct groups that span multiple visual lines.

    Otherwise, line_id is assigned directly on the word-level DataFrame using
    PDF stream order (text_object_id) or a gutter-aware spatial sort.

    In both paths, vertical words (TTB / BTT) are processed separately and
    receive line_ids offset above the horizontal ceiling for each page.

    Parameters
    ----------
    df_words  : word-level DataFrame (output of step_01 through step_06).
    df_shapes : shape DataFrame, only needed if gutter detection has not yet run.

    Returns
    -------
    df_words : word-level DataFrame with line_id (and center_bucket) populated.
    df_struct_groups : struct-group-level DataFrame with one row per group and
        line_id populated, or None when struct_group_id is not available.
    """
    if df_words is None or df_words.empty:
        out = (df_words.copy() if df_words is not None else pd.DataFrame())
        out["line_id"] = pd.Series(dtype="Int64")
        return out, None

    df = df_words.copy()

    has_text_object_id = (
        "text_object_id" in df.columns
        and df["text_object_id"].notna().any()
    )
    has_struct_group = (
        "struct_group_id" in df.columns
        and df["struct_group_id"].notna().any()
    )

    # ── STRUCT GROUP PATH (native PDFs with structure tree) ──────────────────
    if has_struct_group:
        df_sg = _build_struct_groups_df(df)

        if "text_orientation" in df_sg.columns:
            vert_mask_sg = df_sg["text_orientation"].isin(["TTB", "BTT"])
        else:
            vert_mask_sg = pd.Series(False, index=df_sg.index)

        pages = sorted(df_sg["page_number"].unique())
        horiz_sg: list[pd.DataFrame] = []
        vert_sg:  list[pd.DataFrame] = []
        running_line = 0

        for page_num in pages:
            page_mask = df_sg["page_number"] == page_num
            sg_h = df_sg[page_mask & ~vert_mask_sg].copy()
            sg_v = df_sg[page_mask & vert_mask_sg].copy()

            if not sg_h.empty:
                if has_text_object_id and "text_object_id" in sg_h.columns:
                    sg_h = sg_h.sort_values("text_object_id", kind="mergesort").reset_index(drop=True)
                else:
                    sg_h = sg_h.sort_values(["y_top", "x_left"], kind="mergesort").reset_index(drop=True)

                sg_h = assign_line_id(sg_h)
                sg_h["line_id"] = sg_h["line_id"] + running_line
                running_line = int(sg_h["line_id"].max())

            horiz_sg.append(sg_h)

            if not sg_v.empty:
                sg_v = _assign_vertical_line_ids(sg_v, line_id_offset=running_line)
                running_line = int(sg_v["line_id"].max())

            vert_sg.append(sg_v)

        sg_parts = [p for p in horiz_sg + vert_sg if not p.empty]
        if not sg_parts:
            df["line_id"] = pd.Series(dtype="Int64")
            return df, None

        df_sg = (
            pd.concat(sg_parts, ignore_index=True)
            .sort_values(["page_number", "y_top", "x_left"], kind="mergesort")
            .reset_index(drop=True)
        )

        if has_text_object_id:
            df_sg = _assign_stream_groups(df_sg)
            df_sg = _reshuffle_stream_groups(df_sg)

        # Map line_id and center_bucket from struct groups back to words
        sg_id_col = df_sg["struct_group_id"]
        line_id_map    = dict(zip(sg_id_col, df_sg["line_id"]))
        df["line_id"]  = df["struct_group_id"].map(line_id_map).astype("Int64")

        if "center_bucket" in df_sg.columns:
            cb_map = dict(zip(sg_id_col, df_sg["center_bucket"]))
            df["center_bucket"] = df["struct_group_id"].map(cb_map).astype("Int64")

        if "stream_group_id" in df_sg.columns:
            sg_map = dict(zip(sg_id_col, df_sg["stream_group_id"]))
            df["stream_group_id"] = df["struct_group_id"].map(sg_map)

        if "line_y_center" in df_sg.columns:
            lyc_map = dict(zip(sg_id_col, df_sg["line_y_center"]))
            df["line_y_center"] = df["struct_group_id"].map(lyc_map)

        return df, df_sg

    # ── WORD-LEVEL PATH (scanned / non-native PDFs) ──────────────────────────

    # Ensure gutter columns are present for the fallback spatial sort
    if not has_text_object_id:
        gutter_cols_present = (
            "gutter_id_right" in df.columns
            and df["gutter_id_right"].notna().any()
        )
        if not gutter_cols_present:
            from ._utils.gutter_detector import detect_and_annotate_gutters
            df, _, _ = detect_and_annotate_gutters(df, df_shapes or pd.DataFrame())

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
            if has_text_object_id:
                df_h = df_h.sort_values("text_object_id", kind="mergesort").reset_index(drop=True)
            else:
                df_h = _sort_by_gutters(df_h)

            df_h = assign_line_id(df_h)
            df_h["line_id"] = df_h["line_id"] + running_line
            running_line = int(df_h["line_id"].max())

        horiz_pages.append(df_h)

        if not df_v.empty:
            df_v = _assign_vertical_line_ids(df_v, line_id_offset=running_line)
            running_line = int(df_v["line_id"].max())

        vert_pages.append(df_v)

    all_parts = [p for p in horiz_pages + vert_pages if not p.empty]
    if not all_parts:
        df["line_id"] = pd.Series(dtype="Int64")
        return df, None

    result = (
        pd.concat(all_parts, ignore_index=True)
        .sort_values(["page_number", "y_top", "x_left"], kind="mergesort")
        .reset_index(drop=True)
    )