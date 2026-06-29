"""
PPTX paragraph -> line adapter.

Most PPTX paragraphs map 1:1 to a "line" row. The exception is table content:
paragraphs that belong to the same table row are collapsed into one pipe-delimited
line so downstream shared stages see a row-shaped table surface, consistent with
the DOCX and HTML pipelines.

Reading-order layout_id
-----------------------
Shapes are assigned layout_ids in column-first spatial reading order:

1. Group shapes into horizontal bands: shapes whose y-center is within 10 pt
   of each other share a band. Bands are sorted top to bottom.

2. Detect column groups: consecutive bands that have the same number of shapes
   AND whose x-centers align within 10 pt form a column group. These bands
   represent rows of the same multi-column layout (e.g. title row + body row
   of a 4-column section).

3. Assign reading order:
   - Inside a column group: read left-column-first — all shapes in column 0
     (across every band in the group) before column 1, etc.
   - Outside a column group (single-band): read left to right.

4. Assign layout_id sequentially in that order, then re-sequence line_id so
   that both IDs perfectly reflect reading order.
"""

from __future__ import annotations

import pandas as pd

from .._utils.df_aggregation.hierarchical_aggregator import (
    _collect_unique_list,
    aggregate_hierarchical,
    build_standard_agg_spec,
)


_PPTX_LINE_IDENTITY_COLS = [
    "page_number",
    "slide_index",
    "header_footer_type",
    "source_part",
    "source_part_id",
    "text_align",
    "list_num_id",
    "list_level",
    "list_label",
    "outline_level",
    "shape_id",
    "shape_name",
    "shape_type",
    "placeholder_type",
    "block_type",
]


def _has_value(value: object) -> bool:
    if isinstance(value, (list, tuple, set)):
        return any(_has_value(item) for item in value)
    return value is not None and not pd.isna(value) and str(value).strip() != ""


def _add_line_group_key(df: pd.DataFrame) -> pd.DataFrame:
    """Assign one group per paragraph, except one group per table row."""
    out = df.copy()

    has_table_row = (
        out.get("table_id", pd.Series(pd.NA, index=out.index)).map(_has_value)
        & out.get("table_row_id", pd.Series(pd.NA, index=out.index)).map(_has_value)
    )

    paragraph_id = out.get("paragraph_id", pd.Series(range(1, len(out) + 1), index=out.index))
    table_id = out.get("table_id", pd.Series(pd.NA, index=out.index))
    table_row_id = out.get("table_row_id", pd.Series(pd.NA, index=out.index))

    out["_line_group_key"] = "p:" + paragraph_id.astype(str)
    out.loc[has_table_row, "_line_group_key"] = (
        "t:"
        + table_id.loc[has_table_row].astype(str)
        + ":r:"
        + table_row_id.loc[has_table_row].astype(str)
    )
    return out


def _create_line_text(df: pd.DataFrame) -> dict[str, str]:
    """Build text for each line group, pipe-delimiting table-row cells."""
    if "_line_group_key" not in df.columns or "text" not in df.columns:
        return {}

    text_map: dict[str, str] = {}
    working = df.copy()
    working["text"] = working["text"].fillna("").astype(str).str.strip()

    for group_key, group in working.groupby("_line_group_key", sort=False):
        table_id = group["table_id"].iloc[0] if "table_id" in group.columns else None
        has_table = _has_value(table_id)

        if not has_table:
            text_map[group_key] = " ".join(t for t in group["text"].tolist() if t).strip()
            continue

        sort_cols = [col for col in ["table_cell_id", "paragraph_id"] if col in group.columns]
        ordered = group.sort_values(sort_cols, kind="mergesort") if sort_cols else group

        if "table_cell_id" in ordered.columns:
            cell_texts = (
                ordered.groupby("table_cell_id", sort=False)["text"]
                .agg(lambda parts: " ".join(p for p in parts if p).strip())
                .tolist()
            )
        else:
            cell_texts = ordered["text"].tolist()

        text_map[group_key] = " | ".join(t for t in cell_texts if t).strip()

    return text_map


def _cluster_bands(shapes: pd.DataFrame, tol: float = 10.0) -> list[list[int]]:
    """
    Group shape_ids into horizontal bands by y_center proximity.

    Shapes are sorted by y_center. A new band starts whenever the next shape's
    y_center is more than `tol` pts away from the first shape in the current band.
    """
    if shapes.empty:
        return []
    ordered = shapes.sort_values("y_center").reset_index(drop=True)
    bands: list[list[int]] = []
    current: list[int] = []
    band_y: float | None = None
    for _, row in ordered.iterrows():
        y = float(row["y_center"])
        sid = int(row["shape_id"])
        if band_y is None or abs(y - band_y) <= tol:
            current.append(sid)
            if band_y is None:
                band_y = y
        else:
            bands.append(current)
            current = [sid]
            band_y = y
    if current:
        bands.append(current)
    return bands


def _band_x_profile(band: list[int], geom: dict[int, dict]) -> list[float]:
    """Sorted x_center values for a band — its column fingerprint."""
    return sorted(geom[sid]["x_center"] for sid in band)


def _profiles_match(a: list[float], b: list[float], tol: float = 10.0) -> bool:
    """True when two bands have the same count and each x_center pair aligns within tol."""
    if len(a) != len(b):
        return False
    return all(abs(xa - xb) <= tol for xa, xb in zip(a, b))


def _reading_order_for_slide(bands: list[list[int]], geom: dict[int, dict]) -> list[int]:
    """
    Flatten bands into a shape_id reading order.

    Consecutive bands with matching x-profiles become a column group and are
    read left-column-first. All other bands are read left-to-right.
    """
    profiles = [_band_x_profile(b, geom) for b in bands]

    # Identify runs of consecutive matching bands.
    groups: list[list[int]] = []  # list of band-index lists
    i = 0
    while i < len(bands):
        grp = [i]
        j = i + 1
        while j < len(bands) and _profiles_match(profiles[j - 1], profiles[j]):
            grp.append(j)
            j += 1
        groups.append(grp)
        i = j

    result: list[int] = []
    for grp in groups:
        if len(grp) == 1:
            # Single band — read left to right.
            result.extend(sorted(bands[grp[0]], key=lambda sid: geom[sid]["x_center"]))
        else:
            # Column group — read left-column-first, top-to-bottom within column.
            n_cols = len(bands[grp[0]])
            col_slots: list[list[int]] = [[] for _ in range(n_cols)]
            for band_idx in grp:
                band_sorted = sorted(bands[band_idx], key=lambda sid: geom[sid]["x_center"])
                for col_idx, sid in enumerate(band_sorted):
                    if col_idx < n_cols:
                        col_slots[col_idx].append(sid)
            for col in col_slots:
                result.extend(col)

    return result


def _assign_reading_order_layout_ids(lines_df: pd.DataFrame) -> pd.DataFrame:
    """
    Assign layout_id and re-sequence line_id in reading order.

    See module docstring for the full algorithm description.
    """
    if lines_df.empty or "shape_id" not in lines_df.columns:
        return lines_df

    needs = {"x_left", "x_right", "y_top", "y_bottom"}
    if not needs.issubset(lines_df.columns):
        out = lines_df.copy()
        out["layout_id"] = out["shape_id"]
        return out

    group_cols = [c for c in ("page_number", "slide_index") if c in lines_df.columns]

    shape_geom_df = (
        lines_df.dropna(subset=["shape_id"])
        .groupby(group_cols + ["shape_id"], sort=False)[["x_left", "x_right", "y_top", "y_bottom"]]
        .first()
        .reset_index()
    )
    shape_geom_df["x_center"] = (shape_geom_df["x_left"] + shape_geom_df["x_right"]) / 2
    shape_geom_df["y_center"] = (shape_geom_df["y_top"] + shape_geom_df["y_bottom"]) / 2

    layout_counter = 0
    layout_rows: list[dict] = []

    for key, slide_df in shape_geom_df.groupby(group_cols, sort=True):
        key_vals = key if isinstance(key, tuple) else (key,)
        geom: dict[int, dict] = {
            int(row["shape_id"]): {"x_center": float(row["x_center"]), "y_center": float(row["y_center"])}
            for _, row in slide_df.iterrows()
        }
        bands = _cluster_bands(slide_df, tol=10.0)
        # Record which band each shape belongs to (1-based, top-down).
        band_id_for: dict[int, int] = {
            sid: band_idx + 1
            for band_idx, band in enumerate(bands)
            for sid in band
        }
        for sid in _reading_order_for_slide(bands, geom):
            layout_counter += 1
            layout_rows.append(
                {c: v for c, v in zip(group_cols, key_vals)}
                | {"shape_id": sid, "layout_id": layout_counter, "horizontal_band_id": band_id_for.get(sid)}
            )

    rank_df = pd.DataFrame(layout_rows)
    result = lines_df.drop(columns=["layout_id"], errors="ignore").merge(
        rank_df, on=group_cols + ["shape_id"], how="left"
    )

    sort_cols = group_cols + ["layout_id", "line_id"]
    result = result.sort_values(sort_cols).reset_index(drop=True)
    result["line_id"] = range(1, len(result) + 1)
    return result


def build_lines(paragraph_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build PPTX lines from paragraph rows.

    Non-table paragraphs are one line each. Paragraphs inside the same table row
    are aggregated into one line with pipe-delimited cell text.
    """
    if paragraph_df is None or paragraph_df.empty:
        return pd.DataFrame()

    working = _add_line_group_key(paragraph_df)
    working["_paragraph_count_marker"] = 1
    line_text_map = _create_line_text(working)

    agg_spec = build_standard_agg_spec(
        identity_cols=_PPTX_LINE_IDENTITY_COLS,
        include_geometry=True,
        include_hierarchy=False,
        include_style=True,
        include_counts=True,
        include_metadata=True,
        include_table=True,
        extra_agg={
            "paragraph_id": _collect_unique_list,
            "table_cell_id": _collect_unique_list,
            "hyperlink_url": _collect_unique_list,
            "chart_id": "first",
        },
        count_col="_paragraph_count_marker",
    )

    lines_df = aggregate_hierarchical(
        df=working,
        group_col="_line_group_key",
        agg_spec=agg_spec,
        rename_count_col={"_paragraph_count_marker": "paragraph_count"},
        compute_derived=True,
    )

    lines_df["text"] = lines_df["_line_group_key"].map(line_text_map)
    lines_df.insert(0, "line_id", range(1, len(lines_df) + 1))
    lines_df = lines_df.drop(columns=["_line_group_key"])

    lines_df = _assign_reading_order_layout_ids(lines_df)

    return lines_df


__all__ = ["build_lines"]
