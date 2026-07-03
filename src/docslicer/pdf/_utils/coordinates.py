"""Coordinate transformation utilities for PDF cells."""
import logging

import pandas as pd

logger = logging.getLogger(__name__)


def convert_to_global_y_coordinates(df_cells: pd.DataFrame, overwrite: bool = False) -> pd.DataFrame:
    """
    Convert page-relative Y coordinates to document-global Y coordinates.

    For multi-page documents, Y coordinates reset to 0 on each page.
    This causes issues when aggregating chunks that span page boundaries.

    Args:
        df_cells: Cells dataframe with page_number, page_height, y_top, y_bottom.
        overwrite: If True (default), saves original page-relative coords to
            y_top_local / y_bottom_local, then overwrites y_top / y_bottom with
            global coordinates. If False, leaves y_top / y_bottom untouched and
            instead adds new y_top_global / y_bottom_global columns.
    """
    if df_cells.empty:
        return df_cells

    required_cols = ["page_number", "page_height", "y_top", "y_bottom"]
    if not all(col in df_cells.columns for col in required_cols):
        logger.warning("Missing columns for Y coordinate conversion, skipping")
        return df_cells

    page_heights = df_cells.groupby("page_number")["page_height"].first().sort_index()
    cumulative_offsets = page_heights.shift(1, fill_value=0).cumsum()
    offsets = df_cells["page_number"].map(cumulative_offsets)

    if overwrite:
        df_cells["y_top_local"] = df_cells["y_top"]
        df_cells["y_bottom_local"] = df_cells["y_bottom"]
        df_cells["y_top"] = df_cells["y_top_local"] + offsets
        df_cells["y_bottom"] = df_cells["y_bottom_local"] + offsets
    else:
        df_cells["y_top_global"] = df_cells["y_top"] + offsets
        df_cells["y_bottom_global"] = df_cells["y_bottom"] + offsets

    return df_cells
