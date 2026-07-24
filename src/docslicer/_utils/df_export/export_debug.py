# SPDX-FileCopyrightText: 2026 Market Framer Inc.
# SPDX-License-Identifier: AGPL-3.0-only
"""
DataFrame export utilities for debug and production outputs.

Provides standardized DataFrame preprocessing before export:
- Column reordering
- Float rounding
- Excel formula escaping
- Production column filtering
"""

import pandas as pd
import logging

_log = logging.getLogger(__name__)

from .reorder_columns import reorder_columns


def export_debug(df: pd.DataFrame, drop_none: bool = False) -> pd.DataFrame:
    """
    Prepare DataFrame for debug export (CSV inspection).
    
    Operations:
    - Reorders columns by priority (identity, geometry, content, etc.)
    - Converts object columns with numeric values to float (handles pd.NA columns)
    - Rounds all float columns to 2 decimals (preserves ints)
    - Escapes Excel formula characters in text fields
    - Optionally drops None-only columns
    
    Args:
        df: Input DataFrame
        drop_none: If True, drops columns that are all None/NaN
    
    Returns:
        Processed DataFrame ready for export
    """
    if df.empty:
        return df

    df = df.copy()
    
    # Drop None-only columns if requested
    if drop_none:
        df = df.dropna(axis=1, how='all')
    
    # Reorder columns for better readability
    df = reorder_columns(df)
    
    # Convert object columns containing numeric values to float
    # (handles columns initialized with pd.NA that later get numeric values)
    for col in df.columns:
        try:
            if df[col].dtype == 'object':
                # Try to convert to numeric, coerce errors to NaN
                numeric_col = pd.to_numeric(df[col], errors='coerce')
                # If all non-null values successfully converted, use numeric version
                if numeric_col.notna().sum() == df[col].notna().sum():
                    df[col] = numeric_col
        except (AttributeError, KeyError):
            # Skip columns that can't be accessed normally
            continue
    
    # Round all float columns to 2 decimals (keeps ints as ints).
    # Use positional indexing to handle DataFrames with duplicate column names.
    float_dtypes = {'float64', 'float32', 'Float64', 'Float32'}
    for i, dt in enumerate(df.dtypes):
        if str(dt) in float_dtypes:
            df.iloc[:, i] = df.iloc[:, i].round(2)

    # Escape Excel-style formulas in text column
    if "text" in df.columns:
        df["text"] = (
            df["text"]
            .fillna("")
            .astype(str)
            .apply(lambda s: "'" + s if s.startswith(("+", "-", "=")) else s)
        )

    return df


