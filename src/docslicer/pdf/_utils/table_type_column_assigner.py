from __future__ import annotations

from typing import Literal, Sequence, Dict, Any
import re

import pandas as pd
import numpy as np

TableType = Literal["standard", "matrix", "narrative"]

# =============================
# Numeric heuristics
# =============================

_NUMERIC_RE = re.compile(
    r"""
    ^\s*
    [\(\[]?                    # optional opening bracket
    [+\--–—]?                  # optional sign
    \$?                        # optional dollar
    \d{1,3}(?:,\d{3})*|\d+     # integer with optional thousands OR plain integer
    (?:\.\d+)?                 # optional decimal part
    %?                         # optional percentage
    [\)\]]?                    # optional closing bracket
    \s*$
    """,
    re.VERBOSE,
)


def _is_numeric_cell(text: Any) -> bool:
    """
    Treat as numeric only if:
    - it's a string with no alphabetic characters, and
    - it matches a 'number-like' regex (incl $, commas, %, brackets).
    """
    if not isinstance(text, str):
        return False
    stripped = text.strip()
    if not stripped:
        return False
    if any(ch.isalpha() for ch in stripped):
        return False
    return bool(_NUMERIC_RE.match(stripped))


# =============================
# Metrics computation
# =============================

def _compute_table_metrics(table_df: pd.DataFrame) -> Dict[str, float]:
    """
    Compute the features needed to classify one table.
    Assumes table_df is a single logical table.
    """
    n_cells = len(table_df)
    if n_cells == 0:
        return {
            "numeric_density": 0.0,
            "median_cells_per_line": 0.0,
            "median_cells_per_line_ratio": 0.0,
            "numeric_cols_ratio": 0.0,
            "band_total_cols": 0,
            "cells_per_col_cv": 0.0,
            "avg_table_row_score": 0.0,
        }

    # band_total_cols: take the mode, fall back to max if needed
    band_total_mode = table_df["band_total_cols"].mode()
    if len(band_total_mode) > 0:
        band_total_cols = int(band_total_mode.iloc[0])
    else:
        band_total_cols = int(table_df["band_total_cols"].max())

    # numeric density
    numeric_mask = table_df["text"].map(_is_numeric_cell)
    numeric_density = float(numeric_mask.mean())

    # median cells per temp_line_id
    per_line_counts = table_df.groupby("temp_line_id")["cell_id"].count()
    median_cells_per_line = float(per_line_counts.median())
    median_cells_per_line_ratio = (
        median_cells_per_line / band_total_cols if band_total_cols > 0 else 0.0
    )

    # numeric columns: for each col_start, % numeric cells
    per_col = table_df.groupby("col_start")
    per_col_counts = per_col["cell_id"].count()
    per_col_numeric_counts = per_col["text"].apply(
        lambda s: sum(_is_numeric_cell(t) for t in s)
    )
    numeric_cols_mask = per_col_numeric_counts / per_col_counts >= 0.5
    numeric_cols_count = int(numeric_cols_mask.sum())
    numeric_cols_ratio = (
        numeric_cols_count / band_total_cols if band_total_cols > 0 else 0.0
    )

    # cells per column imbalance (coefficient of variation)
    if len(per_col_counts) > 1 and per_col_counts.mean() != 0:
        cells_per_col_cv = float(per_col_counts.std(ddof=0) / per_col_counts.mean())
    else:
        cells_per_col_cv = 0.0

    # average table score: use average_table_score if present, else table_row_score
    if "average_table_score" in table_df.columns:
        avg_table_row_score = float(table_df["average_table_score"].mean())
    else:
        avg_table_row_score = float(table_df["table_row_score"].mean())

    return {
        "numeric_density": numeric_density,
        "median_cells_per_line": median_cells_per_line,
        "median_cells_per_line_ratio": median_cells_per_line_ratio,
        "numeric_cols_ratio": numeric_cols_ratio,
        "band_total_cols": band_total_cols,
        "cells_per_col_cv": cells_per_col_cv,
        "avg_table_row_score": avg_table_row_score,
    }


# =============================
# Scorecard logic
# =============================

def _score_table_types(metrics: Dict[str, float]) -> Dict[TableType, int]:
    """
    Apply the 6 KPI-based rules to assign points to:
    - 'standard'
    - 'matrix'
    - 'narrative'
    """
    num_density = metrics["numeric_density"]
    median_ratio = metrics["median_cells_per_line_ratio"]
    num_cols_ratio = metrics["numeric_cols_ratio"]
    band_cols = metrics["band_total_cols"]
    cv_cols = metrics["cells_per_col_cv"]
    avg_score = metrics["avg_table_row_score"]

    scores: Dict[TableType, int] = {
        "standard": 0,
        "matrix": 0,
        "narrative": 0,
    }

    # -----------------------------
    # KPI 1: Numeric density
    # -----------------------------
    if num_density >= 0.60:
        # Very numeric → standard
        scores["standard"] += 3
    elif 0.30 <= num_density < 0.60:
        # Moderately numeric → light bias to standard
        scores["standard"] += 1
    elif num_density <= 0.10:
        # Almost no numbers → text tables (matrix or narrative)
        scores["matrix"] += 2
        scores["narrative"] += 2

    # -----------------------------
    # KPI 2: Median cells per line vs band_total_cols
    # -----------------------------
    # ratio = median_cells_per_line / band_total_cols
    ratio = median_ratio

    if ratio >= 0.9:
        # Almost every line fills all columns → very matrix-like, also standard-like
        scores["standard"] += 2
        scores["matrix"] += 3
    elif 0.7 <= ratio < 0.9:
        scores["standard"] += 2
        scores["matrix"] += 2
    elif ratio <= 0.5:
        # Lots of lines only use 1–2 columns → narrative
        scores["narrative"] += 3

    # -----------------------------
    # KPI 3: Columns with >50% numbers
    # -----------------------------
    if num_cols_ratio >= 0.5:
        scores["standard"] += 3
    elif 0.2 <= num_cols_ratio < 0.5:
        scores["standard"] += 1
    elif num_cols_ratio <= 0.1:
        scores["matrix"] += 1
        scores["narrative"] += 1

    # -----------------------------
    # KPI 4: Number of columns
    # -----------------------------
    if band_cols >= 6:
        scores["standard"] += 2
    elif 2 <= band_cols <= 4:
        # Shape is compatible with matrix or narrative
        scores["matrix"] += 1
        scores["narrative"] += 1

    # -----------------------------
    # KPI 5: Cells per column balance (CV)
    # -----------------------------
    if cv_cols <= 0.3:
        # Very balanced → strong matrix signal, slight standard
        scores["standard"] += 1
        scores["matrix"] += 3
    elif 0.3 < cv_cols < 0.7:
        # Middling imbalance → doesn't discriminate much
        scores["standard"] += 1
        scores["matrix"] += 1
        scores["narrative"] += 1
    else:  # cv_cols >= 0.7
        # Very imbalanced → narrative (e.g. 1 column with 28 cells, others 4–5)
        scores["narrative"] += 3

    # -----------------------------
    # KPI 6: Average table_row_score
    # -----------------------------
    if avg_score >= 4.0:
        scores["standard"] += 2
    else:
        scores["matrix"] += 1
        scores["narrative"] += 1

    return scores


def _infer_table_type_from_scores(metrics: Dict[str, float]) -> TableType:
    """
    Wrapper: run the scorecard, then apply tie-breaking.
    """
    scores = _score_table_types(metrics)

    # Pick highest score
    max_score = max(scores.values())
    candidates = [t for t, s in scores.items() if s == max_score]

    if len(candidates) == 1:
        return candidates[0]

    # Tie-breaking:
    num_density = metrics["numeric_density"]
    ratio = metrics["median_cells_per_line_ratio"]
    cv_cols = metrics["cells_per_col_cv"]

    # 1) Numeric-ish → standard
    if num_density >= 0.4:
        return "standard"

    # 2) Lines full + balanced → matrix
    if ratio >= 0.8 and cv_cols <= 0.4:
        return "matrix"

    # 3) Otherwise → narrative
    return "narrative"


# =============================
# Public API
# =============================

def add_table_type_column(
    df: pd.DataFrame,
    group_cols: Sequence[str] | None = None,
    table_type_col: str = "table_type",
) -> pd.DataFrame:
    """
    Add a 'table_type' column to df with values: 'standard' | 'matrix' | 'narrative'.

    Parameters
    ----------
    df : DataFrame
        Cell-level table data. Must have at least:
        - 'text'
        - 'temp_line_id'
        - 'cell_id'
        - 'col_start'
        - 'band_total_cols'
        - 'table_row_score' or 'average_table_score'
    group_cols : sequence of str or None
        Columns that uniquely identify a logical table.
        If None, the entire df is treated as one table.
    table_type_col : str
        Name of the output column.

    Returns
    -------
    DataFrame
        Copy of df with an extra 'table_type' column.
    """
    def _classify_group(g: pd.DataFrame) -> pd.DataFrame:
        metrics = _compute_table_metrics(g)
        ttype = _infer_table_type_from_scores(metrics)
        g = g.copy()
        g[table_type_col] = ttype
        return g

    if group_cols is None:
        return _classify_group(df)

    return (
        df.groupby(list(group_cols), group_keys=False)
          .apply(_classify_group)
          .reset_index(drop=True)
    )
