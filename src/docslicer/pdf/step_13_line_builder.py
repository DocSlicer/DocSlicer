"""
step_12_line_builder.py

Cells -> Lines.

Aggregate the per-cell rows from step 09 (the cell builder) into one row per
logical line. A cell already carries a scalar ``line_id`` (the first line it
touches) and a fully merged ``text`` field, so building lines is a single
groupby: join the cell texts left-to-right into a line string and roll the
per-cell geometry up through the shared column registry.

This is deliberately a thin first step. Later stages (paragraph / block
grouping, reading order) consume ``df_lines``; the cross-line decisions live
there, not here.
"""

from __future__ import annotations

import pandas as pd

from .._utils.df_aggregation.registry_aggregator import Agg, aggregate_to
from .._utils.df_aggregation.text_merge import (
    merge_table_rows,
    merge_text_within_line,
)

# is_uppercase fires when >90% of alphabetic chars are uppercase (matches the
# UPPERCASE_THRESHOLD the registry aggregator applies to word-level counts).
_UPPERCASE_THRESHOLD = 0.90


def _line_uppercase(text: pd.Series) -> pd.Series:
    """Bracket-aware uppercase flag per line, fully vectorized.

    The registry aggregator derives is_uppercase from summed word-level
    uppercase_count / alpha_count, which cannot see brackets. Measuring on the
    joined line text lets us drop bracketed spans first, so a trailing
    ``"(Since Aug 5, 2025)"`` does not demote an all-caps heading. Bracket
    removal is done with vectorized regex ``.str.replace`` (no per-row Python),
    so this stays cheap even though it runs on the joined text.

    Note: the alpha/upper counts are ASCII (``[A-Za-z]`` / ``[A-Z]``); non-ASCII
    uppercase (accents, Greek, Cyrillic) is not counted here.
    """
    cleaned = (
        text.fillna("")
        .str.replace(r"\([^)]*\)", "", regex=True)
        .str.replace(r"\[[^\]]*\]", "", regex=True)
        .str.replace(r"\{[^}]*\}", "", regex=True)
    )
    upper = cleaned.str.count(r"[A-Z]")
    alpha = cleaned.str.count(r"[A-Za-z]")
    return ((upper / alpha.where(alpha > 0)).fillna(0.0) > _UPPERCASE_THRESHOLD)


def _build_lines_df(df_cells: pd.DataFrame) -> pd.DataFrame:
    """Aggregate cells into lines via the central column registry.

    Cell ``text`` is already inline-markup'd and de-hyphenated per word, so here
    we only join sibling cells on the same line. Prose lines join with a plain
    space (bullet_sep keeps a cell opening with a bullet glyph on its own visual
    line); lines inside a tagged table (nonblank ``table_id``) join pipe-
    delimited so the column structure survives into ``df_lines``.
    """
    prose_text = merge_text_within_line(
        df_cells["text"], df_cells["line_id"], bullet_sep="\n"
    )
    line_text = prose_text

    # Only pay for the pipe-join when the document actually has tagged cells: on a
    # table-free doc `.any()` short-circuits and we skip the second full string
    # merge entirely. When tables exist, restrict the pipe-join to the tagged
    # cells and splice those rows over the prose default per line_id.
    if "table_id" in df_cells.columns and df_cells["table_id"].notna().any():
        # A line is a table row if any of its cells is tagged; pipe-join *all*
        # cells on such lines (a mixed line must not drop its untagged cells).
        # Restricting the merge to those cells skips the second full string join
        # on the prose majority.
        table_line = df_cells["table_id"].notna().groupby(df_cells["line_id"]).transform("any")
        table_cells = df_cells[table_line]
        table_text = merge_table_rows(table_cells["text"], table_cells["line_id"])
        line_text = prose_text.copy()
        line_text.loc[table_text.index] = table_text

    df_lines = aggregate_to(
        df_cells,
        by="line_id",
        overrides={
            "cell_id": Agg.LIST,
        },
    )
    df_lines = df_lines.rename(columns={"cell_id": "cell_ids"})
    df_lines["cell_count"] = df_lines["cell_ids"].str.len()
    df_lines["text"] = df_lines["line_id"].map(line_text)
    # Override the registry's count-based is_uppercase with the bracket-aware
    # measure on the joined line text (the counts can't exclude bracketed spans).
    df_lines["is_uppercase"] = _line_uppercase(df_lines["text"])
    return df_lines


def build_lines(df_cells: pd.DataFrame) -> pd.DataFrame:
    """
    Cells -> Lines.

    Parameters
    ----------
    df_cells
        One row per cell (output of step 09's ``build_cells``). Must carry a
        scalar ``line_id`` and a merged ``text`` column.

    Returns
    -------
    df_lines
        One row per ``line_id``: joined line ``text``, the list of constituent
        ``cell_ids`` and their count (``cell_count``), and the registry-
        aggregated cell geometry.
    """
    if df_cells is None or df_cells.empty:
        return pd.DataFrame()

    return _build_lines_df(df_cells)
