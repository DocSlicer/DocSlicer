"""Initial DOCX orchestrator surface for debugging the run-level layer."""

from __future__ import annotations

from pathlib import Path
from typing import BinaryIO

import pandas as pd

from .step_01_package_reader import DocxPackage, read_docx_package
from .step_02_run_extractor import extract_runs
from .step_03_table_cell_builder import build_table_cells
from .step_04_paragraph_builder import build_paragraphs


def run_pipeline(
    source: str | Path | bytes | BinaryIO,
    include_headers_footers: bool = True,
    include_notes_comments: bool = True,
) -> tuple[DocxPackage, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Run the initial DOCX extraction slice.

    Returns:
        package: Parsed DOCX package.
        run_df: Run-level DataFrame (one row per text/control/image run event).
        df_table_cells: Cell-level DataFrame (one row per logical table cell).
        df_paragraphs: Paragraph-level DataFrame (text runs aggregated by paragraph_id).
    """
    package = read_docx_package(source)
    run_df = extract_runs(
        package,
        include_headers_footers=include_headers_footers,
        include_notes_comments=include_notes_comments,
    )
    df_table_cells = build_table_cells(
        package,
        run_df,
        include_headers_footers=include_headers_footers,
        include_notes_comments=include_notes_comments,
    )
    df_paragraphs = build_paragraphs(run_df)
    return package, run_df, df_table_cells, df_paragraphs
