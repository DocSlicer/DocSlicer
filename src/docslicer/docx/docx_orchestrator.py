"""DOCX pipeline entry point — runs all extraction steps and returns structured data."""

from __future__ import annotations

from pathlib import Path
from typing import BinaryIO, NamedTuple

import pandas as pd

from .._utils.config import load_compiled_page_label_config
from .step_01_package_reader import DocxPackage, read_docx_package
from .step_02_run_extractor import extract_runs
from .step_03_table_cell_builder import build_table_cells
from .step_04_paragraph_builder import build_paragraphs
from .step_05_line_builder import build_lines
from .step_06_style_prefiller import prefill_block_types


class DocxPipelineResult(NamedTuple):
    """Structured result of :func:`run_pipeline`."""

    package: DocxPackage
    df_runs: pd.DataFrame
    df_table_cells: pd.DataFrame
    df_paragraphs: pd.DataFrame
    df_lines: pd.DataFrame


def run_pipeline(
    source: str | Path | bytes | BinaryIO,
    include_headers_footers: bool = True,
    include_notes_comments: bool = True,
) -> DocxPipelineResult:
    """
    Run the full DOCX extraction pipeline.

    Args:
        source: File path, raw .docx bytes, or a binary file-like object.
        include_headers_footers: Include header and footer parts.
        include_notes_comments: Include footnotes, endnotes, and comments.

    Returns:
        DocxPipelineResult with fields:
            package: Parsed DOCX package.
            df_runs: Run-level DataFrame (one row per text/control/image run event).
            df_table_cells: Cell-level DataFrame (one row per logical table cell).
            df_paragraphs: Paragraph-level DataFrame (runs aggregated by paragraph_id).
            df_lines: Shared-compatible line DataFrame, ready for shared/ steps.
    """
    package = read_docx_package(source)
    page_label_config = load_compiled_page_label_config()
    df_runs = extract_runs(
        package,
        include_headers_footers=include_headers_footers,
        include_notes_comments=include_notes_comments,
        page_label_config=page_label_config,
    )
    df_table_cells = build_table_cells(
        package,
        df_runs,
        include_headers_footers=include_headers_footers,
        include_notes_comments=include_notes_comments,
    )
    df_paragraphs = build_paragraphs(df_runs)
    df_lines = build_lines(df_paragraphs)
    df_lines = prefill_block_types(df_lines)
    return DocxPipelineResult(
        package=package,
        df_runs=df_runs,
        df_table_cells=df_table_cells,
        df_paragraphs=df_paragraphs,
        df_lines=df_lines,
    )
