"""PPTX pipeline entry point — runs extraction steps and returns structured data."""

from __future__ import annotations

from pathlib import Path
from typing import BinaryIO, NamedTuple

import pandas as pd

from .step_01_package_reader import PptxPackage, read_pptx_package
from .step_02_run_extractor import extract_runs
from .step_03_chart_point_builder import extract_chart_points
from .step_04_table_cell_builder import build_table_cells
from .step_05_paragraph_builder import build_paragraphs
from .step_06_reading_order import assign_reading_order
from .step_07_line_builder import build_lines
from .step_08_style_prefiller import prefill_styles


class PptxPipelineResult(NamedTuple):
    """Structured result of :func:`run_pipeline`."""

    package: PptxPackage
    df_runs: pd.DataFrame
    df_chart_points: pd.DataFrame
    df_table_cells: pd.DataFrame
    df_paragraphs: pd.DataFrame
    df_lines: pd.DataFrame


def run_pipeline(
    source: str | Path | bytes | BinaryIO,
    include_notes: bool = True,
) -> PptxPipelineResult:
    """
    Run the full PPTX extraction pipeline.

    Args:
        source: File path, raw .pptx bytes, or a binary file-like object.
        include_notes: Include speaker notes.

    Returns:
        PptxPipelineResult with run, chart-point, table-cell, paragraph, and
        shared-compatible line DataFrames.
    """
    package = read_pptx_package(source)
    df_runs = extract_runs(package, include_notes=include_notes)
    df_chart_points = extract_chart_points(package, df_runs)
    df_table_cells = build_table_cells(package, df_runs, include_notes=include_notes)
    df_paragraphs = build_paragraphs(df_runs, include_notes=include_notes)
    df_paragraphs = assign_reading_order(df_paragraphs, df_runs)
    df_lines = build_lines(df_paragraphs)
    df_lines = prefill_styles(df_lines)

    return PptxPipelineResult(
        package=package,
        df_runs=df_runs,
        df_chart_points=df_chart_points,
        df_table_cells=df_table_cells,
        df_paragraphs=df_paragraphs,
        df_lines=df_lines,
    )
