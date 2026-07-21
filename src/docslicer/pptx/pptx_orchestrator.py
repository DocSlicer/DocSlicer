"""PPTX pipeline entry point — runs extraction steps and returns structured data."""

from __future__ import annotations

import logging
import zipfile
from pathlib import Path
from typing import Any, BinaryIO, Callable, Dict, NamedTuple, Optional

import pandas as pd

from .._utils.password import decrypt_office, is_encrypted_office
from .._utils.safe_call import safe_enrich
from .._utils.timing import timed_step
from ..metadata import add_text_fallbacks, consolidate
from .native_metadata import extract_native_metadata
from .step_01_package_reader import PptxPackage, read_pptx_package
from .step_02_run_extractor import extract_runs
from .step_03_chart_point_builder import extract_chart_points
from .step_04_table_cell_builder import build_table_cells
from .step_05_paragraph_builder import build_paragraphs
from .step_06_reading_order import assign_reading_order
from .step_07_line_builder import build_lines
from .step_08_style_prefiller import prefill_styles

logger = logging.getLogger(__name__)


class PptxPipelineResult(NamedTuple):
    """Structured result of :func:`run_pipeline`."""

    discovered_metadata: Dict[str, Any]
    df_runs: pd.DataFrame
    df_chart_points: pd.DataFrame
    df_table_cells: pd.DataFrame
    df_paragraphs: pd.DataFrame
    df_lines: pd.DataFrame


def run_pipeline(
    source: str | Path | bytes | BinaryIO,
    include_speaker_notes: bool = True,
    password: str | None = None,
    source_filename: str | None = None,
    on_stage: Optional[Callable[[str], None]] = None,
) -> PptxPipelineResult:
    """
    Run the full PPTX extraction pipeline.

    Args:
        source: File path, raw .pptx bytes, or a binary file-like object.
        include_speaker_notes: Include speaker notes.
        on_stage: Optional callback for progress updates.

    Returns:
        PptxPipelineResult with resolved discovered_metadata plus run,
        chart-point, table-cell, paragraph, and shared-compatible line DataFrames.
    """
    if on_stage:
        on_stage("extract_elements")

    # Read the package; an encrypted .pptx surfaces as a BadZipFile we decrypt.
    _is_password_protected = False
    with timed_step("package_reading", logger=logger):
        try:
            package = read_pptx_package(source)
        except zipfile.BadZipFile:
            if not (isinstance(source, bytes) and is_encrypted_office(source)):
                raise
            source = decrypt_office(source, password, source_filename)
            package = read_pptx_package(source)
            _is_password_protected = True

    with timed_step("run_extraction", logger=logger):
        df_runs = extract_runs(package, include_speaker_notes=include_speaker_notes)

    with timed_step("chart_point_extraction", logger=logger):
        df_chart_points = extract_chart_points(package, df_runs)

    with timed_step("table_cell_building", logger=logger):
        df_table_cells = build_table_cells(package, df_runs, include_speaker_notes=include_speaker_notes)

    if on_stage:
        on_stage("process_layouts")

    with timed_step("paragraph_building", logger=logger):
        df_paragraphs = build_paragraphs(df_runs, include_speaker_notes=include_speaker_notes)

    with timed_step("reading_order", logger=logger):
        df_paragraphs = assign_reading_order(df_paragraphs, df_runs)

    with timed_step("line_building", logger=logger):
        df_lines = build_lines(df_paragraphs)

    with timed_step("style_prefill", logger=logger):
        df_lines = prefill_styles(df_lines)

    # ── Document metadata: native → text fallback → consolidate → page info ──
    with timed_step("document_metadata", logger=logger):
        # Native channel — docProps/core.xml + app.xml.
        discovered_metadata: Dict[str, Any] = extract_native_metadata(package)
        discovered_metadata["has_ocr"] = False
        discovered_metadata["is_password_protected"] = _is_password_protected
        # Page info — native <Slides>, else the max page_number seen in df_runs.
        if not discovered_metadata.get("page_count"):
            discovered_metadata["page_count"] = (
                int(df_runs["page_number"].max())
                if not df_runs.empty and "page_number" in df_runs.columns
                else 0
            )
        # Text channel + consolidate — same recipe as the pdf/html pipelines.
        safe_enrich(
            add_text_fallbacks, discovered_metadata, df_lines,
            fallback={"author_text": None, "title_text": None, "language_text": None},
            logger=logger,
        )
        consolidate(discovered_metadata)

    return PptxPipelineResult(
        discovered_metadata=discovered_metadata,
        df_runs=df_runs,
        df_chart_points=df_chart_points,
        df_table_cells=df_table_cells,
        df_paragraphs=df_paragraphs,
        df_lines=df_lines,
    )
