"""DOCX pipeline entry point — runs all extraction steps and returns structured data."""

from __future__ import annotations

import logging
import zipfile
from pathlib import Path
from typing import Any, BinaryIO, Callable, Dict, NamedTuple, Optional

import pandas as pd

from .._utils.io.yaml_loader import load_page_label_config
from .._utils.password import decrypt_office, is_encrypted_office
from .._utils.safe_call import safe_enrich
from .._utils.timing import timed_step
from ..metadata import add_text_fallbacks, consolidate
from .native_metadata import extract_native_metadata
from .step_01_package_reader import DocxPackage, read_docx_package
from .step_02_run_extractor import expand_header_footer_runs, extract_runs
from .step_03_chart_point_builder import build_chart_points
from .step_04_table_cell_builder import build_table_cells
from .step_05_paragraph_builder import build_paragraphs
from .step_06_line_builder import build_lines
from .step_07_style_prefiller import prefill_block_types

logger = logging.getLogger(__name__)


class DocxPipelineResult(NamedTuple):
    """Structured result of :func:`run_pipeline`."""

    discovered_metadata: Dict[str, Any]
    df_runs: pd.DataFrame
    df_chart_points: pd.DataFrame
    df_table_cells: pd.DataFrame
    df_paragraphs: pd.DataFrame
    df_lines: pd.DataFrame


def run_pipeline(
    source: str | Path | bytes | BinaryIO,
    include_headers_footers: bool = False,
    include_footnotes: bool = True,
    include_comments: bool = False,
    password: str | None = None,
    source_filename: str | None = None,
    on_stage: Optional[Callable[[str], None]] = None,
) -> DocxPipelineResult:
    """
    Run the full DOCX extraction pipeline.

    Args:
        source: File path, raw .docx bytes, or a binary file-like object.
        include_headers_footers: When True, header and footer content is
            expanded once per page it appears on and surfaced in df_paragraphs
            and df_lines with block_type "header" / "footer".  When False
            (default) header/footer rows are present in df_runs for inspection
            but are dropped before building paragraphs and lines.
        include_footnotes: Include footnotes and endnotes (default True).
        include_comments: Include reviewer comments (default False) — these
            are review annotations, not document content.
        on_stage: Optional callback for progress updates.

    Returns:
        DocxPipelineResult with fields:
            discovered_metadata: Fully resolved document metadata (native +
                text-fallback channels consolidated, plus page/OCR/password info).
            df_runs: Run-level DataFrame (one row per text/control/image run event).
                Header/footer rows are always present here.
            df_chart_points: Datapoint-level DataFrame (one row per plotted
                point of each embedded chart).
            df_table_cells: Cell-level DataFrame (one row per logical table cell).
            df_paragraphs: Paragraph-level DataFrame (runs aggregated by paragraph_id).
            df_lines: Shared-compatible line DataFrame, ready for shared/ steps.
    """
    if on_stage:
        on_stage("extract_elements")

    # Read the package; an encrypted .docx surfaces as a BadZipFile we decrypt.
    _is_password_protected = False
    with timed_step("package_reading", logger=logger):
        try:
            package = read_docx_package(source)
        except zipfile.BadZipFile:
            if not (isinstance(source, bytes) and is_encrypted_office(source)):
                raise
            source = decrypt_office(source, password, source_filename)
            package = read_docx_package(source)
            _is_password_protected = True
    page_label_config = load_page_label_config()

    # Always extract header/footer runs so df_runs contains them for inspection.
    with timed_step("run_extraction", logger=logger):
        df_runs = extract_runs(
            package,
            include_headers_footers=True,
            include_footnotes=include_footnotes,
            include_comments=include_comments,
            page_label_config=page_label_config,
        )

    with timed_step("chart_point_extraction", logger=logger):
        df_chart_points = build_chart_points(package, df_runs)

    with timed_step("table_cell_building", logger=logger):
        df_table_cells = build_table_cells(
            package,
            df_runs,
            include_headers_footers=include_headers_footers,
            include_footnotes=include_footnotes,
            include_comments=include_comments,
        )

    if include_headers_footers:
        with timed_step("header_footer_run_expansion", logger=logger):
            df_runs_for_para = expand_header_footer_runs(df_runs, package)
    else:
        df_runs_for_para = df_runs

    if on_stage:
        on_stage("process_layouts")

    with timed_step("paragraph_building", logger=logger):
        df_paragraphs = build_paragraphs(df_runs_for_para, include_headers_footers=include_headers_footers)

    with timed_step("line_building", logger=logger):
        df_lines = build_lines(df_paragraphs)

    with timed_step("style_prefill", logger=logger):
        df_lines = prefill_block_types(df_lines)

    # ── Document metadata: native → text fallback → consolidate → page info ──
    with timed_step("document_metadata", logger=logger):
        # Native channel — docProps/core.xml + app.xml.
        discovered_metadata: Dict[str, Any] = extract_native_metadata(package)
        discovered_metadata["has_ocr"] = False
        discovered_metadata["is_password_protected"] = _is_password_protected
        # Page info — native <Pages>, else the max page_number seen in df_runs.
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

    return DocxPipelineResult(
        discovered_metadata=discovered_metadata,
        df_runs=df_runs,
        df_chart_points=df_chart_points,
        df_table_cells=df_table_cells,
        df_paragraphs=df_paragraphs,
        df_lines=df_lines,
    )
