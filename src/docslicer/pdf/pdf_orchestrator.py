# SPDX-FileCopyrightText: 2026 Market Framer Inc.
# SPDX-License-Identifier: AGPL-3.0-only
# pdf_orchestrator.py - PDF-specific document processing pipeline
"""
PDF-specific pipeline steps.

These steps extract and process PDF documents into a lines_df format
that can then be processed by the shared orchestrator.

Pipeline Steps:
    01. Word Extraction      - Extract words from PDF (pypdfium2)
    02. Image Extraction     - Extract images from PDF
    02b. Shape Extraction    - Extract shapes/lines from PDF (pypdfium2)
    03. Link Extraction      - Extract hyperlinks (pypdfium2)
    04. Shape Enhancer       - Merge/enhance shape metadata
    05a. Footnote Detection  - Flag footnote blocks in df_words
    05b. Line Number Detect  - Flag margin line numbers in df_words
    05c. Line Number Drop    - Remove flagged line-number words from df_words
    05d. Gutter Detection    - Detect column gutters, annotate df_words
    [OCR]                    - Run OCR pipeline if scanned document detected
    06. Cell Builder         - Build cells from words + shapes + links
    07. Page Labels          - Assign page labels to cells
    08. Global Y Coords      - Convert page-relative Y to document-global Y
    09. Line Builder         - Build lines + assign horizontal band IDs
    10. Table Builder        - Classify bands, extract table cells
    [OCR font sizes]         - Estimate font sizes from layout if OCR doc

Note: TOC, Exhibit, Doc Region, and Hierarchy detection are handled
by the shared orchestrator.
"""
import tempfile
from pathlib import Path
from typing import Callable, NamedTuple, Optional, Dict, Any

import pandas as pd
import logging

logger = logging.getLogger(__name__)

# PDF Pipeline Steps
from .step_01_word_extractor import extract_words
from .step_02_image_extractor import extract_images
from .step_03_shape_extractor import extract_shapes
from .step_04_link_extractor import extract_links
from .step_05_struct_group import assign_struct_group_id
from .step_06_style_prefiller import prefill_styles
from .step_07_stream_group import assign_stream_group_id
from .step_08_reading_order import assign_reading_order
from .step_09_word_relationships import add_word_relationships
from .step_10_cell_builder import build_cells
from .step_11_page_label_detector import detect_pdf_page_labels
from .step_12_cell_grouper import group_multiline_cells
from .step_13_line_builder import build_lines
from .step_14_table_builder import build_tables

# PDF Utils
from ._utils.struct_context import build_struct_context
from ._utils.coordinates import convert_to_global_y_coordinates


# Global Utils
from .._utils.layout.shape_processor import process_shapes
from .._utils.layout.layouts import assign_layouts, LayoutConfig
from .._utils.layout.reading_order import assign_reading_order as assign_reading_order_fallback
from .._utils.layout.line_number_detector import detect_line_numbers
from .._utils.io.yaml_loader import load_yamls
from .._utils.timing import timed_step
from .._utils.safe_call import safe_enrich
from .native_metadata import extract_native_metadata
from ..metadata import add_page_and_ocr_info, add_text_fallbacks, consolidate


class PdfPipelineResult(NamedTuple):
    """Structured result of :func:`run_pipeline`."""

    discovered_metadata: Dict[str, Any]
    df_lines: pd.DataFrame
    df_table_cells: Optional[pd.DataFrame]
    debug_steps: Dict[str, pd.DataFrame]


def _page_clause(pdf_path, discovered_metadata: Dict[str, Any]) -> str:
    """Return " (N pages)" for an OCR-unavailable message, "" if unknown.

    Sizing the document is what makes that message actionable — it is the
    difference between a stray fax and a 300-page report worth installing OCR
    for. The scanned branch that skips add_page_and_ocr_info has no count yet,
    so fall back to the page tree, which is readable with no text layer.
    """
    pages = discovered_metadata.get("page_count") or 0
    if not pages:
        try:
            import pikepdf
            with pikepdf.open(str(pdf_path)) as _pk:
                pages = len(_pk.pages)
        except Exception:
            return ""
    return f" ({pages} page{'s' if pages != 1 else ''})" if pages else ""


def run_pipeline(
    pdf_bytes: bytes,
    source_url: str = None,
    on_stage: Optional[Callable[[str], None]] = None,
    debug: bool = False,
    password: str | None = None,
    source_filename: str | None = None,
    max_workers: int | None = None,
) -> PdfPipelineResult:
    """
    Run PDF-specific document processing steps.

    Args:
        pdf_bytes: Raw PDF file content
        source_url: Original URL (optional, for metadata)
        on_stage: Optional callback for progress updates
        max_workers: Process-pool width for word extraction, cell building,
            and OCR (None -> auto performance-core count; 1 -> disable
            intra-document parallelism).

    Returns:
        PdfPipelineResult(discovered_metadata, df_lines, df_table_cells, debug_steps).
        debug_steps is an ordered dict of intermediate DataFrames when debug=True,
        empty dict otherwise.
    """
    page_label_dict, page_label_config, _, _ = load_yamls()

    with tempfile.TemporaryDirectory() as tmp_dir:
        pdf_path = Path(tmp_dir) / "input.pdf"
        pdf_path.write_bytes(pdf_bytes)

        # ── Stage: Extraction ────────────────────────────────────────────────
        if on_stage:
            on_stage("extract_elements")

        # Step 00 - Structure context (single pikepdf open, before any pdfium call)
        # Building it here means one pikepdf pass feeds words, images and shapes.
        # pikepdf is now the first library to touch the bytes, so a missing/wrong
        # password surfaces as a clean pikepdf.PasswordError we catch to decrypt —
        # rather than as pdfium's untyped error deeper in extract_words.
        #
        # If a password is supplied, pre-decrypt first so every downstream tool
        # (pikepdf struct-tree, pypdfium2) sees plain bytes.
        import pikepdf
        from .._utils.password import decrypt_pdf

        _is_password_protected = False
        if password is not None:
            pdf_bytes = decrypt_pdf(pdf_bytes, password, source_filename)
            pdf_path.write_bytes(pdf_bytes)
            _is_password_protected = True

        # Step 00 - Parse Struct Tree (pikepdf)
        with timed_step("struct_context_pikepdf", logger=logger):
            try:
                struct_ctx = build_struct_context(pdf_path)
            except pikepdf.PasswordError:
                # Encrypted with no/other password — try the common-password candidates.
                pdf_bytes = decrypt_pdf(pdf_bytes, None, source_filename)
                pdf_path.write_bytes(pdf_bytes)
                _is_password_protected = True
                struct_ctx = build_struct_context(pdf_path)

        # Step 01 - Word Extraction (pypdfium2)
        with timed_step("word_extraction", logger=logger):
            df_words = extract_words(pdf_path, struct_ctx=struct_ctx, max_workers=max_workers)

        # Step 02 - Image Extraction (struct-enriched by shared struct_index)
        with timed_step("image_extraction", logger=logger):
            df_images = extract_images(pdf_path, struct_index=struct_ctx.struct_index)

        # Step 03 - Shape Extraction (pypdfium2, struct-enriched)
        with timed_step("shape_extraction", logger=logger):
            df_shapes = extract_shapes(pdf_path, struct_index=struct_ctx.struct_index)

        # Step 04 - Link Extraction (pypdfium2)
        with timed_step("link_extraction", logger=logger):
            df_links = extract_links(pdf_path)

        # ── OCR check (before enrichment, before cell construction) ─────────
        discovered_metadata: Dict[str, Any] = {}

        if df_words.empty or "text" not in df_words.columns:
            # No text layer at all — definitely scanned
            discovered_metadata["needs_ocr"] = True
            discovered_metadata["is_scanned"] = True
        else:
            # Sophisticated detection: low char count + high image coverage per page
            if "char_count" not in df_words.columns:
                df_words["char_count"] = df_words["text"].str.len().fillna(0).astype(int)
            safe_enrich(
                add_page_and_ocr_info, discovered_metadata, df_words, df_images=df_images,
                fallback={"page_count": 1, "has_ocr": False}, logger=logger,
            )

        if discovered_metadata.get("needs_ocr"):
            # Checked before the OCR import, so a missing extra is reported as
            # "this document needs OCR and OCR is not installed" rather than as
            # an ImportError raised three modules deep. Returning an empty parse
            # instead would be worse: no headings is indistinguishable from a
            # genuinely empty document, and this is recoverable by installing.
            from ..ocr import OcrUnavailableError, ocr_unavailable_reason

            reason = ocr_unavailable_reason()
            if reason:
                raise OcrUnavailableError(
                    f"{source_filename or 'This PDF'} is a scanned document"
                    f"{_page_clause(pdf_path, discovered_metadata)} with no "
                    f"usable text layer, so reading it requires OCR, but {reason}"
                )

            import warnings
            warnings.warn(
                "Scanned PDF detected — running OCR pipeline. "
                "This may take significantly longer than normal parsing. "
                "Install the OCR extra if not already: "
                "pip install 'docslicer[ocr]'"
            )
            from ..ocr.ocr_orchestrator import run_ocr_pipeline, OCRPipelineConfig
            with timed_step("ocr_pipeline", logger=logger):
                df_words, df_shapes, df_grid_cells, df_gutters = run_ocr_pipeline(
                    pdf_bytes, config=OCRPipelineConfig(ocr_workers=max_workers), on_stage=on_stage,
                )
            discovered_metadata["has_ocr"] = True

        if df_words.empty:
            # No text even after OCR — nothing to parse
            return PdfPipelineResult(discovered_metadata, pd.DataFrame(), None, {})

        df_gutters = pd.DataFrame()

        if on_stage:
            on_stage("process_layouts")

        # The OCR pipeline already produces its own line_id via gutter-aware
        # reading order and strips its own margin line numbers, so struct-tree-based
        # enrichment is both unavailable (no struct tree for a scanned page) and
        # redundant here.
        if not discovered_metadata.get("has_ocr"):

            # ── Stage: Raw df cleanups ────────────────────────────────────────────────
            # Cleanup 1 - Shape Merging, Role Assignment & Grid Cells
            with timed_step("shape_merging_grid_cells", logger=logger):
                df_shapes, df_grid_cells = process_shapes(df_shapes)

            # Cleanup 2 - Line Number Detection & Removal
            # Line numbers are margin artefacts that must be removed entirely — unlike
            # other annotations they cannot be represented as a meaningful block_type.
            with timed_step("line_number_detection", logger=logger):
                df_words = detect_line_numbers(df_words)

                # NOTE: This operation removes rows from the df
                if "line_number_flag" in df_words.columns:
                    n_removed = df_words["line_number_flag"].sum()
                    if n_removed:
                        logger.debug("Dropping %d line-number word(s) from df_words", n_removed)
                    df_words = df_words[~df_words["line_number_flag"]].copy()


            # ── Stage: Reading Order ────────────────────────────────────────────────
            # Step 05 - Struct group assignment
            with timed_step("struct_group_assignment", logger=logger):
                df_words = assign_struct_group_id(df_words)

            # If struct_group_id is blank across the whole df, structure-tree data
            # wasn't available — skip stream grouping / reading order and fall
            # back to spatial line ordering instead.
            if "struct_group_id" not in df_words.columns or df_words["struct_group_id"].isna().all():
                # Step 08(b) - Stream Group Assignment - Fallback
                with timed_step("reading_order_fallback", logger=logger):
                    df_words, df_gutters = assign_reading_order_fallback(
                        df_words, df_shapes, df_grid_cells, max_workers=max_workers
                    )
            else:
                # Step 06 - Prefill Styles
                with timed_step("prefill_styles", logger=logger):
                    df_words = prefill_styles(df_words)
                # Step 07 - Stream Group Assignment
                with timed_step("stream_group_assignment", logger=logger):
                    df_words = assign_stream_group_id(df_words)
                # Step 08(a) - Stream Group Assignment
                with timed_step("reading_order", logger=logger):
                    df_words = assign_reading_order(df_words)


        # ── Stage: DF Enrichment (word level) ────────────────────────────────────────────────
        # Step 07 - - Word relationships: links, background rects, grid-cell
        # containment, underline/strikethrough and table rules. Also enriches
        # df_shapes with the hl_ KPI/classification columns.
        with timed_step("word_relationships", logger=logger):
            df_words, df_shapes = add_word_relationships(
                df_words, df_links, df_shapes, df_grid_cells)

        # ── Stage: Cell / Line / Layout Construction ─────────────────────────────────
        # Step 09 - Cell Builder
        # OCR font sizes are estimated per glyph and too noisy for the size/
        # baseline heuristics, so suppress sub/superscript detection on OCR pages.
        with timed_step("cell_building", logger=logger):
            df_cells, df_words = build_cells(
                df_words, detect_scripts=not discovered_metadata.get("has_ocr"), max_workers=max_workers
            )

        # Step 10 - Page Labels
        if page_label_config:
            with timed_step("page_label_detection", logger=logger):
                df_cells = detect_pdf_page_labels(df_cells, page_label_config)

        if not discovered_metadata.get("has_ocr"):
            # Step 12 - Group Multiline Cells
            with timed_step("group_multiline_cells", logger=logger):
                df_cells, df_words = group_multiline_cells(df_cells, df_words)

        # Step 13 - Line Builder
        with timed_step("line_building", logger=logger):
            df_lines = build_lines(df_cells)

        # Step 14 - Layout Assignment (layout_id, layout_type - table vs text, layout_score)
        with timed_step("layout_assignment", logger=logger):
            layout_config = (
                LayoutConfig.for_ocr_second_pass()
                if discovered_metadata.get("has_ocr")
                else LayoutConfig.for_pdf()
            )
            df_lines = assign_layouts(df_lines, config=layout_config)

            # Merge layout_id onto df_cells
            line_layout = df_lines.set_index("line_id")[
                ["layout_id", "layout_type", "layout_score"]
            ]
            df_cells["layout_id"]    = df_cells["line_id"].map(line_layout["layout_id"])
            df_cells["layout_type"]  = df_cells["line_id"].map(line_layout["layout_type"])
            df_cells["layout_score"] = df_cells["line_id"].map(line_layout["layout_score"])

        # ── Stage: Table Construction ─────────────────────────────────
        if on_stage:
            on_stage("extract_tables")

        # Step 10 - Table Builder
        with timed_step("table_building", logger=logger):
            df_cells, df_table_cells = build_tables(df_cells, df_grid_cells)

            # Merge final table_id back onto df_lines (build_tables runs after build_lines,
            # so df_lines otherwise never sees the final, dense table_id assign_layouts
            # doesn't compute — block_type="table" is already set by assign_layouts).
            line_table_id = df_cells.groupby("line_id")["table_id"].first()
            df_lines["table_id"] = df_lines["line_id"].map(line_table_id)

        # Optional - Convert Y coordinates from page-relative to global
        #df_cells = convert_to_global_y_coordinates(df_cells)

        # ── Document Information ─────────────────────────────────────────────
        with timed_step("document_metadata", logger=logger):
            # Native channel — the PDF's own XMP / /Info / catalog metadata. Reopen
            # pikepdf on the (already-decrypted) path; struct_context closed its handle.
            try:
                with pikepdf.open(str(pdf_path)) as _pk:
                    native = extract_native_metadata(_pk)
                # len(pdf.pages) is the true page count (counts trailing blank pages
                # that carry no words); prefer it, but never clobber the count that
                # add_page_and_ocr_info derived with 0 on an unreadable page tree.
                native_pages = native.pop("page_count", 0) or 0
                discovered_metadata.update(native)
                if native_pages:
                    discovered_metadata["page_count"] = native_pages
            except Exception as e:
                logger.error(f"Error in extract_native_metadata: {e}", exc_info=True)
                for key in ("title_meta", "author_meta", "language_meta"):
                    discovered_metadata.setdefault(key, None)
            # Text channel — heuristics over the parsed body as a fallback.
            safe_enrich(
                add_text_fallbacks, discovered_metadata, df_lines,
                fallback={"author_text": None, "title_text": None, "language_text": None},
                logger=logger,
            )
            # Fold both channels into the final title / author / language.
            consolidate(discovered_metadata)

        discovered_metadata["is_password_protected"] = _is_password_protected

        debug_steps: Dict[str, pd.DataFrame] = {}
        if debug:
            debug_steps["words"] = df_words
            debug_steps["shapes"] = df_shapes
            debug_steps["cells"] = df_cells
            debug_steps["lines"] = df_lines
            if df_table_cells is not None:
                debug_steps["table_cells"] = df_table_cells
            if not df_gutters.empty:
                debug_steps["gutters"] = df_gutters

        return PdfPipelineResult(discovered_metadata, df_lines, df_table_cells, debug_steps)
