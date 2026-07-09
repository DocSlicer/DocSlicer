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
import pandas as pd
from typing import Callable, Optional, Tuple, Dict, Any
import logging

logger = logging.getLogger(__name__)

# PDF Pipeline Steps
from .step_01_word_extractor import extract_words
from .step_02_image_extractor import extract_images
from .step_03_shape_extractor import extract_shapes
from .step_04_link_extractor import extract_links
from .step_05_struct_group import assign_struct_group_id
from .step_06_style_prefiller import prefill_styles
from .step_07_word_relationships import add_link_relationships
from .step_08_stream_group import assign_stream_group_id
from .step_09_reading_order import assign_reading_order
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
from .._utils.layout.layouts import assign_layouts
from .._utils.layout.reading_order import assign_reading_order as assign_reading_order_fallback
from .._utils.layout.line_number_detector import detect_line_numbers
from .._utils.io.yaml_loader import load_yamls
from ..metadata import add_page_and_ocr_info, add_document_information


def run_pipeline(
    pdf_bytes: bytes,
    source_url: str = None,
    on_stage: Optional[Callable[[str], None]] = None,
    debug: bool = False,
    password: str | None = None,
    source_filename: str | None = None,
) -> Tuple[Dict[str, Any], pd.DataFrame, Optional[pd.DataFrame], Dict[str, pd.DataFrame]]:
    """
    Run PDF-specific document processing steps.

    Args:
        pdf_bytes: Raw PDF file content
        source_url: Original URL (optional, for metadata)
        on_stage: Optional callback for progress updates

    Returns:
        Tuple of (discovered_metadata, df_lines, df_table_cells, debug_steps).
        debug_steps is an ordered dict of intermediate DataFrames when debug=True,
        empty dict otherwise.
    """
    page_label_dict, page_label_config, _, _ = load_yamls()

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(pdf_bytes)
        pdf_path = Path(tmp.name)

    try:
        # ── Stage: Extraction ────────────────────────────────────────────────
        if on_stage:
            on_stage("parsing")

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
        try:
            struct_ctx = build_struct_context(pdf_path)
        except pikepdf.PasswordError:
            # Encrypted with no/other password — try the common-password candidates.
            pdf_bytes = decrypt_pdf(pdf_bytes, None, source_filename)
            pdf_path.write_bytes(pdf_bytes)
            _is_password_protected = True
            struct_ctx = build_struct_context(pdf_path)

        # Step 01 - Word Extraction (pypdfium2)
        df_words = extract_words(pdf_path, struct_ctx=struct_ctx)

        # Step 02 - Image Extraction (struct-enriched by shared struct_index)
        df_images = extract_images(pdf_path, struct_index=struct_ctx.struct_index)

        # Step 03 - Shape Extraction (pypdfium2, struct-enriched)
        df_shapes = extract_shapes(pdf_path, struct_index=struct_ctx.struct_index)

        # Step 04 - Link Extraction (pypdfium2)
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
            try:
                add_page_and_ocr_info(discovered_metadata, df_words, df_images=df_images)
            except Exception as e:
                logger.error(f"Error in add_page_and_ocr_info: {e}", exc_info=True)
                discovered_metadata.setdefault("page_count", 1)
                discovered_metadata.setdefault("has_ocr", False)

        if discovered_metadata.get("needs_ocr"):
            import warnings
            warnings.warn(
                "Scanned PDF detected — running OCR pipeline. "
                "This may take significantly longer than normal parsing. "
                "Install pytesseract and opencv-python if not already: "
                "pip install 'docslicer[ocr]'"
            )
            from ..ocr.ocr_orchestrator import run_ocr_pipeline
            df_words, df_shapes, df_grid_cells = run_ocr_pipeline(pdf_bytes)
            discovered_metadata["has_ocr"] = True

        if df_words.empty:
            # No text even after OCR — nothing to parse
            return discovered_metadata, pd.DataFrame(), None, {}

        
        # The OCR pipeline already produces its own line_id via gutter-aware
        # reading order and strips its own margin line numbers, so struct-tree-based
        # enrichment is both unavailable (no struct tree for a scanned page) and
        # redundant here.
        if not discovered_metadata.get("has_ocr"):

            # ── Stage: Raw df cleanups ────────────────────────────────────────────────
            if on_stage:
                on_stage("df cleanup")

            # Cleanup 1 - Shape Merging, Role Assignment & Grid Cells
            df_shapes, df_grid_cells = process_shapes(df_shapes)

            # Cleanup 2 - Line Number Detection & Removal
            # Line numbers are margin artefacts that must be removed entirely — unlike
            # other annotations they cannot be represented as a meaningful block_type.
            df_words = detect_line_numbers(df_words)

            # NOTE: This operation removes rows from the df
            if "line_number_flag" in df_words.columns:
                n_removed = df_words["line_number_flag"].sum()
                if n_removed:
                    logger.debug("Dropping %d line-number word(s) from df_words", n_removed)
                df_words = df_words[~df_words["line_number_flag"]].copy()

            # Cleanup 3 - Merge links onto words
            df_words = add_link_relationships(df_words, df_links)

            # ── Stage: Reading Order ────────────────────────────────────────────────
            if on_stage:
                on_stage("reading order")

            # Step 05 - Struct group assignment
            df_words = assign_struct_group_id(df_words)

            # If struct_group_id is blank across the whole df, structure-tree data
            # wasn't available — skip stream grouping / reading order and fall
            # back to spatial line ordering instead.
            if "struct_group_id" not in df_words.columns or df_words["struct_group_id"].isna().all():
                # Step 08(b) - Stream Group Assignment - Fallback
                df_words = assign_reading_order_fallback(df_words, df_shapes)
            else:
                # Step 06 - Prefill Styles
                df_words = prefill_styles(df_words)
                # Step 07 - Stream Group Assignment
                df_words = assign_stream_group_id(df_words)
                # Step 08(a) - Stream Group Assignment
                df_words = assign_reading_order(df_words)

        # ── Stage: Cell / Line / Layout Construction ─────────────────────────────────
        if on_stage:
            on_stage("layout")

        # Step 09 - Cell Builder
        df_cells, df_words = build_cells(df_words)

        # Step 10 - Page Labels
        if page_label_config:
            df_cells = detect_pdf_page_labels(df_cells, page_label_config)

        ###############################
        # NOTE Next steps
        ###############################

        if not discovered_metadata.get("has_ocr"):

            # Step 12 - Group Multiline Cells
            df_cells, df_words = group_multiline_cells(df_cells, df_words)

        # Step 13 - Line Builder
        df_lines = build_lines(df_cells)

        # Step 14 - Layout Assignment (layout_id, layout_type - table vs text, layout_score)
        df_lines = assign_layouts(df_lines)

        # Merge layout_id onto df_cells
        line_layout = df_lines.set_index("line_id")[
            ["layout_id", "layout_type", "layout_score"]
        ]
        df_cells["layout_id"]    = df_cells["line_id"].map(line_layout["layout_id"])
        df_cells["layout_type"]  = df_cells["line_id"].map(line_layout["layout_type"])
        df_cells["layout_score"] = df_cells["line_id"].map(line_layout["layout_score"])

        # ── Stage: Table Construction ─────────────────────────────────
        if on_stage:
            on_stage("table")

        # Step 10 - Table Builder
        df_cells, df_table_cells = build_tables(df_cells, df_grid_cells)

        # Optional - Convert Y coordinates from page-relative to global
        #df_cells = convert_to_global_y_coordinates(df_cells)

        # ── Document Information ─────────────────────────────────────────────
        try:
            add_document_information(discovered_metadata, pdf_path=pdf_path, df_lines=df_lines)
        except Exception as e:
            logger.error(f"Error in add_document_information: {e}", exc_info=True)
            discovered_metadata.setdefault("author_meta", None)
            discovered_metadata.setdefault("author_text", None)
            discovered_metadata.setdefault("title_meta", None)
            discovered_metadata.setdefault("title_text", None)
            discovered_metadata.setdefault("language", "unknown")

        discovered_metadata["is_password_protected"] = _is_password_protected

        debug_steps: Dict[str, pd.DataFrame] = {}
        if debug:
            debug_steps["words"] = df_words
            debug_steps["shapes"] = df_shapes
            debug_steps["cells"] = df_cells
            debug_steps["lines"] = df_lines
            if df_table_cells is not None:
                debug_steps["table_cells"] = df_table_cells

        return discovered_metadata, df_lines, df_table_cells, debug_steps

    finally:
        try:
            pdf_path.unlink()
        except Exception:
            pass
