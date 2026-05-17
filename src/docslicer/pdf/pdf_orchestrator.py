# pdf_orchestrator.py - PDF-specific document processing pipeline
"""
PDF-specific pipeline steps.

These steps extract and process PDF documents into a lines_df format
that can then be processed by the shared orchestrator.

Pipeline Steps:
    01. Word Extraction      - Extract words from PDF (PyMuPDF)
    02. Image Extraction     - Extract images from PDF
    02b. Shape Extraction    - Extract shapes/lines from PDF (pdfplumber)
    03. Link Extraction      - Extract hyperlinks (PyMuPDF)
    04. Shape Enhancer       - Merge/enhance shape metadata
    05a. Line Number Detect  - Flag margin line numbers in df_words
    05b. Gutter Detection    - Detect column gutters, annotate df_words
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

from .step_01_word_extractor import extract_words
from .step_02_image_extractor import extract_images
from .step_03_shape_extractor import extract_shapes
from .step_04_link_extractor import extract_links
from .step_05_shape_merger import merge_shapes
from .step_06_line_number_detector import detect_line_numbers
from .step_07_gutter_detector import detect_and_annotate_gutters
from .step_08_cell_builder import build_cells
from .step_09_page_label_detector import assign_pdf_page_labels
from .step_10_line_builder import build_lines
from .step_11_table_builder import build_tables

from .._utils.yaml_loader import load_yamls
from ..metadata import add_page_and_ocr_info, add_document_information


def convert_to_global_y_coordinates(df_cells: pd.DataFrame) -> pd.DataFrame:
    """
    Convert page-relative Y coordinates to document-global Y coordinates.

    For multi-page documents, Y coordinates reset to 0 on each page.
    This causes issues when aggregating chunks that span page boundaries.

    Saves original page-relative coords to y_top_local / y_bottom_local,
    then overwrites y_top / y_bottom with global coordinates.
    """
    if df_cells.empty:
        return df_cells

    required_cols = ["page_number", "page_height", "y_top", "y_bottom"]
    if not all(col in df_cells.columns for col in required_cols):
        logger.warning("Missing columns for Y coordinate conversion, skipping")
        return df_cells

    df_cells["y_top_local"] = df_cells["y_top"]
    df_cells["y_bottom_local"] = df_cells["y_bottom"]

    page_heights = df_cells.groupby("page_number")["page_height"].first().sort_index()
    cumulative_offsets = page_heights.shift(1, fill_value=0).cumsum()

    df_cells["y_top"] = df_cells["y_top_local"] + df_cells["page_number"].map(cumulative_offsets)
    df_cells["y_bottom"] = df_cells["y_bottom_local"] + df_cells["page_number"].map(cumulative_offsets)

    return df_cells


def run_pipeline(
    pdf_bytes: bytes,
    source_url: str = None,
    on_stage: Optional[Callable[[str], None]] = None,
) -> Tuple[Dict[str, Any], pd.DataFrame, Optional[pd.DataFrame]]:
    """
    Run PDF-specific document processing steps.

    Args:
        pdf_bytes: Raw PDF file content
        source_url: Original URL (optional, for metadata)
        on_stage: Optional callback for progress updates

    Returns:
        Tuple of (discovered_metadata, df_lines, df_table_cells)
    """
    page_label_dict, page_label_config, _, _ = load_yamls()

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(pdf_bytes)
        pdf_path = Path(tmp.name)

    try:
        # ── Stage: Extraction ────────────────────────────────────────────────
        if on_stage:
            on_stage("parsing")

        # Step 01 - Word Extraction (PyMuPDF)
        df_words = extract_words(pdf_path)

        # Step 02 - Image Extraction
        df_images = extract_images(pdf_path)

        # Step 02b - Shape Extraction (pdfplumber)
        df_shapes = extract_shapes(pdf_path)

        # Step 03 - Link Extraction (PyMuPDF)
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
            df_words, df_shapes, _ = run_ocr_pipeline(pdf_bytes)
            discovered_metadata["has_ocr"] = True

        if df_words.empty:
            # No text even after OCR — nothing to parse
            return discovered_metadata, pd.DataFrame(), None

        # ── Stage: Enrichment ────────────────────────────────────────────────
        if on_stage:
            on_stage("enrichment")

        # Step 04 - Shape Enhancement
        df_shapes = merge_shapes(df_shapes, merge_lines=True)

        # Step 05a - Line Number Detection
        df_words = detect_line_numbers(df_words)

        # Step 05b - Gutter Detection
        df_words, _, _ = detect_and_annotate_gutters(df_words, df_shapes)

        # ── Stage: Cell & Line Construction ─────────────────────────────────
        if on_stage:
            on_stage("layout")

        # Step 06 - Cell Builder
        df_cells, df_words = build_cells(df_words, df_shapes, df_links)

        # Step 07 - Page Labels
        if page_label_config:
            df_cells = assign_pdf_page_labels(df_cells, page_label_config)

        # Step 08 - Convert Y coordinates from page-relative to global
        df_cells = convert_to_global_y_coordinates(df_cells)

        # Step 09 - Line Builder
        df_lines, df_cells = build_lines(df_cells)

        # Step 10 - Table Builder
        df_lines, df_cells, df_table_cells = build_tables(df_lines, df_cells, df_words)

        # OCR font size estimation (requires layout_id from step 10)
        if discovered_metadata.get("needs_ocr"):
            from ..ocr.font_size_estimator import estimate_ocr_font_sizes
            df_lines = estimate_ocr_font_sizes(df_lines)

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

        discovered_metadata["is_password_protected"] = False

        return discovered_metadata, df_lines, df_table_cells

    finally:
        try:
            pdf_path.unlink()
        except Exception:
            pass
