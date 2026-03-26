# pdf_orchestrator.py - PDF-specific document processing pipeline
"""
PDF-specific pipeline steps.

These steps extract and process PDF documents into a lines_df format
that can then be processed by the shared orchestrator.

Pipeline Steps:
    01. Word Extraction - Extract words from PDF (PyMuPDF)
    02. Image Extraction - Extract images from PDF
    02b. Shape Extraction - Extract shapes/lines from PDF (pdfplumber)
    03. Link Extraction - Extract hyperlinks (PyMuPDF)
    04. Shape Enhancer - Enhance/merge shape metadata
    05a. Line Number Detection - Flag margin line numbers in df_words
    05b. Gutter Detection - Detect column gutters (future use)
    06. Cell Builder - Build cells from words + shapes
    07. Page Labels - Assign page labels
    08. Temp Line Builder - Build temporary lines
    09. Layout Detector - Detect page layout (columns)
    10. Table Processor - Process detected tables
    11. Line Merger - Merge cells into final lines
    
Note: TOC, Exhibit, Doc Region, and Hierarchy detection are handled
by the shared orchestrator.
"""
import tempfile
from pathlib import Path
import pandas as pd
from typing import Callable, Optional, Tuple, Dict, Any
import logging

logger = logging.getLogger(__name__)

# PDF-specific step imports (using new numbering with gutter at step 05)
from .step_01_word_extractor import extract_words
from .step_02_image_extractor import extract_images
from .step_02_shape_extractor import extract_shapes
from .step_03_link_extractor import extract_links
from .step_04_shape_merger import merge_shapes
from .step_05_line_number_detector import detect_line_numbers
from .step_06_gutter_detector import detect_and_annotate_gutters  # Imported but not yet used
from .step_07_cell_builder import build_cells
from .step_08_page_label_detector import assign_pdf_page_labels

# Config loaders
from .._utils.yaml_loader import load_yamls

# Metadata utilities
from ..metadata import add_page_and_ocr_info, add_document_information


def convert_to_global_y_coordinates(df_cells: pd.DataFrame) -> pd.DataFrame:
    """
    Convert page-relative Y coordinates to document-global Y coordinates.
    
    For multi-page documents, Y coordinates reset to 0 on each page.
    This causes issues when aggregating chunks that span page boundaries.
    
    This function:
    1. Saves original page-relative coords to y_top_local, y_bottom_local
    2. Calculates cumulative page heights to get global offsets
    3. Overwrites y_top, y_bottom with global coordinates
    
    Args:
        df_cells: DataFrame with page_number, page_height, y_top, y_bottom
    
    Returns:
        DataFrame with global Y coordinates (original saved to _local columns)
    """
    if df_cells.empty:
        return df_cells
    
    # Required columns check
    required_cols = ['page_number', 'page_height', 'y_top', 'y_bottom']
    if not all(col in df_cells.columns for col in required_cols):
        logger.warning(f"Missing columns for Y coordinate conversion, skipping")
        return df_cells
    
    # Save original page-relative coordinates
    df_cells['y_top_local'] = df_cells['y_top']
    df_cells['y_bottom_local'] = df_cells['y_bottom']
    
    # Calculate cumulative page offsets
    # Get the first page_height for each page (they should all be the same per page)
    page_heights = df_cells.groupby('page_number')['page_height'].first().sort_index()
    
    # Cumulative offset = sum of all previous page heights
    # Page 0: offset = 0
    # Page 1: offset = height[0]
    # Page 2: offset = height[0] + height[1]
    cumulative_offsets = page_heights.shift(1, fill_value=0).cumsum()
    
    # Convert to global coordinates
    df_cells['y_top'] = df_cells['y_top_local'] + df_cells['page_number'].map(cumulative_offsets)
    df_cells['y_bottom'] = df_cells['y_bottom_local'] + df_cells['page_number'].map(cumulative_offsets)
    
    #logger.info(f"✅ Converted Y coordinates to global (page offsets: {cumulative_offsets.to_dict()})")
    
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
    # Load configs (only need page_label for PDF steps)
    page_label_dict, page_label_config, _, _ = load_yamls()
    
    # Write bytes to temp file (PDF libraries need file path)
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(pdf_bytes)
        pdf_path = Path(tmp.name)
    
    try:
        # ============================================================
        # STAGE: PARSING (PDF Extraction)
        # ============================================================
        if on_stage:
            on_stage("parsing")
        
        # Step 01 - Word Extraction (PyMuPDF)
        df_words = extract_words(pdf_path)
        
        if df_words.empty:
            return {}, df_words, None
        
        # Step 02 - Image Extraction
        df_images = extract_images(pdf_path)
        
        # Step 02b - Shape Extraction (pdfplumber)
        df_shapes = extract_shapes(pdf_path)
        
        # Step 03 - Link Extraction (PyMuPDF)
        df_links = extract_links(pdf_path)
        
        # ============================================================
        # STAGE: HIERARCHY (Cell & Line Construction)
        # ============================================================
        if on_stage:
            on_stage("hierarchy")
        
        # Step 04 - Shape Enhancement
        df_shapes = merge_shapes(df_shapes, merge_lines=True)
        
        # Step 05a - Line Number Detection
        df_words = detect_line_numbers(df_words)

        # Step 05b - Gutter Detection (imported but not yet integrated)
        # df_words = detect_and_annotate_gutters(df_words)  # Future use

        # Step 06 - Cell Builder
        df_cells, df_words = build_cells(df_words, df_shapes, df_links)
        
        # Step 07 - Page Labels
        if page_label_config:
            df_cells = assign_pdf_page_labels(df_cells, page_label_config)
        
        # Convert Y coordinates from page-relative to global
        # (Saves original to y_top_local, y_bottom_local)
        # This ensures chunks spanning page boundaries aggregate correctly
        df_cells = convert_to_global_y_coordinates(df_cells)
        
        # Initialize metadata dict (need it for add_page_and_ocr_info)
        discovered_metadata = {}
        
        # Add page and OCR info (AFTER page labels)
        try:
            add_page_and_ocr_info(discovered_metadata, df_cells, df_images=df_images)
        except Exception as e:
            logger.error(f"❌ Error in add_page_and_ocr_info: {e}", exc_info=True)
            # Set defaults if extraction fails
            discovered_metadata.setdefault("page_count", 1)
            discovered_metadata.setdefault("is_password_protected", False)
            discovered_metadata.setdefault("has_ocr", False)
        
        # Step 08 - Temp Line Builder
        df_temp_lines, df_cells = build_temp_lines(df_cells, df_words)
        
        # Step 09 - Layout Detection
        df_cells = build_layout(df_cells)
        
        # Step 10 - Table Processing
        df_table_cells, df_cells = process_tables(df_cells)
        
        # Step 11 - Line Merger (final lines_df)
        df_lines = merge_cells_to_lines(df_cells)
        
        # Add document information (AFTER lines are created)
        # This extracts author, title, language, profile from PDF and df_lines
        try:
            add_document_information(discovered_metadata, pdf_path=pdf_path, df_lines=df_lines)
        except Exception as e:
            logger.error(f"❌ Error in add_document_information: {e}", exc_info=True)
            # Set defaults if extraction fails
            discovered_metadata.setdefault("author_meta", None)
            discovered_metadata.setdefault("author_text", None)
            discovered_metadata.setdefault("title_meta", None)
            discovered_metadata.setdefault("title_text", None)
            discovered_metadata.setdefault("language", "unknown")
            discovered_metadata.setdefault("profile", "unknown")
        
        # ============================================================
        # Finalize Discovered Metadata
        # ============================================================
        
        # Check if PDF was password protected
        discovered_metadata["is_password_protected"] = False
        
        # df_table_cells already set from process_tables() at line 139
        # Don't re-extract from df_lines as it loses the proper cell structure
        
        return discovered_metadata, df_lines, df_table_cells
        
    finally:
        # Clean up temp file
        try:
            pdf_path.unlink()
        except Exception:
            pass
