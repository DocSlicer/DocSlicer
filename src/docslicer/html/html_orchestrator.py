# html_orchestrator.py - HTML-specific document processing pipeline
"""
HTML-specific pipeline steps.

These steps extract and process HTML documents into a lines_df format
that can then be processed by the shared orchestrator.

Pipeline Steps:
    01. Box Extraction (Playwright) - Extract visual boxes from HTML
    02. Box Cleaning - Clean boxes, add text features
    03. Page Labels - Detect page numbers/labels
    04. Table Extraction - Extract tables from HTML
    05. Line Merger - Merge boxes into visual lines
    
Note: TOC, Exhibit, Doc Region, and Hierarchy detection are handled
by the shared orchestrator.
"""
import pandas as pd
from typing import Callable, Optional, Tuple, Dict, Any
import logging

logger = logging.getLogger(__name__)

# HTML-specific step imports
from .step_01_box_extractor import extract_boxes_with_playwright
from .step_02_box_cleaner import clean_boxes
from .step_03_page_labels import assign_page_labels
from .step_04_line_merger import merge_boxes_to_lines
from .step_05_table_extractor import extract_tables_from_html

# Config loaders
from .._utils.yaml_loader import load_yamls

# Metadata utilities
from ..metadata import add_page_and_ocr_info, add_document_information


def run_pipeline(
    html: str,
    source_url: str,
    on_stage: Optional[Callable[[str], None]] = None,
) -> Tuple[Dict[str, Any], pd.DataFrame, Optional[pd.DataFrame]]:
    """
    Run HTML-specific document processing steps.
    
    Pipeline Steps:
        01. Box Extraction (Playwright) - Extract visual boxes from HTML
        02. Box Cleaning - Clean boxes, add text features
        03. Page Labels - Detect page numbers/labels
        04. Table Extraction - Extract tables from HTML
        05. Line Merger - Merge boxes into visual lines
    
    Args:
        html: Raw HTML content
        source_url: Original URL (for link normalization)
        on_stage: Optional callback for progress updates
    
    Returns:
        Tuple of (discovered_metadata, df_lines, df_table_cells)
    """
    # Load configs (only need page_label for HTML steps)
    page_label_dict, page_label_config, _, _ = load_yamls()
    
    # ============================================================
    # STAGE: PARSING (Steps 01-04)
    # ============================================================
    if on_stage:
        on_stage("parsing")
    
    # Step 01 - Box Extraction (Playwright)
    # For URL processing: pass source_url, html=None (Playwright navigates to URL)
    # For HTML content: pass html, source_url=None (Playwright renders HTML string)
    # Returns: boxes, rendered_html, screenshot_base64, page_dimensions
    if source_url:
        boxes, rendered_html, screenshot_base64, page_dimensions = extract_boxes_with_playwright(
            None, page_label_dict, source_url
        )
    else:
        boxes, rendered_html, screenshot_base64, page_dimensions = extract_boxes_with_playwright(
            html, page_label_dict, None
        )
    df_boxes = pd.DataFrame(boxes)
    logger.info(f"📄 Step 01 - Box extraction complete, {len(df_boxes)} boxes, page: {page_dimensions}")
    
    if df_boxes.empty:
        logger.warning("📄 Empty DataFrame after box extraction, returning early")
        return {}, df_boxes, None
    
    # Step 02 - Box Cleaning (adds text features like char_count, word_count, etc.)
    df_boxes = clean_boxes(df_boxes, keep_debug_cols=False, dry_run=False)
    
    # Step 03 - Page Labels (runs on BOXES - needs box_id)
    df_boxes, page_labels, page_label_groups = assign_page_labels(
        df_boxes, page_label_config
    )
    
    # Initialize metadata dict (need it for add_page_and_ocr_info)
    discovered_metadata = {}
    discovered_metadata["rendered_html"] = rendered_html
    
    # Add page and OCR info (AFTER page labels)
    try:
        add_page_and_ocr_info(discovered_metadata, df_boxes, df_images=pd.DataFrame())
    except Exception as e:
        logger.error(f"❌ Error in add_page_and_ocr_info: {e}", exc_info=True)
        # Set defaults if extraction fails
        discovered_metadata.setdefault("page_count", 1)
        discovered_metadata.setdefault("is_password_protected", False)
        discovered_metadata.setdefault("has_ocr", False)
    
    # Step 04 - Table Extraction
    df_table_cells = extract_tables_from_html(
        html,
        min_rows=2,
        page_number=0,
        remove_single_row_tables=True,
    )
    
    # Map page labels to tables
    if df_table_cells is not None and not df_table_cells.empty and not df_boxes.empty:
        if 'page_number' in df_boxes.columns and 'page_label' in df_boxes.columns:
            page_info = df_boxes[['page_number', 'page_label']].drop_duplicates().sort_values('page_number')
            
            if not page_info.empty:
                unique_table_ids = sorted(df_table_cells['table_id'].unique())
                table_page_mapping = []
                for i, table_id in enumerate(unique_table_ids):
                    page_idx = min(i, len(page_info) - 1)
                    table_page_mapping.append({
                        'table_id': table_id,
                        'page_number_mapped': page_info.iloc[page_idx]['page_number'],
                        'page_label': page_info.iloc[page_idx]['page_label']
                    })
                
                table_page_mapping_df = pd.DataFrame(table_page_mapping)
                df_table_cells = df_table_cells.merge(table_page_mapping_df, on='table_id', how='left')
                df_table_cells['page_number'] = df_table_cells['page_number_mapped']
                df_table_cells = df_table_cells.drop(columns=['page_number_mapped'])
    
    # Convert from pixels to points (96 px = 72 pt)
    # NOTE: page_width/page_height stay in PIXELS for screenshot overlay alignment
    PX_TO_PT = 0.75
    geometry_cols = ["x_left", "x_right", "y_top", "y_bottom", "width", "height", "font_size"]
    
    for col in geometry_cols:
        if col in df_boxes.columns:
            df_boxes[col] = df_boxes[col] * PX_TO_PT
    
    # ============================================================
    # STAGE: HIERARCHY (Step 05)
    # ============================================================
    if on_stage:
        on_stage("hierarchy")
    
    # Step 05 - Line Merger (boxes -> lines)
    df_lines = merge_boxes_to_lines(df_boxes, remove_single_row_tables=True)
    
    # Add document information (AFTER lines are created)
    # This extracts author, title, language, profile from rendered HTML and df_lines
    try:
        add_document_information(discovered_metadata, html_content=rendered_html, df_lines=df_lines)
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
    
    # HTML is never password protected
    discovered_metadata["is_password_protected"] = False
    
    # HTML doesn't use OCR
    discovered_metadata["has_ocr"] = False
    
    # Store screenshot for pixel-perfect overlay
    discovered_metadata["screenshot_base64"] = screenshot_base64
    discovered_metadata["page_dimensions"] = page_dimensions
    
    # Return None for table_cells if empty
    if df_table_cells is not None and df_table_cells.empty:
        df_table_cells = None
    
    return discovered_metadata, df_lines, df_table_cells
