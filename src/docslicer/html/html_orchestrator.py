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
import logging
from typing import Callable, Optional, Tuple, Dict, Any

import pandas as pd

logger = logging.getLogger(__name__)

# HTML-specific step imports
from .step_01_box_extractor import extract_boxes_with_playwright
from .step_02_box_cleaner import clean_boxes
from .step_03_page_label_detector import assign_page_labels
from .step_04_line_builder import merge_boxes_to_lines
from .step_05_table_extractor import extract_table_cells

# Config loaders
from .._utils.yaml_loader import load_yamls

# Metadata utilities
from ..metadata import add_page_and_ocr_info, add_document_information

# Scraping
from ..scraping.dispatcher import fetch_url, _is_sec_url


def _inject_base_url(html: str, base_url: str) -> str:
    """Inject <base href> so relative URLs resolve correctly in about:blank context."""
    tag = f'<base href="{base_url}">'
    if "<head>" in html:
        return html.replace("<head>", f"<head>{tag}", 1)
    if "<Head>" in html or "<HEAD>" in html:
        import re
        return re.sub(r"<[Hh][Ee][Aa][Dd]>", f"<head>{tag}", html, count=1)
    return tag + html


def run_pipeline(
    html: str | None,
    source_url: str | None,
    on_stage: Optional[Callable[[str], None]] = None,
) -> Tuple[Dict[str, Any], pd.DataFrame, Optional[pd.DataFrame]]:
    """
    Run HTML-specific document processing steps.

    Accepts either pre-loaded HTML content or a URL. When a URL is given:
    - SEC/Congress URLs are fetched via SecHttpFetcher (rate-limited, bytes).
    - All other URLs are rendered directly by Playwright (JS execution, full quality).

    Args:
        html: Raw HTML string, or None when source_url is provided.
        source_url: URL to fetch/render, or None when html is provided.
        on_stage: Optional callback for progress updates.

    Returns:
        Tuple of (discovered_metadata, df_lines, df_table_cells)
    """
    _, page_label_config, _, _ = load_yamls()

    # ============================================================
    # Fetch step — resolve source_url to html bytes if needed
    # ============================================================
    if source_url and html is None:
        if _is_sec_url(source_url):
            # SEC: fetch bytes via rate-limited fetcher, render as static HTML.
            # Inject <base href> so relative URLs resolve correctly in Playwright's
            # about:blank context (otherwise parent.href returns "about:blank/...")
            scraped = fetch_url(source_url)
            html = scraped.raw_bytes.decode(scraped.encoding or "utf-8", errors="replace")
            html = _inject_base_url(html, scraped.final_url)
            source_url = None  # box extractor will use set_content
        # else: non-SEC → keep source_url=source_url, html=None
        #        box extractor will page.goto(source_url) for full JS rendering
    elif source_url and html is not None:
        # Caller already fetched/provided HTML but still gave its original URL for
        # link resolution. Render via set_content, with a <base> tag for relative
        # URLs, because the box extractor accepts either html or url, not both.
        html = _inject_base_url(html, source_url)
        source_url = None

    # ============================================================
    # STAGE: PARSING (Steps 01-03)
    # ============================================================
    if on_stage:
        on_stage("parsing")

    # Step 01 - Box Extraction (Playwright)
    # html XOR source_url must be set (enforced inside extract_boxes_with_playwright)
    # For non-SEC URLs: Playwright navigates directly. If that fails, fall back to
    # fetching bytes via http_fetcher and rendering with set_content.
    try:
        boxes, rendered_html = extract_boxes_with_playwright(html, source_url)
    except Exception as e:
        if source_url:
            logger.warning(f"Playwright navigation failed for {source_url}, falling back to http_fetcher: {e}")
            scraped = fetch_url(source_url)
            html = scraped.raw_bytes.decode(scraped.encoding or "utf-8", errors="replace")
            html = _inject_base_url(html, scraped.final_url)
            boxes, rendered_html = extract_boxes_with_playwright(html, source_url=None)
        else:
            raise
    df_boxes = pd.DataFrame(boxes)
    logger.info(f"Step 01 - Box extraction complete, {len(df_boxes)} boxes")

    if df_boxes.empty:
        logger.warning("Empty DataFrame after box extraction, returning early")
        return {}, df_boxes, None

    # Step 02 - Box Cleaning
    df_boxes = clean_boxes(df_boxes, keep_debug_cols=False, dry_run=False)

    # Step 03 - Page Labels (runs on boxes — needs box_id)
    df_boxes, page_labels, page_label_groups = assign_page_labels(df_boxes, page_label_config)

    discovered_metadata: Dict[str, Any] = {}
    discovered_metadata["rendered_html"] = rendered_html

    try:
        add_page_and_ocr_info(discovered_metadata, df_boxes, df_images=pd.DataFrame())
    except Exception as e:
        logger.error(f"Error in add_page_and_ocr_info: {e}", exc_info=True)
        discovered_metadata.setdefault("page_count", 1)
        discovered_metadata.setdefault("is_password_protected", False)
        discovered_metadata.setdefault("has_ocr", False)

    # Convert pixels to points (96 px = 72 pt)
    PX_TO_PT = 0.75
    geometry_cols = ["x_left", "x_right", "y_top", "y_bottom", "width", "height", "font_size"]
    for col in geometry_cols:
        if col in df_boxes.columns:
            df_boxes[col] = df_boxes[col] * PX_TO_PT

    # ============================================================
    # STAGE: HIERARCHY (Steps 04-05)
    # ============================================================
    if on_stage:
        on_stage("hierarchy")

    # Step 04 - Line Builder (boxes → lines); must run before table extractor
    # so that the final reindexed table_id values are available for ID matching.
    df_lines = merge_boxes_to_lines(df_boxes, remove_single_row_tables=True)

    # Step 05 - Table Extractor (uses df_lines.original_table_id for table_id sync)
    df_table_cells = extract_table_cells(
        df_lines=df_lines,
        rendered_html=rendered_html,
    )

    try:
        add_document_information(discovered_metadata, html_content=rendered_html, df_lines=df_lines)
    except Exception as e:
        logger.error(f"Error in add_document_information: {e}", exc_info=True)
        discovered_metadata.setdefault("author_meta", None)
        discovered_metadata.setdefault("author_text", None)
        discovered_metadata.setdefault("title_meta", None)
        discovered_metadata.setdefault("title_text", None)
        discovered_metadata.setdefault("language", "unknown")
        discovered_metadata.setdefault("profile", "unknown")

    # ============================================================
    # Finalize metadata
    # ============================================================
    discovered_metadata["is_password_protected"] = False
    discovered_metadata["has_ocr"] = False

    if df_table_cells is not None and not isinstance(df_table_cells, pd.DataFrame):
        df_table_cells = None
    elif df_table_cells is not None and df_table_cells.empty:
        df_table_cells = None

    return discovered_metadata, df_lines, df_table_cells
