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
from .step_02_box_cleaner import clean_boxes
from .step_03_page_label_detector import assign_page_labels
from .step_04_line_builder import merge_boxes_to_lines
from .step_05_table_extractor import extract_table_cells

# Config loaders
from .._utils.io.yaml_loader import load_yamls

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
    debug: bool = False,
    session: "Any | None" = None,
) -> Tuple[Dict[str, Any], pd.DataFrame, Optional[pd.DataFrame], Dict[str, pd.DataFrame]]:
    """
    Run HTML-specific document processing steps.

    Accepts either pre-loaded HTML content or a URL. When a URL is given:
    - SEC/Congress URLs are fetched via SecHttpFetcher (rate-limited, bytes).
    - All other URLs are rendered directly by Playwright (JS execution, full quality).

    Args:
        html: Raw HTML string, or None when source_url is provided.
        source_url: URL to fetch/render, or None when html is provided.
        on_stage: Optional callback for progress updates.
        debug: When True, return intermediate DataFrames as a fourth tuple element.

    Returns:
        Tuple of (discovered_metadata, df_lines, df_table_cells, debug_steps).
        debug_steps is an ordered dict of intermediate DataFrames when debug=True,
        empty dict otherwise.
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

    # Step 01 - Box Extraction
    # Try Playwright first; if not installed, fall back to static extractor.
    try:
        from .step_01_box_extractor import BrowserSession, extract_boxes_with_playwright
        _playwright_available = True
    except ImportError:
        logger.warning(
            "Playwright is not installed — falling back to static box extractor. "
            "Accuracy will be degraded: layout coordinates are unavailable and CSS-class styles are not resolved. "
            "Bold or styled headings may go undetected, which can degrade chunk boundaries and hierarchy quality."
        )
        _playwright_available = False

    # Reuse one browser across the extraction attempts below. When a caller
    # (e.g. DocumentParser) supplies a session, reuse it across documents too and
    # leave closing to the caller; otherwise own a session scoped to this call so
    # even a single parse only launches the browser once.
    _owns_session = False
    if _playwright_available and session is None:
        session = BrowserSession()
        _owns_session = True

    try:
        # Steps 01-02 — Box extraction + cleaning, with one networkidle retry.
        #
        # JS-heavy pages sometimes finish `domcontentloaded` before their content
        # is painted, so for live URLs we retry once with `networkidle` whenever
        # the first pass yields nothing usable — either no boxes at all, or none
        # that survive cleaning. The first non-empty cleaned frame wins.
        df_boxes = pd.DataFrame()
        rendered_html = ""
        wait_until = "domcontentloaded"

        while True:
            if _playwright_available:
                try:
                    boxes, rendered_html = extract_boxes_with_playwright(
                        html, source_url, wait_until=wait_until, session=session
                    )
                except Exception as e:
                    # Navigation failed: fetch the bytes ourselves and render them
                    # statically. Drop source_url so we don't retry navigation.
                    if not source_url:
                        raise
                    logger.warning(f"Playwright navigation failed for {source_url}, falling back to http_fetcher: {e}")
                    scraped = fetch_url(source_url)
                    html = scraped.raw_bytes.decode(scraped.encoding or "utf-8", errors="replace")
                    html = _inject_base_url(html, scraped.final_url)
                    source_url = None
                    boxes, rendered_html = extract_boxes_with_playwright(html, source_url=None, session=session)
            else:
                if html is None:
                    # No Playwright to navigate, must have HTML content to proceed.
                    raise ValueError("Playwright is not installed and no HTML content was provided — cannot extract boxes.")
                from .step_01_static_box_extractor import extract_boxes_static
                boxes = extract_boxes_static(html)
                rendered_html = html

            df_boxes = pd.DataFrame(boxes)
            logger.info(f"Step 01 - Box extraction complete ({wait_until}), {len(df_boxes)} boxes")

            # Step 02 - Box Cleaning (skip when nothing was extracted).
            # Static extraction: hr/img are already in DOM order, skip y_top-based reordering.
            if not df_boxes.empty:
                df_boxes = clean_boxes(df_boxes, keep_debug_cols=False, dry_run=False, reorder_by_coordinates=_playwright_available)

            # Got usable boxes, or no retry strategy left → stop.
            if not df_boxes.empty or not (_playwright_available and source_url and wait_until != "networkidle"):
                break
            logger.info("No usable boxes after domcontentloaded — retrying extraction with networkidle")
            wait_until = "networkidle"

        if df_boxes.empty:
            logger.warning("Empty DataFrame after box extraction/cleaning, returning early")
            return {}, df_boxes, None, {}
    finally:
        # The browser is only needed for box extraction above; release it (when we
        # own it) before the heavier downstream processing runs.
        if _owns_session and session is not None:
            session.close()

    # Step 03 - Page Labels (runs on boxes — needs box_id)
    df_boxes, page_labels, page_label_groups = assign_page_labels(
        df_boxes, page_label_config, use_coordinate_filters=_playwright_available
    )

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
    df_lines = merge_boxes_to_lines(df_boxes, remove_single_row_tables=True, merge_by_coordinates=_playwright_available)

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

    debug_steps: Dict[str, pd.DataFrame] = {}
    if debug:
        debug_steps["boxes"] = df_boxes
        debug_steps["lines"] = df_lines
        if df_table_cells is not None:
            debug_steps["table_cells"] = df_table_cells

    return discovered_metadata, df_lines, df_table_cells, debug_steps
