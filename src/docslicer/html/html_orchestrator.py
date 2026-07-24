# SPDX-FileCopyrightText: 2026 Market Framer Inc.
# SPDX-License-Identifier: AGPL-3.0-only
# html_orchestrator.py - HTML-specific document processing pipeline
"""
HTML-specific pipeline steps.

These steps extract and process HTML documents into a lines_df format
that can then be processed by the shared orchestrator.

Pipeline Steps:
    01. Box Extraction (Playwright) - Extract visual boxes from HTML
    02. Box Cleaning - Clean boxes, add text features
    03. Page Labels - Detect page numbers/labels
    04. Line Merger - Merge boxes into visual lines
    05. Table Extraction - Extract tables from HTML
    06. Style Prefiller - Assign block_type from struct_ancestors (code, heading, block_quote)

Note: TOC, Exhibit, Doc Region, and Hierarchy detection are handled
by the shared orchestrator.
"""
import logging
from typing import Callable, NamedTuple, Optional, Dict, Any

import pandas as pd

logger = logging.getLogger(__name__)

# HTML-specific step imports
from .step_02_box_cleaner import clean_boxes
from .step_03_page_label_detector import assign_page_labels
from .step_04_line_builder import merge_boxes_to_lines
from .step_05_table_extractor import extract_table_cells
from .step_06_style_prefiller import prefill_styles

# Config loaders
from .._utils.io.yaml_loader import load_yamls
from .._utils.timing import timed_step
from .._utils.safe_call import safe_enrich

# Metadata utilities — native channel (HTML <head>) + shared text/consolidate/page steps
from .native_metadata import extract_native_metadata
from ..metadata import add_page_info, add_text_fallbacks, consolidate

# Scraping
from ..scraping.dispatcher import fetch_url, _is_sec_url


class HtmlPipelineResult(NamedTuple):
    """Structured result of :func:`run_pipeline`."""

    discovered_metadata: Dict[str, Any]
    df_lines: pd.DataFrame
    df_table_cells: Optional[pd.DataFrame]
    debug_steps: Dict[str, pd.DataFrame]


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
    use_browser: bool = True,
) -> HtmlPipelineResult:
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
        use_browser: When False, skip Playwright entirely and use the static
            (BeautifulSoup) box extractor, even if Playwright is installed —
            ~15x faster, no browser launch, but x_right/y_top/y_bottom/width/
            height are all 0.0 (no layout coordinates) and only inline-style +
            semantic-tag typography is resolved (no CSS class rules / external
            stylesheets). A URL source still needs plain HTTP fetching, which
            happens automatically the same way it does for the no-Playwright
            fallback. Best for inline-style-heavy documents (SEC filings,
            Word-exported HTML, legal documents); degrades on CSS-class-heavy
            modern pages.

    Returns:
        HtmlPipelineResult(discovered_metadata, df_lines, df_table_cells, debug_steps).
        debug_steps is an ordered dict of intermediate DataFrames when debug=True,
        empty dict otherwise.
    """
    _, page_label_config, _, _ = load_yamls()

    # ============================================================
    # Fetch step — resolve source_url to html bytes if needed
    # ============================================================
    if source_url and html is None:
        if _is_sec_url(source_url) or not use_browser:
            # SEC, or use_browser=False: fetch bytes via plain HTTP (rate-limited
            # for SEC), render as static HTML. Without a browser there is no
            # page.goto to do this fetch for us. Inject <base href> so relative
            # URLs resolve correctly (Playwright's about:blank context otherwise
            # returns "about:blank/..."; the static extractor never resolves them).
            with timed_step("fetch_html_http", logger=logger):
                scraped = fetch_url(source_url)
                html = scraped.raw_bytes.decode(scraped.encoding or "utf-8", errors="replace")
                html = _inject_base_url(html, scraped.final_url)
            source_url = None  # box extractor will use set_content
        # else: non-SEC, browser available → keep source_url=source_url, html=None
        #       box extractor will page.goto(source_url) for full JS rendering
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
        on_stage("extract_elements")

    # Step 01 - Box Extraction
    # Try Playwright first; if not installed (or the caller opted out via
    # use_browser=False), fall back to the static extractor.
    if not use_browser:
        _playwright_available = False
    else:
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
            step_name = f"box_extraction_{wait_until}" if _playwright_available else "box_extraction_static"
            with timed_step(step_name, logger=logger):
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
                logger.info(f"Step 01 - Box extraction complete ({step_name}), {len(df_boxes)} boxes")

            # Step 02 - Box Cleaning (skip when nothing was extracted).
            # Static extraction: hr/img are already in DOM order, skip y_top-based reordering.
            if not df_boxes.empty:
                with timed_step("box_cleaning", logger=logger):
                    df_boxes = clean_boxes(df_boxes, keep_debug_cols=False, dry_run=False, reorder_by_coordinates=_playwright_available)

            # Got usable boxes, or no retry strategy left → stop.
            if not df_boxes.empty or not (_playwright_available and source_url and wait_until != "networkidle"):
                break
            logger.info("No usable boxes after domcontentloaded — retrying extraction with networkidle")
            wait_until = "networkidle"

        if df_boxes.empty:
            logger.warning("Empty DataFrame after box extraction/cleaning, returning early")
            return HtmlPipelineResult({}, df_boxes, None, {})
    finally:
        # The browser is only needed for box extraction above; release it (when we
        # own it) before the heavier downstream processing runs.
        if _owns_session and session is not None:
            session.close()

    # Step 03 - Page Labels (runs on boxes — needs box_id)
    with timed_step("page_label_detection", logger=logger):
        df_boxes, page_labels, page_label_groups = assign_page_labels(
            df_boxes, page_label_config, use_coordinate_filters=_playwright_available
        )

    discovered_metadata: Dict[str, Any] = {}
    discovered_metadata["rendered_html"] = rendered_html

    with timed_step("page_info", logger=logger):
        # HTML has no scanned/OCR concept — page geometry + char totals only.
        safe_enrich(
            add_page_info, discovered_metadata, df_boxes,
            fallback={"page_count": 1, "chars": 0},
            logger=logger,
        )

    # Convert pixels to points (96 px = 72 pt)
    PX_TO_PT = 0.75
    geometry_cols = ["x_left", "x_right", "y_top", "y_bottom", "width", "height", "font_size"]
    for col in geometry_cols:
        if col in df_boxes.columns:
            df_boxes[col] = df_boxes[col] * PX_TO_PT

    # ============================================================
    # STAGE: STRUCTURE (Steps 04-06)
    # ============================================================
    if on_stage:
        on_stage("process_layouts")

    # Step 04 - Line Builder (boxes → lines); must run before table extractor
    # so that the final reindexed table_id values are available for ID matching.
    with timed_step("line_building", logger=logger):
        df_lines = merge_boxes_to_lines(df_boxes, remove_single_row_tables=True, merge_by_coordinates=_playwright_available)

    if on_stage:
        on_stage("extract_tables")

    # Step 05 - Table Extractor (uses df_lines.original_table_id for table_id sync)
    with timed_step("table_extraction", logger=logger):
        df_table_cells = extract_table_cells(
            df_lines=df_lines,
            rendered_html=rendered_html,
        )

    # Step 06 - Style Prefiller (block_type from struct_ancestors: code, heading,
    # block_quote). Runs after table extraction so table block_type is already set
    # and preserved; only fills rows with no existing block_type.
    with timed_step("style_prefill", logger=logger):
        df_lines = prefill_styles(df_lines)

    with timed_step("document_metadata", logger=logger):
        # Native channel — the page's own <head> metadata (title/author/lang/dates).
        discovered_metadata.update(extract_native_metadata(rendered_html))
        # Text channel — heuristics over the parsed body, a fallback for the three
        # fields that native <head> metadata routinely omits (SEC filings, etc.).
        safe_enrich(
            add_text_fallbacks, discovered_metadata, df_lines,
            fallback={"author_text": None, "title_text": None, "language_text": None},
            logger=logger,
        )
        # Fold both channels into the final title / author / language.
        consolidate(discovered_metadata)

    # ============================================================
    # Finalize metadata
    # ============================================================
    discovered_metadata["is_password_protected"] = False
    discovered_metadata["has_ocr"] = False
    discovered_metadata["needs_ocr"] = False
    discovered_metadata["is_scanned"] = False

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

    return HtmlPipelineResult(discovered_metadata, df_lines, df_table_cells, debug_steps)
