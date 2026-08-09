# SPDX-FileCopyrightText: 2026 Market Framer Inc.
# SPDX-License-Identifier: AGPL-3.0-only
"""Extract text boxes from rendered HTML using a Playwright browser session."""

# d01_box_extractor.py
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List

from playwright.sync_api import sync_playwright

from docslicer.scraping.config import (
    BROWSER_ARGS,
    BROWSER_USER_AGENT,
    COOKIE_CONSENT_JS_PATH,
    STEALTH_INIT_JS_PATH,
)

_log = logging.getLogger(__name__)

# ====== CONFIG ======
HEADLESS = True  # set False to debug visually

# JS lives beside this HTML extraction step.
PIPELINE_DIR = Path(__file__).parent
EXTRACTOR_JS_PATH = PIPELINE_DIR / "extract_boxes.js"


class BrowserUnavailableError(RuntimeError):
    """No browser could be launched (binaries not installed, sandbox, missing libs).

    Distinct from a navigation or extraction failure: the browser itself is
    unusable, so retrying through Playwright cannot help and callers should
    fall back to the static extractor.
    """


# ----------------------------
# Browser session (reusable across documents and across retries)
# ----------------------------
class BrowserSession:
    """Owns a single Playwright browser process, reused across extractions.

    Launching Chromium is the dominant fixed cost of HTML extraction, so a
    long-lived session amortizes it across many documents (when held by a
    DocumentParser) and across the retry attempts within a single document.
    A fresh browser *context* is created per extraction, preserving the
    per-document isolation (cookies, stealth init script) that a brand-new
    browser used to provide. The browser launches lazily on first extraction,
    so holding a session that is never used for HTML costs nothing.
    """

    def __init__(self) -> None:
        self._pw = None
        self._browser = None

    def _ensure_browser(self):
        if self._browser is None:
            try:
                if self._pw is None:
                    self._pw = sync_playwright().start()
                # Prefer real Chrome (less detectable) and fall back to bundled Chromium.
                try:
                    self._browser = self._pw.chromium.launch(headless=HEADLESS, channel="chrome", args=BROWSER_ARGS)
                except Exception:
                    self._browser = self._pw.chromium.launch(headless=HEADLESS, args=BROWSER_ARGS)
            except Exception as e:
                # Playwright the package is installed but no browser can run here —
                # binaries never downloaded (`playwright install chromium`), missing
                # system libraries, or a sandbox that forbids launching one.
                raise BrowserUnavailableError(str(e)) from e
        return self._browser

    def extract(
        self,
        html: str | None,
        source_url: str | None = None,
        wait_until: str = "domcontentloaded",
    ) -> tuple[List[Dict[str, Any]], str]:
        return _extract_with_browser(self._ensure_browser(), html, source_url, wait_until)

    def close(self) -> None:
        if self._browser is not None:
            try:
                self._browser.close()
            finally:
                self._browser = None
        if self._pw is not None:
            try:
                self._pw.stop()
            finally:
                self._pw = None

    def __enter__(self) -> "BrowserSession":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


# ----------------------------
# Public API
# ----------------------------
def extract_boxes_with_playwright(
    html: str,
    source_url: str = None,
    wait_until: str = "domcontentloaded",
    session: "BrowserSession | None" = None,
) -> tuple[List[Dict[str, Any]], str]:
    """
    Extract boxes from HTML using Playwright + in-page JS extractor.

    Args:
        html: Raw HTML string, or None when source_url is provided.
        source_url: URL to navigate to, or None when html is provided.
        wait_until: Playwright navigation wait strategy ("domcontentloaded" or "networkidle").
        session: Optional reusable BrowserSession. When provided, its browser is
            reused (a fresh context is created per call). When None, a temporary
            browser is launched and closed for this single extraction — preserving
            the original standalone behavior.

    Returns:
        Tuple of (boxes list, rendered_html string)
    """
    if session is not None:
        return session.extract(html, source_url, wait_until)

    session = BrowserSession()
    try:
        return session.extract(html, source_url, wait_until)
    finally:
        session.close()


def _extract_with_browser(
    browser,
    html: str | None,
    source_url: str | None,
    wait_until: str,
) -> tuple[List[Dict[str, Any]], str]:
    """Run one extraction on an already-launched browser.

    Creates a fresh context/page, performs navigation + in-page JS extraction,
    and tears down the *context* (but not the browser) before returning, so the
    browser can be reused for the next extraction.
    """
    if (html is None and source_url is None) or (html is not None and source_url is not None):
        raise ValueError("Exactly one of 'html' or 'url' must be provided")

    js_code = EXTRACTOR_JS_PATH.read_text(encoding="utf-8")
    cookie_consent_js = COOKIE_CONSENT_JS_PATH.read_text(encoding="utf-8")
    stealth_init_js = STEALTH_INIT_JS_PATH.read_text(encoding="utf-8")
    #js_code = _inject_is_page_label_token(js_code, page_label_config)

    # Use consistent viewport width for coordinate alignment
    VIEWPORT_WIDTH = 1280
    VIEWPORT_HEIGHT = 800

    context = browser.new_context(
        user_agent=BROWSER_USER_AGENT,
        viewport={"width": VIEWPORT_WIDTH, "height": VIEWPORT_HEIGHT},
        extra_http_headers={"Accept-Language": "en-US,en;q=0.9"},
        bypass_csp=True,
    )
    try:
        context.add_init_script(stealth_init_js)
        page = context.new_page()

        # For URLs: navigate to execute client-side JS (React, etc.)
        # For files: set content directly
        navigated_url = False
        if source_url and source_url.startswith(('http://', 'https://')):
            try:
                page.goto(source_url, wait_until=wait_until, timeout=60000)
                navigated_url = True
            except Exception as e:
                # If we have html as fallback, use it
                if html:
                    _log.warning("Failed to navigate to %s, falling back to set_content: %s", source_url, e)
                    page.set_content(html, wait_until="load")
                else:
                    raise RuntimeError(f"Failed to load URL {source_url}: {e}") from e
        else:
            # For file uploads or when URL is not provided
            page.set_content(html, wait_until="load")

        # For URL navigation, wait for fonts to finish loading before measuring layout.
        # set_content("load") already guarantees this; goto("domcontentloaded") does not.
        if navigated_url:
            page.evaluate("document.fonts.ready")
            # Dismiss cookie banners only when they visually block the page.
            dismissed = page.evaluate(cookie_consent_js)
            if dismissed:
                page.wait_for_timeout(600)  # let dismiss animation finish

        # Inject CSS reset to ensure consistent margins (removes default body margin)
        # This must be done BEFORE extracting coordinates
        page.add_style_tag(content="""
            html, body { margin: 0 !important; padding: 0 !important; }
            /* Hide scrollbar to ensure consistent width */
            ::-webkit-scrollbar { display: none; }
            html { scrollbar-width: none; }
        """)

        # Run extraction. The JS also annotates every <table> with
        # data-docslicer-table-id so Python can look up tables by their JS
        # table_id rather than by document-order position (which diverges for
        # nested tables). Capture rendered_html AFTER so annotations are present.
        boxes = page.evaluate(js_code)

        # Get the rendered HTML (includes data-docslicer-table-id attributes)
        rendered_html = page.content()

        # Get the actual page height after rendering
        page_height = page.evaluate("document.documentElement.scrollHeight")
        page_dimensions = {
            'width': VIEWPORT_WIDTH,
            'height': page_height
        }

        # Pause to inspect the rendered page (press Enter to continue)
        #input("⏸️  Browser window open - Press Enter to close and continue...")
    finally:
        context.close()

    # Defensive: ensure plain python primitives
    if not isinstance(boxes, list):
        raise ValueError(f"JS extractor returned unexpected type: {type(boxes)}")

    # Filter and clean boxes
    valid_boxes = []
    for box in boxes:
        box.pop('style', None)

        # Get coordinates (with defaults)
        x_left = box.get('x_left', 0)
        y_top = box.get('y_top', 0)
        x_right = box.get('x_right', 0)
        y_bottom = box.get('y_bottom', 0)

        # Skip boxes with invalid coordinates (hidden elements, off-screen, etc.)
        if x_left < 0 or y_top < 0:
            continue
        if x_right <= x_left or y_bottom <= y_top:
            continue
        if x_right > VIEWPORT_WIDTH * 2:  # Way off screen
            continue

        # Store the actual page dimensions (not viewport - actual rendered size)
        box['page_width'] = page_dimensions['width']
        box['page_height'] = page_dimensions['height']
        valid_boxes.append(box)

    return valid_boxes, rendered_html
