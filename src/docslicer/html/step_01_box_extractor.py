# d01_box_extractor.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from playwright.sync_api import sync_playwright

from docslicer.scraping.config import (
    BROWSER_ARGS,
    BROWSER_USER_AGENT,
    COOKIE_CONSENT_JS_PATH,
    STEALTH_INIT_JS_PATH,
)

# ====== CONFIG ======
HEADLESS = True  # set False to debug visually

# JS lives beside this HTML extraction step.
PIPELINE_DIR = Path(__file__).parent
EXTRACTOR_JS_PATH = PIPELINE_DIR / "extract_boxes.js"

# ----------------------------
# Public API
# ----------------------------
def extract_boxes_with_playwright(
    html: str,
    source_url: str = None,
    wait_until: str = "domcontentloaded",
) -> tuple[List[Dict[str, Any]], str]:
    """
    Extract boxes from HTML using Playwright + in-page JS extractor.

    Args:
        html: Raw HTML string, or None when source_url is provided.
        source_url: URL to navigate to, or None when html is provided.
        wait_until: Playwright navigation wait strategy ("domcontentloaded" or "networkidle").

    Returns:
        Tuple of (boxes list, rendered_html string)
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
    
    with sync_playwright() as p:
        # Prefer real Chrome (less detectable) and fall back to bundled Chromium.
        try:
            browser = p.chromium.launch(headless=HEADLESS, channel="chrome", args=BROWSER_ARGS)
        except Exception:
            browser = p.chromium.launch(headless=HEADLESS, args=BROWSER_ARGS)
        context = browser.new_context(
            user_agent=BROWSER_USER_AGENT,
            viewport={"width": VIEWPORT_WIDTH, "height": VIEWPORT_HEIGHT},
            extra_http_headers={"Accept-Language": "en-US,en;q=0.9"},
            bypass_csp=True,
        )
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
                    print(f"Warning: Failed to navigate to {source_url}, falling back to set_content: {e}")
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

        browser.close()

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
