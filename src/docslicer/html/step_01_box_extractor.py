# d01_box_extractor.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from playwright.sync_api import sync_playwright

# ====== CONFIG ======
HEADLESS = True  # set False to debug visually
BROWSER_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)

# JS lives in document_pipeline/js folder
PIPELINE_DIR = Path(__file__).parent  # document_pipeline/
EXTRACTOR_JS_PATH = PIPELINE_DIR / "js" / "extract_boxes.js"

# ----------------------------
# Public API
# ----------------------------
def extract_boxes_with_playwright(
    html: str,
    source_url: str = None,
) -> tuple[List[Dict[str, Any]], str]:
    """
    Extract boxes from HTML using Playwright + in-page JS extractor.
    
    For URLs (when source_url is provided), navigates to the URL to execute
    client-side JavaScript. For file uploads, sets content directly.

    Args:
        html: HTML content
        page_label_config: Page label patterns config dict loaded from YAML
        source_url: Optional URL - if provided, will navigate instead of setting content

    Returns:
        Tuple of (boxes list, rendered_html string, page_dimensions dict)
    """
    if (html is None and source_url is None) or (html is not None and source_url is not None):
        raise ValueError("Exactly one of 'html' or 'url' must be provided")
    
    js_code = EXTRACTOR_JS_PATH.read_text(encoding="utf-8")
    #js_code = _inject_is_page_label_token(js_code, page_label_config)

    # Use consistent viewport width for coordinate alignment
    VIEWPORT_WIDTH = 1280
    VIEWPORT_HEIGHT = 800
    
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=HEADLESS, args=["--disable-http2"])
        context = browser.new_context(
            user_agent=BROWSER_USER_AGENT,
            viewport={"width": VIEWPORT_WIDTH, "height": VIEWPORT_HEIGHT}
        )
        page = context.new_page()

        # For URLs: navigate to execute client-side JS (React, etc.)
        # For files: set content directly
        if source_url and source_url.startswith(('http://', 'https://')):
            try:
                # Navigate and wait for DOM to be loaded
                # domcontentloaded is more reliable than networkidle for pages with ongoing network activity
                page.goto(source_url, wait_until="domcontentloaded", timeout=30000)
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
        
        # Inject CSS reset to ensure consistent margins (removes default body margin)
        # This must be done BEFORE extracting coordinates
        page.add_style_tag(content="""
            html, body { margin: 0 !important; padding: 0 !important; }
            /* Hide scrollbar to ensure consistent width */
            ::-webkit-scrollbar { display: none; }
            html { scrollbar-width: none; }
        """)
        
        # Wait a moment for styles to apply and fonts to load
        page.wait_for_timeout(500)

        # Get the rendered HTML
        rendered_html = page.content()

        # Get the actual page height after rendering
        page_height = page.evaluate("document.documentElement.scrollHeight")
        page_dimensions = {
            'width': VIEWPORT_WIDTH,
            'height': page_height
        }

        # JS file is an IIFE returning rows
        boxes = page.evaluate(js_code)

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

    return valid_boxes, rendered_html, page_dimensions
