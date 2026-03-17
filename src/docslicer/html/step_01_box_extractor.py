# d01_box_extractor.py
from __future__ import annotations

import re
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
# JS injection: isPageLabelToken
# ----------------------------
def _generate_js_is_page_label_token(config: dict) -> str:
    """
    Generate JavaScript isPageLabelToken function from config dict (YAML).
    Ensures Python and JS share identical pattern logic.

    Expected config schema:
      {
        "max_length": 8,
        "patterns": [
          {"name": "...", "regex": "...", "flags": "i" or ""}
        ]
      }
    """
    max_length = int(config.get("max_length", 8))
    patterns = config.get("patterns", [])

    tests: list[str] = []
    for i, p in enumerate(patterns):
        name = p.get("name", f"p{i}")
        regex = p.get("regex", "")
        flags = p.get("flags", "")

        # JS regex literal
        js_regex = f"/{regex}/{flags}" if flags else f"/{regex}/"

        # Join with || except last
        sep = " ||" if i < len(patterns) - 1 else ""
        tests.append(f"{js_regex}.test(s){sep}  /* {name} */")

    tests_joined = "\n        ".join(tests) if tests else "false"

    # IMPORTANT: empty text is NOT a page label token
    # (prevents over-triggering HR-border on empty bordered containers)
    return f"""// AUTO-GENERATED from page_label_patterns.yaml - DO NOT EDIT DIRECTLY
  const isPageLabelToken = (t) => {{
    const s = normalize(t);
    if (!s) return false;
    if (s.length > {max_length}) return false;
    return (
        {tests_joined}
    );
  }};
"""


def _inject_is_page_label_token(js_code: str, page_label_config: dict) -> str:
    start_marker = "// __MF_INJECT_IS_PAGE_LABEL_TOKEN_START__"
    end_marker = "// __MF_INJECT_IS_PAGE_LABEL_TOKEN_END__"

    start_idx = js_code.find(start_marker)
    end_idx = js_code.find(end_marker)

    if start_idx == -1 or end_idx == -1 or end_idx <= start_idx:
        raise ValueError(
            "Could not find injection markers in extract_boxes.js. "
            "Add the marker block:\n"
            f"{start_marker}\n...\n{end_marker}"
        )

    generated_fn = _generate_js_is_page_label_token(page_label_config)

    # Replace everything between markers (exclusive) with generated_fn
    before = js_code[: start_idx + len(start_marker)]
    after = js_code[end_idx:]

    # Ensure clean spacing
    return before + "\n" + generated_fn + "\n" + after

# ----------------------------
# Public API
# ----------------------------
def extract_boxes_with_playwright(
    html: str,
    page_label_dict: dict,
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
        Tuple of (boxes list, rendered_html string) - rendered_html is what was actually parsed
    """
    if (html is None and source_url is None) or (html is not None and source_url is not None):
        raise ValueError("Exactly one of 'html' or 'url' must be provided")
    
    js_code = EXTRACTOR_JS_PATH.read_text(encoding="utf-8")
    #js_code = _inject_is_page_label_token(js_code, page_label_config)

    # Use consistent viewport width for coordinate alignment
    VIEWPORT_WIDTH = 1280
    VIEWPORT_HEIGHT = 800
    
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=HEADLESS)
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
        
        # Capture full-page screenshot for pixel-perfect overlay
        # This is the ONLY way to guarantee alignment
        import base64
        screenshot_bytes = page.screenshot(full_page=True, type="png")
        screenshot_base64 = base64.b64encode(screenshot_bytes).decode('utf-8')
        
        # Get the actual page height after rendering
        # IMPORTANT: Use VIEWPORT_WIDTH for width since screenshot is captured at viewport width
        # scrollWidth can differ due to content layout, but screenshot width = viewport width
        page_height = page.evaluate("document.documentElement.scrollHeight")
        page_dimensions = {
            'width': VIEWPORT_WIDTH,  # Screenshot width is always viewport width
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

    # Return boxes, rendered HTML, screenshot, and dimensions
    return valid_boxes, rendered_html, screenshot_base64, page_dimensions
