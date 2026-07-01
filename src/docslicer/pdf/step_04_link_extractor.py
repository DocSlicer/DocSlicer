"""
Step 04 – Raw link extraction

Output columns:
    page_number, link_id,
    x_left, y_top, x_right, y_bottom,
    link_url, link_dest, link_type

Coordinate system: FPDFLink_GetAnnotRect returns an FS_RECTF in PDF space
(y increases upward). We convert to screen space (y increases downward):
y_top = page_height - rect.top.
"""

from __future__ import annotations

import ctypes
from ctypes import c_int, byref

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pypdfium2 as pdfium
import pypdfium2.raw as pdfium_c
import pandas as pd


# ── Module-level raw API references ──────────────────────────────────────────

_link_enumerate  = pdfium_c.FPDFLink_Enumerate
_link_get_rect   = pdfium_c.FPDFLink_GetAnnotRect
_link_get_action = pdfium_c.FPDFLink_GetAction
_link_get_dest   = pdfium_c.FPDFLink_GetDest
_action_get_type = pdfium_c.FPDFAction_GetType
_action_get_uri  = pdfium_c.FPDFAction_GetURIPath
_dest_get_page   = pdfium_c.FPDFDest_GetDestPageIndex

_URI_TYPE = pdfium_c.PDFACTION_URI  # 3


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_uri(doc: Any, action: Any) -> Optional[str]:
    buflen = _action_get_uri(doc, action, None, 0)
    if buflen <= 1:
        return None
    buf = ctypes.create_string_buffer(buflen)
    _action_get_uri(doc, action, buf, buflen)
    uri = buf.value.decode("utf-8", errors="replace").strip()
    return uri or None


def _serialize_dest(doc: Any, link: Any) -> Optional[str]:
    dest = _link_get_dest(doc, link)
    if not dest:
        return None
    page_index = _dest_get_page(doc, dest)
    if page_index < 0:
        return None
    return str({"page": page_index})


# ── Per-page extraction ───────────────────────────────────────────────────────

def _extract_links_for_page(
    doc: pdfium.PdfDocument,
    page: pdfium.PdfPage,
    page_number: int,
    *,
    start_link_id: int,
) -> Tuple[List[Dict[str, Any]], int]:
    records: List[Dict[str, Any]] = []
    next_link_id = start_link_id
    page_height = page.get_height()

    start_pos = c_int(0)
    link = pdfium_c.FPDF_LINK()
    rect = pdfium_c.FS_RECTF()

    while _link_enumerate(page, byref(start_pos), byref(link)):
        _link_get_rect(link, byref(rect))

        # PDF coords (y upward) → screen coords (y downward).
        # Normalize x and y because some PDFs store rects with swapped corners.
        x_left   = min(rect.left, rect.right)
        x_right  = max(rect.left, rect.right)
        sy1 = page_height - rect.top
        sy2 = page_height - rect.bottom
        y_top    = min(sy1, sy2)
        y_bottom = max(sy1, sy2)

        # Determine URL / dest
        uri: Optional[str] = None
        dest_str: Optional[str] = None

        action = _link_get_action(link)
        if action:
            atype = _action_get_type(action)
            if atype == _URI_TYPE:
                uri = _get_uri(doc, action)
            else:
                dest_str = _serialize_dest(doc, link)
        else:
            # Link with no action — dest only
            dest_str = _serialize_dest(doc, link)

        next_link_id += 1
        records.append({
            "page_number": page_number,
            "link_id":     next_link_id,
            "x_left":      x_left,
            "x_right":     x_right,
            "y_top":       y_top,
            "y_bottom":    y_bottom,
            "link_url":    uri,
            "link_dest":   dest_str,
            "link_type":   "external" if uri else "internal",
        })

    return records, next_link_id


# ── Public API ────────────────────────────────────────────────────────────────

def extract_links(
    pdf_path: str | Path,
    pages_to_process: Optional[List[int]] = None,
) -> pd.DataFrame:
    """
    Extract all link annotations from a PDF and return a DataFrame.

    Args:
        pdf_path: Path to PDF file
        pages_to_process: Page numbers (1-indexed), or None for all pages
    """
    pdf_path = Path(pdf_path).expanduser().resolve()

    all_records: List[Dict[str, Any]] = []
    next_link_id = 0

    with pdfium.PdfDocument(pdf_path) as doc:
        total_pages = len(doc)

        page_numbers = (
            range(1, total_pages + 1) if pages_to_process is None else pages_to_process
        )

        for page_number in page_numbers:
            if page_number < 1 or page_number > total_pages:
                continue

            try:
                page = doc[page_number - 1]
            except Exception:
                continue
            page_records, next_link_id = _extract_links_for_page(
                doc, page, page_number, start_link_id=next_link_id
            )
            all_records.extend(page_records)

    if not all_records:
        return pd.DataFrame()

    return pd.DataFrame(all_records)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python step_04_link_extractor.py <pdf_path>")
        sys.exit(1)

    df = extract_links(sys.argv[1])

    if df.empty:
        print("No links found")
    else:
        print(f"Found {len(df)} links")
        print(df[["page_number", "link_type", "link_url", "link_dest"]].head(10))
