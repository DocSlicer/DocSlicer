# step_03_link_extractor.py

"""
Step 04 – Raw link extraction (PyMuPDF version)

Responsibility:
    - Open a PDF with PyMuPDF (fitz)
    - Extract link annotations via page.get_links()
      - external URLs (uri)
      - internal destinations (page / to)
    - Attach minimal geometry + link metadata:
      - x_left, y_top, x_right, y_bottom
      - link_url (external)
      - link_dest (serialized internal dest / target)
      - link_type: "external" if uri else "internal"

Output columns (per row):
    page_number            (1-based)

    link_id                (global counter across document)
    x_left, y_top, x_right, y_bottom

    link_url               (external URL or None)
    link_dest              (serialized internal dest or None)
    link_type              ("external" | "internal")
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import fitz  # PyMuPDF
import pandas as pd


# ==================================================
# Helpers
# ==================================================

def _rect_to_bbox(rect_obj: Any) -> Tuple[float, float, float, float]:
    """
    Normalize a PyMuPDF Rect or rect-like tuple into (x_left, y_top, x_right, y_bottom).
    """
    if rect_obj is None:
        return 0.0, 0.0, 0.0, 0.0

    # PyMuPDF Rect object
    if hasattr(rect_obj, "x0") and hasattr(rect_obj, "x1"):
        return float(rect_obj.x0), float(rect_obj.y0), float(rect_obj.x1), float(rect_obj.y1)

    # Tuple / list [x0, y0, x1, y1]
    x0, y0, x1, y1 = rect_obj
    return float(x0), float(y0), float(x1), float(y1)


def _serialize_dest_from_link(link_dict: Dict[str, Any]) -> Optional[str]:
    """
    Build a simple, serializable representation of an internal destination
    from a PyMuPDF link dict.

    We consider:
      - link_dict["page"] -> target page index (0-based)
      - link_dict["to"]   -> (x, y, zoom) or similar
      - link_dict["file"] -> external target file (rare)
    """
    page = link_dict.get("page")
    to = link_dict.get("to")
    file = link_dict.get("file")

    if page is None and to is None and file is None:
        return None

    dest_obj = {
        "page": page,
        "to": to,
        "file": file,
    }
    # Simple stringification is enough for later dedupe / logging
    return str(dest_obj)


def _normalize_link_type(uri: Optional[str]) -> str:
    """
    Map raw uri into a link_type value.
    - "external" if a URI is present
    - "internal" otherwise (dest / named / goto)
    """
    return "external" if uri else "internal"


# ==================================================
# Core page extractor
# ==================================================

def _extract_links_for_page(
    page,
    page_number: int,
    *,
    start_link_id: int,
) -> Tuple[List[Dict[str, Any]], int]:
    """
    Extract link dicts for a single PyMuPDF page using page.get_links().

    Returns:
        (list_of_link_dicts, next_link_id)
    """
    link_dicts: List[Dict[str, Any]] = page.get_links() or []

    records: List[Dict[str, Any]] = []
    next_link_id = start_link_id

    for link in link_dicts:
        rect = link.get("from")
        if rect is None:
            continue

        x_left, y_top, x_right, y_bottom = _rect_to_bbox(rect)

        uri = link.get("uri")
        dest_str = _serialize_dest_from_link(link)
        link_type = _normalize_link_type(uri)

        next_link_id += 1

        rec: Dict[str, Any] = {
            "page_number": page_number,

            # Geometry
            "x_left": x_left,
            "x_right": x_right,
            "y_top": y_top,
            "y_bottom": y_bottom,

            # Link identity + metadata
            "link_id": next_link_id,
            "link_url": uri,
            "link_dest": dest_str,
            "link_type": link_type,
        }
        records.append(rec)

    return records, next_link_id


# =============================
# Public API
# =============================

def extract_links(
    pdf_path: str | Path,
    pages_to_process: Optional[List[int]] = None,
) -> pd.DataFrame:
    """
    High-level API:

    Given a PDF path, return a DataFrame with one row per link
    Given a PDF path, return a DataFrame with one row per link.

    Only minimal fields are populated:
      - page_number
      - x_left, x_right, y_top, y_bottom
      - link_id, link_url, link_dest, link_type
    """
    pdf_path = Path(pdf_path).expanduser().resolve()

    all_records: List[Dict[str, Any]] = []
    next_link_id = 0

    with fitz.open(pdf_path) as doc:
        total_pages = doc.page_count

        if pages_to_process is None:
            page_numbers = range(1, total_pages + 1)
        else:
            page_numbers = pages_to_process

        for page_number in page_numbers:
            if page_number < 1 or page_number > total_pages:
                continue

            page = doc.load_page(page_number - 1)

            page_records, next_link_id = _extract_links_for_page(
                page,
                page_number=page_number,
                start_link_id=next_link_id,
            )

            if page_records:
                all_records.extend(page_records)

    if not all_records:
        return pd.DataFrame()

    df = pd.DataFrame(all_records)
    return df
