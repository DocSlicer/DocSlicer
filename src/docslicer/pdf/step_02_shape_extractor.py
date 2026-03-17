"""
Step 02 – Raw shape extraction (PyMuPDF version)

Responsibility:
    - Open a PDF with PyMuPDF (fitz)
    - Extract raw geometric shapes via page.get_drawings():
        - rectangles
        - lines
        - curves
    - Attach geometry + raw graphics state:
        - non_stroking_color (fill) - converted to hex format
        - stroking_color (stroke / border) - converted to hex format
        - linewidth
        - fill / stroke flags
    - Deduplicate and filter to avoid shape explosion
    - NO high-level semantics (no "is_background_panel", no "is_table_border", etc.)

Output columns (per row):
    page_number
    shape_id
    shape_type          (e.g. 'rect', 'line', 'curve')

    x_left, y_top, x_right, y_bottom
    width, height
    area

    non_stroking_color  (hex string: #rrggbb or None)
    stroking_color      (hex string: #rrggbb or None)
    linewidth
    fill
    stroke
    paint_op
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
import hashlib

import fitz  # PyMuPDF
import pandas as pd

from ._utils.color_utils import pdf_color_to_hex


def _classify_drawing_type(item: Dict[str, Any]) -> str:
    """
    Classify a PyMuPDF drawing item into 'rect', 'line', or 'curve'.
    
    PyMuPDF uses 'type' field with values like 'f', 's', 'fs', etc.
    We classify based on the geometry in the 'items' list.
    """
    items = item.get("items", [])
    if not items:
        return "other"
    
    # Check if it's a simple rectangle
    # Rectangles typically have 're' (rectangle) commands
    if item.get("type") in ("f", "s", "fs") and len(items) == 1:
        cmd, *coords = items[0]
        if cmd == "re":  # Rectangle command
            return "rect"
    
    # Check for lines (single line segment)
    if len(items) == 2:
        cmd1 = items[0][0] if items[0] else None
        cmd2 = items[1][0] if items[1] else None
        if cmd1 == "m" and cmd2 == "l":  # moveto + lineto
            return "line"
    
    # Everything else is a curve/path
    if any(cmd in ("c", "v", "y", "l") for it in items for cmd in [it[0]] if it):
        return "curve"
    
    return "other"


def _get_drawing_bbox(item: Dict[str, Any]) -> Tuple[float, float, float, float]:
    """
    Extract bounding box from a PyMuPDF drawing item.
    Returns (x0, y0, x1, y1) in PDF coordinates.
    """
    rect = item.get("rect")
    if rect:
        return tuple(rect)
    
    # Fallback: compute from items
    items = item.get("items", [])
    if not items:
        return (0, 0, 0, 0)
    
    xs, ys = [], []
    for cmd_data in items:
        if not cmd_data:
            continue
        # Skip command letter, get coordinate pairs
        coords = cmd_data[1:]
        for i in range(0, len(coords), 2):
            if i + 1 < len(coords):
                xs.append(coords[i])
                ys.append(coords[i + 1])
    
    if not xs or not ys:
        return (0, 0, 0, 0)
    
    return (min(xs), min(ys), max(xs), max(ys))


def _shape_hash(shape_dict: Dict[str, Any]) -> str:
    """
    Create a hash for shape deduplication.
    Based on geometry and visual properties.
    """
    # Round coordinates to avoid floating point differences
    x0 = round(shape_dict["x_left"], 2)
    y0 = round(shape_dict["y_top"], 2)
    x1 = round(shape_dict["x_right"], 2)
    y1 = round(shape_dict["y_bottom"], 2)
    
    # Include colors and linewidth in hash
    key = (
        x0, y0, x1, y1,
        shape_dict.get("raw_shape_type"),
        shape_dict.get("non_stroking_color"),
        shape_dict.get("stroking_color"),
        round(shape_dict.get("linewidth", 0) or 0, 2),
    )
    
    return hashlib.md5(str(key).encode()).hexdigest()


def _drawing_to_dict(
    item: Dict[str, Any],
    *,
    page_number: int,
    raw_shape_id: int,
) -> Optional[Dict[str, Any]]:
    """
    Convert a PyMuPDF drawing item to our normalized dict format.
    Returns None if the item should be filtered out.
    """
    # Classify the shape type
    raw_shape_type = _classify_drawing_type(item)
    
    # Get bounding box
    x0, y0, x1, y1 = _get_drawing_bbox(item)
    
    # Filter out tiny or invalid shapes
    width = abs(x1 - x0)
    height = abs(y1 - y0)
    if width < 0.1 and height < 0.1:
        return None
    
    area = width * height
    
    # Extract colors
    fill_color = item.get("fill")
    stroke_color = item.get("color")
    
    non_stroking_color = pdf_color_to_hex(fill_color)
    stroking_color = pdf_color_to_hex(stroke_color)
    
    # Extract linewidth
    linewidth = item.get("width")
    if linewidth is not None:
        try:
            linewidth = float(linewidth)
        except (TypeError, ValueError):
            linewidth = None
    
    # Determine fill/stroke flags
    # PyMuPDF type: 'f' = fill, 's' = stroke, 'fs' = fill+stroke
    item_type = item.get("type", "")
    fill = "f" in item_type
    stroke = "s" in item_type
    
    return {
        "page_number": page_number,
        "raw_shape_id": raw_shape_id,
        "raw_shape_type": raw_shape_type,
        "x_left": min(x0, x1),
        "x_right": max(x0, x1),
        "y_top": min(y0, y1),
        "y_bottom": max(y0, y1),
        "width": width,
        "height": height,
        "area": area,
        "non_stroking_color": non_stroking_color,
        "stroking_color": stroking_color,
        "linewidth": linewidth,
        "fill": fill,
        "stroke": stroke,
        "paint_op": item_type or None,
    }


def _extract_raw_shapes_for_page(
    page: fitz.Page,
    page_number: int,
    *,
    include_types: Optional[Sequence[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Extract shape dicts for a single PyMuPDF page.
    
    Returns list of dicts directly for faster DataFrame construction.
    Deduplicates shapes to avoid explosion.
    
    Note: raw_shape_id will be assigned later after sorting all pages.
    """
    raw_shapes: List[Dict[str, Any]] = []
    seen_hashes = set()
    
    # Get all drawing paths from the page
    drawings = page.get_drawings()
    
    if not drawings:
        return raw_shapes
    
    for drawing in drawings:
        shape_dict = _drawing_to_dict(
            drawing,
            page_number=page_number,
            raw_shape_id=0,  # Temporary, will be reassigned
        )
        
        if shape_dict is None:
            continue
        
        # Filter by type if requested
        if include_types and shape_dict["raw_shape_type"] not in include_types:
            continue
        
        # Deduplicate
        shape_hash = _shape_hash(shape_dict)
        if shape_hash in seen_hashes:
            continue
        
        seen_hashes.add(shape_hash)
        raw_shapes.append(shape_dict)
    
    return raw_shapes


def extract_shapes(
    pdf_path: str | Path,
    pages_to_process: Optional[List[int]] = None,
    include_types: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """
    Extract all shapes from a PDF and return a DataFrame with one row per shape.

    Args:
        pdf_path: Path to PDF file
        pages_to_process: Page numbers (1-indexed), or None for all pages
        include_types: Shape types to include — None for all, or a subset of
            ``['rect', 'line', 'curve']``

    Shapes are sorted by page_number → y_top → x_left.
    ``raw_shape_id`` is assigned sequentially (1-based) after sorting.
    """
    pdf_path = Path(pdf_path).expanduser().resolve()
    
    all_shapes: List[Dict[str, Any]] = []
    
    with fitz.open(pdf_path) as doc:
        total_pages = len(doc)
        
        if pages_to_process is None:
            page_numbers = range(1, total_pages + 1)
        else:
            page_numbers = pages_to_process
        
        for page_number in page_numbers:
            if page_number < 1 or page_number > total_pages:
                continue
            
            page = doc[page_number - 1]  # 0-indexed
            
            page_shapes = _extract_raw_shapes_for_page(
                page,
                page_number=page_number,
                include_types=include_types,
            )
            
            all_shapes.extend(page_shapes)
    
    if not all_shapes:
        return pd.DataFrame()
    
    # Create DataFrame directly from list of dicts
    df = pd.DataFrame(all_shapes)
    
    # Sort by page_number, then y_top, then x_left for logical ordering
    df = df.sort_values(
        by=["page_number", "y_top", "x_left"],
        ignore_index=True
    )
    
    # Assign sequential raw_shape_id (1-based, no gaps)
    df["raw_shape_id"] = range(1, len(df) + 1)
    
    return df