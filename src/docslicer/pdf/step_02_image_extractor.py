"""
Step 02 – Image extraction (PyMuPDF version)

Responsibility:
    - Open a PDF with PyMuPDF (fitz)
    - Extract image metadata via page.get_images() and doc.extract_image()
    - Capture both placement info (bbox) and intrinsic image properties
    - NO high-level semantics (no "is_logo", no "is_chart", etc.)

Output columns (per row):
    page_number
    image_id
    xref                    (PDF object reference number)
    
    # Placement on page (where the image appears)
    x_left, y_top, x_right, y_bottom
    width, height           (display size in PDF points)
    area
    
    # Intrinsic image properties
    image_width             (actual image pixels)
    image_height            (actual image pixels)
    bpc                     (bits per component)
    colorspace              (integer: 1=gray, 3=RGB, 4=CMYK, etc.)
    colorspace_name         (string: DeviceGray, DeviceRGB, etc.)
    
    # Image format and encoding
    ext                     (file extension: jpeg, png, tiff, etc.)
    filter                  (compression: DCTDecode, FlateDecode, etc.)
    
    # Transparency/masking
    smask                   (xref of soft mask, 0 if none)
    has_transparency        (boolean)
    
    # Resolution (calculated from display size vs actual pixels)
    dpi_x
    dpi_y
    
    # Optional: image bytes (if needed for further processing)
    # image_bytes          (binary data - usually excluded from df)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import fitz  # PyMuPDF
import pandas as pd


def _colorspace_to_name(cs_int: int) -> str:
    """
    Convert PyMuPDF colorspace integer to readable name.
    Based on PyMuPDF documentation.
    """
    colorspace_map = {
        0: "None",
        1: "DeviceGray",
        3: "DeviceRGB",
        4: "DeviceCMYK",
    }
    return colorspace_map.get(cs_int, f"Colorspace({cs_int})")


def _calculate_dpi(
    display_width: float,
    display_height: float,
    image_width: int,
    image_height: int,
) -> Tuple[float, float]:
    """
    Calculate DPI from display dimensions (PDF points) vs actual image pixels.
    
    1 inch = 72 PDF points
    DPI = pixels / inches = pixels / (points / 72)
    """
    if display_width <= 0 or display_height <= 0:
        return 0.0, 0.0
    
    dpi_x = (image_width * 72.0) / display_width
    dpi_y = (image_height * 72.0) / display_height
    
    return round(dpi_x, 2), round(dpi_y, 2)


def _extract_image_metadata(
    doc: fitz.Document,
    page: fitz.Page,
    img_info: tuple,
    *,
    page_number: int,
    image_id: int,
) -> Optional[Dict[str, Any]]:
    """
    Extract comprehensive metadata for a single image.
    
    img_info tuple from page.get_images(full=True):
        (xref, smask, width, height, bpc, colorspace, alt_colorspace, name, filter, referencer)
    """
    try:
        xref = img_info[0]
        smask = img_info[1]
        image_width = img_info[2]
        image_height = img_info[3]
        bpc = img_info[4]  # bits per component
        colorspace = img_info[5]
        # alt_colorspace = img_info[6]  # alternative colorspace (rarely used)
        # name = img_info[7]  # image name in PDF
        img_filter = img_info[8]  # compression filter
        # referencer = img_info[9]  # xref of the referencing object
        
        # Get bounding box of where image appears on page
        # Use get_image_rects (preferred method) which returns list of rects
        # Note: get_image_bbox was unreliable in some PyMuPDF versions
        try:
            bbox_rects = page.get_image_rects(xref)
            
            if not bbox_rects:
                # Try alternative method: get_image_bbox (single instance)
                bbox_list = page.get_image_bbox(xref)
                
                if not bbox_list or not bbox_list.is_valid:
                    # Image is referenced but not displayed on this page
                    return None
                
                bbox = bbox_list
            else:
                # Use first occurrence (image may appear multiple times)
                bbox = bbox_rects[0]
        except Exception:
            # Skip images with invalid bounding boxes
            return None
        
        x_left = float(bbox.x0)
        y_top = float(bbox.y0)
        x_right = float(bbox.x1)
        y_bottom = float(bbox.y1)
        
        display_width = x_right - x_left
        display_height = y_bottom - y_top
        area = display_width * display_height
        
        # Calculate DPI
        dpi_x, dpi_y = _calculate_dpi(
            display_width, display_height,
            image_width, image_height
        )
        
        # Extract detailed image info including format
        img_dict = doc.extract_image(xref)
        ext = img_dict.get("ext", "unknown")
        colorspace_name = img_dict.get("cs-name", _colorspace_to_name(colorspace))
        
        # Determine if image has transparency
        has_transparency = smask > 0 or ext == "png"
        
        return {
            "page_number": page_number,
            "image_id": image_id,
            "xref": xref,
            
            # Display position
            "x_left": x_left,
            "x_right": x_right,
            "y_top": y_top,
            "y_bottom": y_bottom,
            "width": display_width,
            "height": display_height,
            "area": area,
            
            # Intrinsic properties
            "image_width": image_width,
            "image_height": image_height,
            "bpc": bpc,
            "colorspace": colorspace,
            "colorspace_name": colorspace_name,
            
            # Format
            "ext": ext,
            "filter": img_filter,
            
            # Transparency
            "smask": smask,
            "has_transparency": has_transparency,
            
            # Resolution
            "dpi_x": dpi_x,
            "dpi_y": dpi_y,
        }
        
    except Exception:
        # Skip images that can't be processed (malformed, exotic formats, etc.)
        return None


def _extract_images_for_page(
    doc: fitz.Document,
    page: fitz.Page,
    page_number: int,
    *,
    start_image_id: int,
    min_width: int = 0,
    min_height: int = 0,
) -> Tuple[List[Dict[str, Any]], int]:
    """
    Extract image metadata for a single PyMuPDF page.
    
    Args:
        min_width: Minimum pixel width to include (filters out tiny images)
        min_height: Minimum pixel height to include
    
    Returns list of dicts directly for faster DataFrame construction.
    """
    images: List[Dict[str, Any]] = []
    seen_xrefs = set()
    
    # Get all images on page with full info
    # full=True includes referencer xref
    img_list = page.get_images(full=True)
    
    if not img_list:
        return images, start_image_id
    
    next_image_id = start_image_id
    
    for img_info in img_list:
        xref = img_info[0]
        
        # Skip duplicates (same image displayed multiple times on page)
        if xref in seen_xrefs:
            continue
        
        seen_xrefs.add(xref)
        
        # Filter by minimum size
        img_width = img_info[2]
        img_height = img_info[3]

        if img_width < min_width or img_height < min_height:
            continue

        img_metadata = _extract_image_metadata(
            doc,
            page,
            img_info,
            page_number=page_number,
            image_id=next_image_id + 1,
        )

        if img_metadata:
            next_image_id += 1
            images.append(img_metadata)
    
    return images, next_image_id


def extract_images(
    pdf_path: str | Path,
    pages_to_process: Optional[List[int]] = None,
    min_width: int = 0,
    min_height: int = 0,
    min_dpi: Optional[float] = None,
) -> pd.DataFrame:
    """
    Extract all images from a PDF and return a DataFrame with one row per image.

    Args:
        pdf_path: Path to PDF file
        pages_to_process: Page numbers (1-indexed), or None for all pages
        min_width: Minimum image width in pixels to include
        min_height: Minimum image height in pixels to include
        min_dpi: Minimum DPI to include (applied after extraction)
    """
    pdf_path = Path(pdf_path).expanduser().resolve()
    
    all_images: List[Dict[str, Any]] = []
    next_image_id = 0
    
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
            
            page_images, next_image_id = _extract_images_for_page(
                doc,
                page,
                page_number=page_number,
                start_image_id=next_image_id,
                min_width=min_width,
                min_height=min_height,
            )
            
            all_images.extend(page_images)
    
    if not all_images:
        return pd.DataFrame()
    
    # Create DataFrame
    df = pd.DataFrame(all_images)
    
    # Apply DPI filter if requested
    if min_dpi is not None:
        df = df[
            (df["dpi_x"] >= min_dpi) | (df["dpi_y"] >= min_dpi)
        ].reset_index(drop=True)
    
    return df


# Example usage
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python step_02_image_extractor.py <pdf_path>")
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    
    # Extract all images with minimum size filter
    df = extract_images(
        pdf_path,
        min_width=10,  # Skip tiny images
        min_height=10,
        # min_dpi=72,  # Optional: only include images with decent resolution
    )
    
    if df.empty:
        print("No images found")
    else:
        print(f"Found {len(df)} images")
        print("\nFirst few images:")
        print(df[["page_number", "image_width", "image_height", 
                  "ext", "dpi_x", "dpi_y", "colorspace_name"]].head())
        
        # Optionally save to CSV
        # df.to_csv("images.csv", index=False)