"""
Step 02 – Image extraction

Output columns:
    page_number, image_id, obj_index,
    x_left, y_top, x_right, y_bottom, width, height, area,
    image_width, image_height, bpc, colorspace, colorspace_name,
    ext, filter, smask, has_transparency,
    dpi_x, dpi_y

Coordinate system: FPDFPageObj_GetBounds returns (left, bottom, right, top) in
PDF space (y increases upward). We convert to screen space (y increases downward):
y_top = page_height - pdf_top.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pypdfium2 as pdfium
import pypdfium2.raw as pdfium_c
from pypdfium2._helpers.pageobjects import PdfImage
import pandas as pd


# FPDF_COLORSPACE_* integer → human name
_COLORSPACE_NAMES: Dict[int, str] = {
    0: "Unknown",
    1: "DeviceGray",
    2: "DeviceRGB",
    3: "DeviceCMYK",
    4: "CalGray",
    5: "CalRGB",
    6: "Lab",
    7: "ICCBased",
    8: "Separation",
    9: "DeviceN",
    10: "Indexed",
    11: "Pattern",
}

# FPDF_COLORSPACE_* → number of colour channels (for bpc computation)
_COLORSPACE_CHANNELS: Dict[int, int] = {
    1: 1,  # DeviceGray
    2: 3,  # DeviceRGB
    3: 4,  # DeviceCMYK
    4: 1,  # CalGray
    5: 3,  # CalRGB
    6: 3,  # Lab
}

# PDF compression filter → file extension
_FILTER_TO_EXT: Dict[str, str] = {
    "DCTDecode": "jpeg",
    "JPXDecode": "jp2",
    "CCITTFaxDecode": "tiff",
    "JBIG2Decode": "jbig2",
    "FlateDecode": "png",
    "LZWDecode": "tiff",
    "RunLengthDecode": "bmp",
}


def _colorspace_to_name(cs_int: int) -> str:
    return _COLORSPACE_NAMES.get(cs_int, f"Colorspace({cs_int})")


def _bpc_from_metadata(bits_per_pixel: int, colorspace: int) -> int:
    channels = _COLORSPACE_CHANNELS.get(colorspace, 0)
    if channels > 0 and bits_per_pixel > 0:
        return bits_per_pixel // channels
    return bits_per_pixel


def _ext_from_filters(filters: List[str]) -> Optional[str]:
    for f in filters:
        ext = _FILTER_TO_EXT.get(f)
        if ext:
            return ext
    return None


def _extract_image_metadata(
    page: pdfium.PdfPage,
    obj: Any,
    obj_index: int,
    *,
    page_number: int,
    image_id: int,
    page_height: float,
    min_width: int,
    min_height: int,
) -> Optional[Dict[str, Any]]:
    try:
        # Wrap as PdfImage to access image-specific methods.
        # needs_free=False (page is set) so the raw handle is not destroyed on GC.
        img = PdfImage(obj.raw, page=page, pdf=page.pdf)

        # Intrinsic metadata
        meta = img.get_metadata()
        image_width = meta.width
        image_height = meta.height

        if image_width < min_width or image_height < min_height:
            return None

        colorspace = meta.colorspace
        bpc = _bpc_from_metadata(meta.bits_per_pixel, colorspace)
        colorspace_name = _colorspace_to_name(colorspace)

        # DPI already calculated by pdfium from the object matrix
        dpi_x = round(float(meta.horizontal_dpi), 2)
        dpi_y = round(float(meta.vertical_dpi), 2)

        # Compression filter(s) and inferred extension
        filters = img.get_filters()
        img_filter = filters[0] if filters else None
        ext = _ext_from_filters(filters)

        # Display bbox: PDF coords (y upward) → screen coords (y downward)
        l, b, r, t = img.get_bounds()
        x_left = float(l)
        x_right = float(r)
        y_top = page_height - float(t)
        y_bottom = page_height - float(b)

        display_width = x_right - x_left
        display_height = y_bottom - y_top
        area = display_width * display_height

        has_transparency = ext in ("png", "jp2")

        return {
            "page_number": page_number,
            "image_id": image_id,
            "obj_index": obj_index,

            "x_left": x_left,
            "x_right": x_right,
            "y_top": y_top,
            "y_bottom": y_bottom,
            "width": display_width,
            "height": display_height,
            "area": area,

            "image_width": image_width,
            "image_height": image_height,
            "bpc": bpc,
            "colorspace": colorspace,
            "colorspace_name": colorspace_name,

            "ext": ext,
            "filter": img_filter,

            "smask": 0,
            "has_transparency": has_transparency,

            "dpi_x": dpi_x,
            "dpi_y": dpi_y,
        }

    except Exception:
        return None


def _extract_images_for_page(
    page: pdfium.PdfPage,
    page_number: int,
    *,
    start_image_id: int,
    min_width: int,
    min_height: int,
) -> Tuple[List[Dict[str, Any]], int]:
    images: List[Dict[str, Any]] = []
    page_height = page.get_height()
    next_image_id = start_image_id

    for obj_index, obj in enumerate(
        page.get_objects(filter=[pdfium_c.FPDF_PAGEOBJ_IMAGE])
    ):
        img_metadata = _extract_image_metadata(
            page,
            obj,
            obj_index,
            page_number=page_number,
            image_id=next_image_id + 1,
            page_height=page_height,
            min_width=min_width,
            min_height=min_height,
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
        min_width: Minimum image pixel width to include
        min_height: Minimum image pixel height to include
        min_dpi: Minimum DPI to include (applied after extraction)
    """
    pdf_path = Path(pdf_path).expanduser().resolve()

    all_images: List[Dict[str, Any]] = []
    next_image_id = 0

    with pdfium.PdfDocument(pdf_path) as doc:
        total_pages = len(doc)

        page_numbers = (
            range(1, total_pages + 1) if pages_to_process is None else pages_to_process
        )

        for page_number in page_numbers:
            if page_number < 1 or page_number > total_pages:
                continue

            page = doc[page_number - 1]
            page_images, next_image_id = _extract_images_for_page(
                page,
                page_number,
                start_image_id=next_image_id,
                min_width=min_width,
                min_height=min_height,
            )
            all_images.extend(page_images)

    if not all_images:
        return pd.DataFrame()

    df = pd.DataFrame(all_images)

    if min_dpi is not None:
        df = df[
            (df["dpi_x"] >= min_dpi) | (df["dpi_y"] >= min_dpi)
        ].reset_index(drop=True)

    return df


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python step_02_image_extractor.py <pdf_path>")
        sys.exit(1)

    df = extract_images(
        sys.argv[1],
        min_width=10,
        min_height=10,
    )

    if df.empty:
        print("No images found")
    else:
        print(f"Found {len(df)} images")
        print(df[["page_number", "image_width", "image_height",
                  "ext", "dpi_x", "dpi_y", "colorspace_name"]].head())
