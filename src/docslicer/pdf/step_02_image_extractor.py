"""
Step 02 – Image extraction

Output columns:
    page_number, image_id, obj_index,
    x_left, y_top, x_right, y_bottom, width, height, area,
    image_width, image_height, bpc, colorspace, colorspace_name,
    ext, filter, smask, has_transparency,
    dpi_x, dpi_y

Coordinate system: FPDFPageObj_GetBounds returns (left, bottom, right, top) in
raw, unrotated PDF space (y increases upward). We convert to screen space
(y increases downward, in the page's *displayed* orientation) via
_utils.page_rotation.make_rotation_transform, which also accounts for the
page's /Rotate entry — see that module's docstring for why this is needed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pypdfium2 as pdfium
import pypdfium2.raw as pdfium_c
from pypdfium2._helpers.pageobjects import PdfImage
import pandas as pd

from ._utils.page_rotation import make_rotation_transform
from ._utils.struct_tree import StructInfo, struct_info_to_columns


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


def _struct_for_object(
    obj_raw: Any,
    struct_index: Optional[Dict[Tuple[Optional[int], int], StructInfo]],
    page_index: int,
) -> Tuple[Optional[int], Optional[StructInfo]]:
    """Resolve a page object's ``(mcid, StructInfo)`` via its marked-content id.

    Returns ``(None, None)`` when no struct index is supplied, the object carries
    no MCID, or nothing matches. Elements with no resolvable /Pg are keyed under
    page ``None`` (same fallback the word join uses)."""
    if struct_index is None:
        return None, None
    mcid = pdfium_c.FPDFPageObj_GetMarkedContentID(obj_raw)
    if mcid < 0:
        return None, None
    info = struct_index.get((page_index, mcid)) or struct_index.get((None, mcid))
    return mcid, info


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
    to_screen,
    min_width: int,
    min_height: int,
    struct_index: Optional[Dict[Tuple[Optional[int], int], StructInfo]] = None,
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

        # Display bbox: raw PDF-space bounds → rotation-aware screen coords
        #
        # get_bounds() only applies the image's own matrix, which is relative to
        # its immediate parent (a Form XObject, if the image is nested in one) —
        # it does NOT compose in the ancestor Form XObjects' own placement
        # matrices. For images placed directly on the page (the common case)
        # this is already page space and no correction is needed. For images
        # nested inside one or more Form XObjects, we have to walk up the
        # container chain and fold each ancestor's matrix in ourselves, or the
        # bounds come out in the innermost form's local coordinate space.
        l, b, r, t = img.get_bounds()
        container = obj.container
        while container is not None:
            l, b, r, t = container.get_matrix().on_rect(l, b, r, t)
            container = container.container
        x_left, y_top, x_right, y_bottom = to_screen(float(l), float(b), float(r), float(t))

        display_width = x_right - x_left
        display_height = y_bottom - y_top
        area = display_width * display_height

        has_transparency = ext in ("png", "jp2")

        result: Dict[str, Any] = {
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

        # Struct-tree enrichment (Figure tag, ancestors, alt text) — only when a
        # struct index is supplied, so standalone extraction keeps its legacy schema.
        if struct_index is not None:
            mcid, info = _struct_for_object(obj.raw, struct_index, page_number - 1)
            result["mcid"] = mcid
            result.update(struct_info_to_columns(info))
            # img_alt: authored alternative text for the figure (/Alt), falling back
            # to /ActualText — the accessible description a tagged PDF gives an image.
            result["img_alt"] = (info.alt or info.actual_text) if info is not None else None

        return result

    except Exception:
        return None


def _extract_images_for_page(
    page: pdfium.PdfPage,
    page_number: int,
    *,
    start_image_id: int,
    min_width: int,
    min_height: int,
    struct_index: Optional[Dict[Tuple[Optional[int], int], StructInfo]] = None,
) -> Tuple[List[Dict[str, Any]], int]:
    images: List[Dict[str, Any]] = []
    page_width  = float(page.get_width())
    page_height = float(page.get_height())
    to_screen, _rotation = make_rotation_transform(page, page_width, page_height)
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
            to_screen=to_screen,
            min_width=min_width,
            min_height=min_height,
            struct_index=struct_index,
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
    struct_index: Optional[Dict[Tuple[Optional[int], int], StructInfo]] = None,
) -> pd.DataFrame:
    """
    Extract all images from a PDF and return a DataFrame with one row per image.

    Args:
        pdf_path: Path to PDF file
        pages_to_process: Page numbers (1-indexed), or None for all pages
        min_width: Minimum image pixel width to include
        min_height: Minimum image pixel height to include
        min_dpi: Minimum DPI to include (applied after extraction)
        struct_index: Optional ``{(page, mcid): StructInfo}`` from the shared
            :class:`StructContext`. When supplied, each image is joined to its
            struct-tree leaf by marked-content id, adding ``mcid``, the same
            ``struct_*`` columns words carry, and ``img_alt`` (the figure's /Alt or
            /ActualText). Omit it to keep the legacy image-only schema.
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

            try:
                page = doc[page_number - 1]
            except Exception:
                continue
            page_images, next_image_id = _extract_images_for_page(
                page,
                page_number,
                start_image_id=next_image_id,
                min_width=min_width,
                min_height=min_height,
                struct_index=struct_index,
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
