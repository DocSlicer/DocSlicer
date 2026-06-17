# Migration: PyMuPDF → pypdfium2

Replace `pymupdf` (AGPL) with `pypdfium2` (BSD 3-Clause) across the four PDF extraction steps.

## Dependency change

```toml
# pyproject.toml — replace:
"pymupdf>=1.25"
# with:
"pypdfium2>=4.30.0,!=4.30.1,<6.0.0"
```

---

## pypdfium2 orientation

```python
import pypdfium2 as pdfium

doc = pdfium.PdfDocument(path)     # replaces fitz.open()
len(doc)                           # replaces doc.page_count
page = doc[i]                      # 0-indexed; replaces doc[i] / doc.load_page(i)
page.get_width() / page.get_height()  # replaces page.rect.width / page.rect.height
doc.close()                        # or: with pdfium.PdfDocument(path) as doc:
```

Coordinates come out in PDF space (origin bottom-left, y increases upward).  
PyMuPDF already uses top-left origin (y increases downward), so **no coordinate flip is needed** — both libraries return the same PDF user-space values for normal pages.

---

## Step 01 — `step_01_word_extractor.py` · Difficulty: HIGH

This is the hardest step. PyMuPDF's `get_text("words")` and `get_text("rawdict")` have no direct pypdfium2 equivalents. pypdfium2 exposes only **character-level** access.

### Strategy: try pdftext first

[`pdftext`](https://github.com/VikParuchuri/pdftext) (Apache 2.0) is built on pypdfium2 and already solves the hardest parts: char→word grouping, font name/size/weight, and character bboxes. **Try this path before building from scratch.**

**Add dependency:**
```toml
"pdftext>=0.3"
```

**Test approach:**

1. Run `pdftext.dictionary_output(pdf_path)` on several representative PDFs.
2. Inspect span granularity: if each span's `text` is a single whitespace-delimited token → near drop-in. If spans contain multiple words → split on whitespace and inherit the span's style for each child word.
3. Check that `bbox`, `font`, `fontsize`, `fontweight` are present at the span level.
4. Add text color on top via raw pypdfium2 calls (pdftext does not expose fill/stroke color):
   ```python
   import ctypes, pypdfium2.raw as pdfium_c
   textpage = page.get_textpage()   # already opened by pdftext internally; may need re-open
   r, g, b, a = ctypes.c_uint(), ctypes.c_uint(), ctypes.c_uint(), ctypes.c_uint()
   pdfium_c.FPDFText_GetFillColor(textpage, char_index, r, g, b, a)
   pdfium_c.FPDFText_GetStrokeColor(textpage, char_index, r, g, b, a)
   ```
5. Validate against PyMuPDF output on test PDFs: word count, bboxes, font names, colors.

**Accept criteria to continue with pdftext:** word bbox error < 0.5 pt, font name match > 95%, color extracted correctly. If any criterion fails → fall back to the from-scratch approach below.

---

### Fallback: build from scratch with raw pypdfium2

Use this path only if pdftext's output is insufficient (missing fields, wrong granularity, or unacceptable accuracy on your test PDFs).

### What needs to change

| PyMuPDF call | pypdfium2 replacement |
|---|---|
| `page.get_text("rawdict")` | Build from `textpage` char loop (see below) |
| `page.get_text("words")` | Build by grouping chars on spaces / line breaks |
| `span.get("font")` | `pdfium.raw.FPDFText_GetFontInfo(textpage, i, buf, buflen, flags)` |
| `span.get("size")` | `pdfium.raw.FPDFText_GetFontSize(textpage, i)` |
| `span.get("color")` | `pdfium.raw.FPDFText_GetFillColor(textpage, i, &r, &g, &b, &a)` |
| `span.get("stroke_color")` | `pdfium.raw.FPDFText_GetStrokeColor(textpage, i, &r, &g, &b, &a)` |
| `char["bbox"]` | `textpage.get_charbox(i)` → `(left, bottom, right, top)` |
| `span.get("dir")` | Derive from consecutive char center positions (same logic as existing fallback) |

### Key implementation notes

**Open a textpage:**
```python
textpage = page.get_textpage()
n = textpage.count_chars()
```

**Per-character loop (replaces rawdict):**
```python
import ctypes, pypdfium2 as pdfium

for i in range(n):
    l, b, r, t = textpage.get_charbox(i)   # PDF coords (bottom-left origin)
    # pypdfium2 char boxes: left, bottom, right, top
    # map to existing naming: x0=l, y0=t_flipped, x1=r, y1=b_flipped
    # BUT: since PyMuPDF also works in PDF space and the rest of docslicer
    # normalizes coords later, keep as-is and verify with test PDFs.

    char_text = textpage.get_text_range(i, 1)

    # Font name (raw ctypes call)
    buf = ctypes.create_string_buffer(256)
    flags = ctypes.c_int(0)
    pdfium.raw.FPDFText_GetFontInfo(textpage, i, buf, 256, ctypes.byref(flags))
    fontname = buf.value.decode("utf-8", errors="replace")

    fontsize = pdfium.raw.FPDFText_GetFontSize(textpage, i)

    # Fill color → replaces span["color"]  (non_stroking)
    r, g, b, a = ctypes.c_uint(), ctypes.c_uint(), ctypes.c_uint(), ctypes.c_uint()
    pdfium.raw.FPDFText_GetFillColor(textpage, i, r, g, b, a)
    fill_color = (r.value, g.value, b.value)   # 0–255 each → pass through pdf_color_to_hex

    # Stroke color → replaces span["stroke_color"]  (stroking)
    pdfium.raw.FPDFText_GetStrokeColor(textpage, i, r, g, b, a)
    stroke_color = (r.value, g.value, b.value)
```

**Word segmentation (replaces `get_text("words")`):**
The cleanest approach is to replicate PDFium's own word-break logic:
- A new word starts when `char_text in (' ', '\t', '\r', '\n')` or when the char box has a large horizontal gap from the previous char.
- Accumulate chars into a buffer; flush on whitespace or direction change.
- Word bbox = union of all constituent char bboxes.

Consider using `textpage.get_text_bounded()` split on whitespace as a simpler fallback if per-word bboxes are not needed at high precision — but the existing style-attachment logic depends on accurate word bboxes, so char-level grouping is recommended.

**Color conversion:** pypdfium2 fill/stroke colors come as `(R, G, B)` in 0–255 integer range. The existing `pdf_color_to_hex()` in `_utils/color_utils.py` likely accepts tuples — verify and adapt if needed (PyMuPDF encodes RGB as a packed integer).

**`_build_span_df` → `_build_char_df`:**
The existing span DF contains one row per run of same-style characters. In pypdfium2 you'll build a char-level DF and optionally group consecutive chars with identical (fontname, fontsize, fill_color, stroke_color) into synthetic spans. The downstream `_attach_representative_spans` works on any DF with geometry columns, so this grouping step is optional but will improve performance on long documents.

**Direction vectors:**
The existing "derive dx/dy from first→last char center" logic in `_build_span_df` translates directly — you already have char bboxes and that logic is pure Python/numpy.

---

## Step 02 — `step_02_image_extractor.py` · Difficulty: HIGH

PyMuPDF's `page.get_images()` + `doc.extract_image()` are very high-level. pypdfium2 requires iterating page objects and using raw PDFium calls.

### What needs to change

| PyMuPDF call | pypdfium2 replacement |
|---|---|
| `page.get_images(full=True)` | Iterate `page.get_objects()`, filter `PdfObjectType.IMAGE` |
| `img_info[0]` (xref) | `obj.get_identifier()` or use object index |
| `img_info[1]` (smask) | `pdfium.raw.FPDFImageObj_GetImageMetadata()` → `has_alpha` |
| `img_info[2..3]` (width, height) | `meta.width`, `meta.height` from metadata struct |
| `img_info[4]` (bpc) | `meta.bits_per_pixel / channels` |
| `img_info[5]` (colorspace int) | `meta.colorspace` |
| `img_info[8]` (filter string) | Not directly exposed; omit or set to `None` |
| `page.get_image_rects(xref)` | `obj.get_pos()` → matrix → derive bbox |
| `page.get_image_bbox(xref)` | Same |
| `doc.extract_image(xref)["ext"]` | Infer from `meta.colorspace` / render to bitmap |
| `doc.extract_image(xref)["cs-name"]` | Map from `meta.colorspace` int |

### Key implementation notes

```python
import pypdfium2 as pdfium, pypdfium2.raw as pdfium_raw, ctypes

for page_obj in page.get_objects():
    if page_obj.type != pdfium.PdfObjectType.IMAGE:
        continue

    # Intrinsic metadata
    meta = pdfium_raw.FPDF_IMAGEOBJ_METADATA()
    pdfium_raw.FPDFImageObj_GetImageMetadata(page_obj, page, ctypes.byref(meta))
    image_width  = meta.width
    image_height = meta.height
    bpc          = meta.bits_per_pixel  # total; divide by channels for per-component
    colorspace   = meta.colorspace      # FPDF_COLORSPACE_* enum int
    has_alpha    = meta.marked_content_id  # use meta.has_alpha if available

    # Display bbox from object matrix
    matrix = pdfium_raw.FS_MATRIX()
    pos    = pdfium_raw.FS_RECTF()
    pdfium_raw.FPDFPageObj_GetBounds(page_obj, ctypes.byref(pos))
    x_left, y_bottom, x_right, y_top = pos.left, pos.bottom, pos.right, pos.top
    # Note: PDF coords are bottom-left origin; flip y if needed to match prior output
```

**Colorspace mapping:** PDFium uses `FPDF_COLORSPACE_DEVICEGRAY=1`, `DEVICERGB=2`, `DEVICECMYK=3` — different integers than PyMuPDF's (1/3/4). Update `_colorspace_to_name()`.

**`ext` / `filter`:** PDFium doesn't expose the raw compression filter name conveniently. Either drop these columns or infer `ext` by rendering to a bitmap and re-encoding. If downstream code only uses `ext` for `has_transparency` detection, replace with the `meta` alpha flag instead.

**`xref`:** PDFium has no concept of xref numbers in Python. Replace `xref` with a page-local object index, or drop it if nothing downstream requires it.

---

## Step 03 — `step_03_shape_extractor.py` · Difficulty: HIGH

`page.get_drawings()` is a PyMuPDF convenience API with no pypdfium2 equivalent. Shape extraction requires iterating path objects via the raw PDFium C API.

### What needs to change

| PyMuPDF call | pypdfium2 replacement |
|---|---|
| `page.get_drawings()` | Iterate `page.get_objects()`, filter `PdfObjectType.PATH` |
| `drawing["type"]` ('f','s','fs') | `pdfium_raw.FPDFPath_GetDrawMode(obj, &fillmode, &stroke)` |
| `drawing["rect"]` | `pdfium_raw.FPDFPageObj_GetBounds(obj, &rect)` |
| `drawing["fill"]` (fill color) | `pdfium_raw.FPDFPageObj_GetFillColor(obj, &r,&g,&b,&a)` |
| `drawing["color"]` (stroke color) | `pdfium_raw.FPDFPageObj_GetStrokeColor(obj, &r,&g,&b,&a)` |
| `drawing["width"]` (linewidth) | `pdfium_raw.FPDFPageObj_GetStrokeWidth(obj, &width)` |
| `drawing["items"]` (path segments) | `pdfium_raw.FPDFPath_CountSegments(obj)` + `FPDFPath_GetPathSegment(obj, i)` |
| Segment cmd `"re"` / `"m"` / `"l"` / `"c"` | `pdfium_raw.FPDFPathSegment_GetType(seg)` → `FPDF_SEGMENT_*` enum |

### Key implementation notes

```python
import ctypes, pypdfium2 as pdfium, pypdfium2.raw as pdfium_raw

for page_obj in page.get_objects():
    if page_obj.type != pdfium.PdfObjectType.PATH:
        continue

    # Bounding box
    rect = pdfium_raw.FS_RECTF()
    pdfium_raw.FPDFPageObj_GetBounds(page_obj, ctypes.byref(rect))

    # Fill/stroke mode
    fillmode = ctypes.c_int(0)
    stroke   = ctypes.c_int(0)
    pdfium_raw.FPDFPath_GetDrawMode(page_obj, ctypes.byref(fillmode), ctypes.byref(stroke))
    # fillmode: 0=none, 1=alternate, 2=winding  → any non-zero means fill
    is_fill   = fillmode.value != 0
    is_stroke = stroke.value != 0

    # Colors (0–255)
    r, g, b, a = ctypes.c_uint(), ctypes.c_uint(), ctypes.c_uint(), ctypes.c_uint()
    pdfium_raw.FPDFPageObj_GetFillColor(page_obj, r, g, b, a)
    fill_rgb = (r.value, g.value, b.value)

    pdfium_raw.FPDFPageObj_GetStrokeColor(page_obj, r, g, b, a)
    stroke_rgb = (r.value, g.value, b.value)

    # Linewidth
    w = ctypes.c_float(0)
    pdfium_raw.FPDFPageObj_GetStrokeWidth(page_obj, ctypes.byref(w))
    linewidth = w.value
```

**Shape classification (`_classify_drawing_type`):**  
Rebuild using `FPDFPath_CountSegments` + `FPDFPathSegment_GetType`. Map segment types:
- `FPDF_SEGMENT_MOVETO` → `"m"`
- `FPDF_SEGMENT_LINETO` → `"l"`
- `FPDF_SEGMENT_BEZIERTO` → `"c"`
- `FPDF_SEGMENT_UNKNOWN` → start of a sub-path (often closes a rect)

A path with 4 line segments forming a closed rectangle → classify as `"rect"`.  
A path with 1 moveto + 1 lineto → `"line"`.  
Everything else → `"curve"`.

**`paint_op`:** PyMuPDF exposes `'f'`/`'s'`/`'fs'` as strings. Reconstruct from `is_fill` + `is_stroke`:
```python
paint_op = ("fs" if is_fill and is_stroke else "f" if is_fill else "s" if is_stroke else "")
```

**Color conversion:** pypdfium2 raw colors are 0–255 per channel tuples; verify `pdf_color_to_hex()` handles this format (PyMuPDF packs RGB as a single integer).

---

## Step 04 — `step_04_link_extractor.py` · Difficulty: MEDIUM

pypdfium2 exposes links through page annotations. The logic maps fairly cleanly.

### What needs to change

| PyMuPDF call | pypdfium2 replacement |
|---|---|
| `page.get_links()` | `page.get_links()` — pypdfium2 also has this! (returns `PdfLinkAnnot` objects) |
| `link["from"]` (Rect) | `link.get_pos()` → `FS_RECTF` struct or 4-tuple |
| `link["uri"]` | `link.get_url()` → `str` or `None` |
| `link["page"]` | `dest = link.get_dest(); dest.get_page_index(doc)` |
| `link["to"]` | `dest.get_view()` → view mode + coordinates |
| `link["file"]` | Not directly exposed; omit |
| `rect_obj.x0` etc. | Adapt `_rect_to_bbox()` for pypdfium2 rect format |

### Key implementation notes

```python
import pypdfium2 as pdfium

# pypdfium2 PdfPage has get_links() returning an iterator of PdfLinkAnnot
for link in page.get_links():
    rect = link.get_pos()           # returns (left, bottom, right, top) or FS_RECTF
    url  = link.get_url()           # str if external, None if internal

    dest = link.get_dest()
    if dest:
        target_page = dest.get_page_index(doc)   # 0-based
        view        = dest.get_view()             # (view_mode, params_tuple)
    else:
        target_page = None
        view        = None
```

**`_rect_to_bbox()`:** Update to accept pypdfium2's rect format (may be a named tuple or ctypes struct rather than a Rect object with `.x0`/`.y0` attributes).

**`_serialize_dest_from_link()`:** Adapt to build the dest dict from `dest.get_page_index()` + `dest.get_view()` instead of `link["page"]` + `link["to"]`.

**Note:** If `page.get_links()` is not available in the installed pypdfium2 version, fall back to iterating `page.get_annotations()` and filtering for `PdfAnnotType.LINK`.

---

## Color conversion — `_utils/color_utils.py`

PyMuPDF encodes RGB as a **packed 24-bit integer** (e.g. `0xFF0000` = red).  
pypdfium2 raw calls return **3 separate `c_uint` values** in 0–255 range.

Update `pdf_color_to_hex()` to accept both forms, or add a second function:

```python
def rgb_tuple_to_hex(r: int, g: int, b: int) -> str:
    return f"#{r:02x}{g:02x}{b:02x}"
```

---

## Testing strategy

1. Pick 2–3 representative PDFs (simple, complex layout, scanned+OCR).
2. Run both extractors and diff the DataFrames column by column.
3. Priority columns to validate: `text`, `x_left/y_top/x_right/y_bottom`, `font_name`, `font_size`, `non_stroking_color`.
4. Accept small floating-point differences (< 0.5 pt) — PDFium and MuPDF parse glyph metrics slightly differently.
5. Shapes and images: validate row counts and bbox overlap, not exact pixel values.

---

## Effort estimate

| Step | Effort (optimistic) | Effort (fallback) | Blocking on |
|---|---|---|---|
| Step 01 | ~0.5–1 day (pdftext path) | ~2–3 days (from scratch) | pdftext span granularity + color gap |
| Step 02 | ~1–2 days | — | raw image metadata struct, bbox extraction |
| Step 03 | ~1–2 days | — | path segment API, shape classification rebuild |
| Step 04 | ~0.5 day | — | link dest serialization |
| color_utils | ~1 hour | — | tuple vs int input format |
| **Total** | **~4–5 days** | **~7–9 days** | |
