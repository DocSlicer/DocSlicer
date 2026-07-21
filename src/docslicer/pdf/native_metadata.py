"""Extract document metadata from a PDF's XMP packet, /Info dict and catalog.

Mirrors the docx/pptx native extractors: same output keys, read off an already
-open ``pikepdf.Pdf`` handle so the pipeline never reopens the file. Field
sourcing follows what the corpus actually carries:

    title_meta        dc:title  ->  /Info /Title
    author_meta       dc:creator (list)  ->  /Info /Author   (never /Creator — that is the tool)
    language_meta     catalog /Root /Lang   (absent from /Info and XMP)
    page_count        len(pdf.pages)
    created           xmp:CreateDate  ->  /Info /CreationDate   (normalized to ISO-8601)
    modified          xmp:ModifyDate  ->  /Info /ModDate        (normalized to ISO-8601)
    last_modified_by  None            (PDF has no "last editor" concept — OOXML only)
    creator_tool      xmp:CreatorTool  ->  /Info /Creator       (the source authoring app)
    producer          pdf:Producer  ->  /Info /Producer         (the PDF-generation engine)
    generator         normalized vendor label from creator_tool + producer
"""

from __future__ import annotations

import re
from typing import Any

import pikepdf

from ..metadata.generator import classify_generator


def _clean(value: Any) -> str | None:
    """Coerce a pikepdf String/Name/etc. to a stripped str, or None if empty."""
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _pdf_date_to_iso(value: Any) -> str | None:
    """Normalize a PDF date to ISO-8601.

    Accepts both the ``/Info`` ``D:YYYYMMDDHHmmSS+HH'mm'`` form and XMP's
    already-ISO ``YYYY-MM-DDTHH:MM:SS+HH:MM`` form, returning ISO-8601
    (e.g. ``2017-03-21T10:40:55-04:00`` / ``...Z``). Returns None if unparseable.
    """
    text = _clean(value)
    if text is None:
        return None

    # Already ISO-ish (has the date separators or a 'T') — pass through, only
    # tidying a trailing offset written as +HH'mm' just in case.
    if "T" in text or re.match(r"\d{4}-\d{2}", text):
        return text.replace("'", "")

    if text.startswith("D:"):
        text = text[2:]

    m = re.match(
        r"(\d{4})(\d{2})?(\d{2})?(\d{2})?(\d{2})?(\d{2})?"       # date + time
        r"(Z|[+-]\d{2}'?\d{2}'?)?",                              # timezone
        text,
    )
    if not m:
        return None
    year, month, day, hh, mm, ss, tz = m.groups()
    month = month or "01"
    day = day or "01"
    hh, mm, ss = hh or "00", mm or "00", ss or "00"

    iso = f"{year}-{month}-{day}T{hh}:{mm}:{ss}"
    if tz:
        if tz == "Z":
            iso += "Z"
        else:                                    # +HH'mm'  ->  +HH:MM
            digits = tz.replace("'", "")
            iso += f"{digits[:3]}:{digits[3:5]}"
    return iso


def _xmp_creators(meta) -> list[str]:
    """dc:creator from the XMP view, as a list (it is an ordered Seq)."""
    try:
        raw = meta.get("dc:creator")
    except Exception:
        return []
    if not raw:
        return []
    if isinstance(raw, (list, tuple)):
        return [c.strip() for c in raw if c and str(c).strip()]
    cleaned = _clean(raw)
    return [cleaned] if cleaned else []


def extract_native_metadata(pdf: pikepdf.Pdf) -> dict[str, Any]:
    """Read a PDF's embedded metadata into the shared native-metadata shape.

    Args:
        pdf: An already-open ``pikepdf.Pdf`` (the pipeline's shared handle).

    Returns:
        dict with keys: title_meta, author_meta, language_meta, page_count,
        created, modified, last_modified_by.
    """
    docinfo = pdf.docinfo if getattr(pdf, "docinfo", None) is not None else {}

    def info(key: str) -> Any:
        try:
            return docinfo.get(key)
        except Exception:
            return None

    # --- XMP view (merges docinfo); tolerate PDFs with no metadata stream ---
    # update_docinfo=False is critical: the default (True) syncs the XMP packet
    # back into /Info on context exit, so a PDF with an empty/absent XMP stream
    # (common — e.g. Workiva/Wdesk output) would have its /Info Title/Author/
    # Creator/Producer/dates wiped out, and every /Info fallback below returns None.
    title_xmp = created_xmp = modified_xmp = None
    creator_tool_xmp = producer_xmp = None
    xmp_creators: list[str] = []
    try:
        with pdf.open_metadata(set_pikepdf_as_editor=False, update_docinfo=False) as meta:
            title_xmp = _clean(meta.get("dc:title"))
            created_xmp = meta.get("xmp:CreateDate")
            modified_xmp = meta.get("xmp:ModifyDate")
            creator_tool_xmp = _clean(meta.get("xmp:CreatorTool"))
            producer_xmp = _clean(meta.get("pdf:Producer"))
            xmp_creators = _xmp_creators(meta)
    except Exception:
        pass

    # --- title: XMP dc:title, else /Info /Title ---
    title_meta = title_xmp or _clean(info("/Title"))

    # --- author: dc:creator (list), else /Info /Author. Never /Creator. ---
    author_meta = xmp_creators
    if not author_meta:
        info_author = _clean(info("/Author"))
        if info_author:
            author_meta = [info_author]
    author_meta = author_meta or None

    # --- language: catalog /Root /Lang only ---
    language = None
    try:
        language = _clean(pdf.Root.get("/Lang"))
    except Exception:
        language = None

    # --- dates: prefer XMP (ISO), fall back to /Info (D:...); normalize both ---
    created = _pdf_date_to_iso(created_xmp) or _pdf_date_to_iso(info("/CreationDate"))
    modified = _pdf_date_to_iso(modified_xmp) or _pdf_date_to_iso(info("/ModDate"))

    # --- generator: Creator = source app, Producer = PDF engine ---
    creator_tool = creator_tool_xmp or _clean(info("/Creator"))
    producer = producer_xmp or _clean(info("/Producer"))
    generator = classify_generator(creator_tool=creator_tool, producer=producer)

    # --- page_count ---
    try:
        page_count = len(pdf.pages)
    except Exception:
        page_count = 0

    return {
        "title_meta": title_meta,
        "author_meta": author_meta,
        "language_meta": language,
        "page_count": page_count,
        "created": created,
        "modified": modified,
        "last_modified_by": None,   # no PDF equivalent (OOXML-only field)
        "creator_tool": creator_tool,
        "producer": producer,
        "generator": generator,
    }


if __name__ == "__main__":   # quick manual check: python -m docslicer.pdf.native_metadata file.pdf
    import sys
    for path in sys.argv[1:]:
        with pikepdf.open(path) as _pdf:
            print(path, "->", extract_native_metadata(_pdf))
