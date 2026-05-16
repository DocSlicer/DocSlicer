"""Extract document metadata from DOCX core/app properties XML parts."""

from __future__ import annotations

from typing import Any

from .step_01_package_reader import DocxPackage

_DC = "http://purl.org/dc/elements/1.1/"
_APP = "http://schemas.openxmlformats.org/officeDocument/2006/extended-properties"


def _text(root, tag_ns: str, tag_local: str) -> str | None:
    elem = root.find(f"{{{tag_ns}}}{tag_local}")
    if elem is None:
        return None
    val = (elem.text or "").strip()
    return val or None


def extract_core_properties(package: DocxPackage) -> dict[str, Any]:
    """
    Read docProps/core.xml and docProps/app.xml and return a metadata dict
    with the same keys that add_document_information populates:

        title_meta, author_meta, language, page_count
    """
    title_meta: str | None = None
    author_meta: list[str] | None = None
    language: str | None = None
    page_count: int = 0

    core = package.get_xml("docProps/core.xml")
    if core is not None:
        title_raw = _text(core, _DC, "title")
        if title_raw:
            title_meta = title_raw

        creator = _text(core, _DC, "creator")
        if creator:
            author_meta = [creator]

        lang = _text(core, _DC, "language")
        if lang:
            language = lang

    app = package.get_xml("docProps/app.xml")
    if app is not None:
        pages_raw = _text(app, _APP, "Pages")
        try:
            page_count = int(pages_raw) if pages_raw else 0
        except ValueError:
            page_count = 0

    return {
        "title_meta": title_meta,
        "author_meta": author_meta,
        "language": language,
        "page_count": page_count,
    }
