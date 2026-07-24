# SPDX-FileCopyrightText: 2026 Market Framer Inc.
# SPDX-License-Identifier: AGPL-3.0-only
"""Extract document metadata from PPTX core/app properties XML parts."""

from __future__ import annotations

from typing import Any

from .step_01_package_reader import PptxPackage
from ..metadata.generator import office_generator

_DC = "http://purl.org/dc/elements/1.1/"
_DCTERMS = "http://purl.org/dc/terms/"
_CP = "http://schemas.openxmlformats.org/package/2006/metadata/core-properties"
_APP = "http://schemas.openxmlformats.org/officeDocument/2006/extended-properties"


def _text(root, tag_ns: str, tag_local: str) -> str | None:
    elem = root.find(f"{{{tag_ns}}}{tag_local}")
    if elem is None:
        return None
    val = (elem.text or "").strip()
    return val or None


def extract_native_metadata(package: PptxPackage) -> dict[str, Any]:
    """Read docProps/core.xml and docProps/app.xml into the shared metadata shape."""
    title_meta: str | None = None
    author_meta: list[str] | None = None
    language: str | None = None
    page_count: int = 0
    created: str | None = None
    modified: str | None = None
    last_modified_by: str | None = None
    application: str | None = None
    app_version: str | None = None

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

        # Timestamps are W3CDTF (ISO-8601) already, e.g. "2019-07-08T16:11:33Z".
        created = _text(core, _DCTERMS, "created")
        modified = _text(core, _DCTERMS, "modified")
        last_modified_by = _text(core, _CP, "lastModifiedBy")

    app = package.get_xml("docProps/app.xml")
    if app is not None:
        slides_raw = _text(app, _APP, "Slides")
        try:
            page_count = int(slides_raw) if slides_raw else 0
        except ValueError:
            page_count = 0

        application = _text(app, _APP, "Application")
        app_version = _text(app, _APP, "AppVersion")

    # Pages/Google Slides write no <Application>; classify by package shape.
    generator = office_generator(
        application, has_app_part=app is not None, has_core_part=core is not None
    )

    return {
        "title_meta": title_meta,
        "author_meta": author_meta,
        "language_meta": language,
        "page_count": page_count,
        "created": created,
        "modified": modified,
        "last_modified_by": last_modified_by,
        "application": application,
        "app_version": app_version,
        "generator": generator,
    }
