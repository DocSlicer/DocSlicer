# SPDX-FileCopyrightText: 2026 Market Framer Inc.
# SPDX-License-Identifier: AGPL-3.0-only
"""Normalize authoring/generator strings into a shared vendor label.

Used by every native-metadata extractor so PDF and OOXML emit the *same*
``generator`` taxonomy. Input strings come from different places per format:

    PDF     creator_tool (xmp:CreatorTool / /Creator) + producer (pdf:Producer)
    OOXML   application (docProps/app.xml <Application>)

The office extractors additionally detect ``google-docs`` / ``apple`` from the
package *shape* (absent vs empty docProps) — those exporters write no generator
string at all — and only fall back to this matcher when a string is present.
"""

from __future__ import annotations

import re

# Checked in order; first hit wins. Authoring-app hints precede PDF-engine hints
# so "Acrobat PDFMaker 25 for Word" classifies as the source app (microsoft),
# not the plugin (adobe).
_RULES: list[tuple[str, tuple[str, ...]]] = [
    ("microsoft", ("microsoft", "powerpoint", "for word", "for microsoft", "pscript")),
    ("adobe", ("indesign", "framemaker", "distiller", "acrobat", "adobe")),
    ("apple", ("quartz", "pdfcontext", "preview", "macintosh", "pages", "keynote", "iwork")),
    ("libreoffice", ("libreoffice", "openoffice")),
    ("google-docs", ("google", "skia/pdf")),
    ("workiva", ("workiva", "wdesk")),
    ("ghostscript", ("ghostscript",)),
    ("aspose", ("aspose",)),
    ("antenna-house", ("antenna house", "xsl formatter")),
    ("pdflib", ("pdflib",)),
    ("reportlab", ("reportlab",)),
    ("arbortext", ("arbortext",)),
]


def _classify_one(text: str) -> str | None:
    s = text.lower()
    for label, needles in _RULES:
        if any(n in s for n in needles):
            return label
    # TeX family: word-boundary match so "PDFContext" doesn't trip on "tex".
    if any(n in s for n in ("pdftex", "dvips", "latex")) or re.search(r"\btex\b", s):
        return "latex"
    return None


def classify_generator(
    application: str | None = None,
    creator_tool: str | None = None,
    producer: str | None = None,
) -> str | None:
    """Return a normalized vendor label, or None if nothing matches.

    Sources are tried in decreasing authority: the OOXML ``application`` (an
    authoring app by definition), then the PDF ``creator_tool`` (source app),
    then ``producer`` (the PDF engine — least specific about the origin app).
    """
    for source in (application, creator_tool, producer):
        if source:
            label = _classify_one(source)
            if label:
                return label
    return None


def office_generator(
    application: str | None,
    *,
    has_app_part: bool,
    has_core_part: bool,
) -> str | None:
    """Classify an OOXML document's generator, including string-less exporters.

    MS Office / LibreOffice write a matchable ``<Application>``. Apple Pages and
    Google Docs write none — they are told apart by package shape: Google Docs
    ships no ``docProps`` at all, Pages ships an empty ``docProps/app.xml``.
    """
    label = classify_generator(application=application)
    if label:
        return label
    if not has_app_part and not has_core_part:
        return "google-docs"
    if has_app_part:            # app.xml present but no <Application> string
        return "apple"
    return None
