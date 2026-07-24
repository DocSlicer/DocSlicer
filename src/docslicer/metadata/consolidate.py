# SPDX-FileCopyrightText: 2026 Market Framer Inc.
# SPDX-License-Identifier: AGPL-3.0-only
"""Consolidate the native and text metadata channels into the final fields.

By the time this runs, a channel's ``discovered_metadata`` already carries both:

    native (from */native_metadata.py)   title_meta, author_meta, language_meta, …
    text   (from text_fallback.py)        title_text, author_text, language_text

The rule is deliberately simple — **native wins when present, text is the
fallback** — with one guard: a native ``author`` is often a software artifact
("Microsoft Office User", "Adobe"), so the picked author is filtered through
``_is_fake_author`` and we fall through to the text channel if nothing survives.

``consolidate`` mutates the dict in place, setting ``title`` / ``author`` /
``language``. It leaves the per-channel ``*_meta`` / ``*_text`` keys untouched.
"""

from __future__ import annotations

import re
from typing import Any, Dict

from .text_fallback import _normalize_language_code


# ─────────────────────────────────────────────
# Fake-author gate (a picking concern, not a text-extraction one)
# ─────────────────────────────────────────────

_FAKE_AUTHORS = {
    # Placeholders (all lowercase for case-insensitive matching)
    "", "unknown", "n/a", "na", "none", "null", "anonymous",
    "author", "user", "admin", "content", "version",
    "system", "default", "root", "test", "demo", "sample",
    "posted by", "written by", "contributor", "guest",
    "rss", "contact", "info", "webmaster", "reporter",
    # PDF / software artifacts
    "microsoft", "office", "word", "powerpoint", "excel",
    "adobe", "acrobat", "indesign", "quartz", "pdfcontext",
    "preview", "scan", "hp", "canon", "epson", "pdf",
    "abbyy", "finereader", "scanner",
    # Filing / financial system generators
    "workiva", "platform", "xbrl", "edgar",
    # Web / CMS junk
    "wordpress", "drupal", "joomla", "latex", "package",
    "sitecore", "contentful", "squarespace",
}


def _is_fake_author(author: str) -> bool:
    """True if the name is likely a placeholder / software artifact, not a person."""
    if not author:
        return True

    author_lower = author.lower().strip()

    if author_lower in _FAKE_AUTHORS:
        return True

    for fake in _FAKE_AUTHORS:
        if fake and fake in author_lower:
            return True

    if len(author_lower) < 3:  # Too short
        return True

    if author_lower.startswith(("v", "ver", "version", "rev")):  # Version strings
        return True

    if any(c.isdigit() for c in author_lower):  # Contains numbers
        if re.match(r'^[a-z\s]+\d+$', author_lower):      # "John Smith 2" — could be valid
            pass
        elif re.search(r'\d+\.\d+', author_lower):        # "v1.2.3" — version pattern
            return True

    return False


# ─────────────────────────────────────────────
# Field resolvers — native first, text as fallback
# ─────────────────────────────────────────────

def _resolve_title(meta: Dict[str, Any]) -> str | None:
    return meta.get("title_meta") or meta.get("title_text") or None


def _resolve_author(meta: Dict[str, Any]) -> list[str] | None:
    """Native author list if any survive the fake-author gate, else the text list."""
    for key in ("author_meta", "author_text"):
        candidates = meta.get(key) or []
        valid = [a for a in candidates if a and not _is_fake_author(a)]
        if valid:
            return valid[:5]
    return None


def _resolve_language(meta: Dict[str, Any]) -> str:
    """Native normalized language code wins over text; "unknown" if neither."""
    lang_meta = _normalize_language_code(meta.get("language_meta") or "")
    lang_text = _normalize_language_code(meta.get("language_text") or "")
    return lang_meta or lang_text or "unknown"


# ─────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────

def consolidate(meta: Dict[str, Any]) -> None:
    """Fold the native + text channels into final title/author/language fields.

    Args:
        meta: Metadata dict carrying both channels, modified in place.
    """
    meta["title"] = _resolve_title(meta)
    meta["author"] = _resolve_author(meta)
    meta["language"] = _resolve_language(meta)
