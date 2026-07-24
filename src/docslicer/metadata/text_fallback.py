# SPDX-FileCopyrightText: 2026 Market Framer Inc.
# SPDX-License-Identifier: AGPL-3.0-only
"""Text-based metadata fallbacks — derive author / title / language from the
document body when native metadata is missing.

Every pipeline runs its own ``native_metadata`` extractor first; this module is
the second pass, reading the already-parsed ``df_lines`` to recover fields that
the file's embedded metadata didn't carry (a very common case: SEC EDGAR HTML
ships an empty <head>, many scanned/exported PDFs carry no XMP).

It populates three keys only — it does **not** resolve native-vs-text or touch
the final ``title`` / ``author`` / ``language`` fields; that merge happens in the
consolidation step:

    author_text     list[str]   names found in title-case lines near the top
    title_text      str | None  largest-font line on the first pages
    language_text   str | None  langdetect over the first ~10k chars
"""

from __future__ import annotations

import re
from typing import Any, Dict, Optional

import pandas as pd


# ─────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────

def _is_title_case(text: str) -> bool:
    """True if text is title case (each significant word capitalized, not ALL CAPS)."""
    words = re.findall(r'\b[a-zA-Z]+\b', text)
    if not words:
        return False

    if all(w.isupper() for w in words):   # ALL CAPS — reject
        return False
    if all(w.islower() for w in words):   # all lowercase — reject
        return False

    small_words = {"a", "an", "the", "of", "in", "on", "at", "to", "for", "and", "or", "but"}

    capitalized_count = 0
    significant_word_count = 0
    for i, word in enumerate(words):
        if word.lower() in small_words and i > 0:   # small words may be lowercase (except first)
            continue
        significant_word_count += 1
        if word[0].isupper():
            capitalized_count += 1

    if significant_word_count == 0:
        return False

    return (capitalized_count / significant_word_count) >= 0.7


def _load_common_author_names() -> set[str]:
    """Load the lowercase common-name set from config/common_author_names.csv."""
    from importlib.resources import files

    try:
        with (files("docslicer") / "config" / "common_author_names.csv").open("rb") as f:
            df = pd.read_csv(f)
        if "name" in df.columns:
            return set(df["name"].dropna().astype(str).str.strip())
        return set()
    except Exception:
        return set()


def _normalize_language_code(lang: str) -> str:
    """Normalize a language code to its base ISO 639-1 form ("en-US" → "en")."""
    if not lang or not isinstance(lang, str):
        return ""

    lang = lang.strip().lower()
    for delimiter in ('-', '_'):
        if delimiter in lang:
            lang = lang.split(delimiter)[0]

    if len(lang) >= 2:
        return lang[:2]
    return lang


# ─────────────────────────────────────────────
# Field extractors
# ─────────────────────────────────────────────

def extract_author_from_text(
    df_lines: pd.DataFrame,
    common_names: Optional[set[str]] = None,
) -> list[str]:
    """Find author names by matching common first names in title-case lines near the top.

    Args:
        df_lines: DataFrame with a 'text' column.
        common_names: Optional pre-loaded lowercase name set (loaded on demand otherwise).

    Returns:
        Up to 5 candidate author lines (empty if none).
    """
    if df_lines.empty or "text" not in df_lines.columns:
        return []

    if common_names is None:
        common_names = _load_common_author_names()
    if not common_names:
        return []

    authors: list[str] = []

    # Authors typically appear at the start of the document.
    for _, row in df_lines.head(20).iterrows():
        text = str(row.get("text", "")).strip()
        if not text or len(text) > 100:
            continue
        if not _is_title_case(text):
            continue

        words = re.findall(r'\b[a-zA-Z]+\b', text)
        if any(w.lower() in common_names for w in words):
            authors.append(text)
            if len(authors) >= 5:
                break

    return authors


def extract_title_from_text(df_lines: pd.DataFrame) -> Optional[str]:
    """Guess the title as the largest-font qualifying line on the first pages.

    Scans pages 1→3 in order; on each, takes the first line at the maximum
    font size (length > 5, not a "page "/"chapter " artifact).

    Args:
        df_lines: DataFrame with 'page_number', 'font_size', 'text' columns.

    Returns:
        Title string or None.
    """
    if df_lines.empty:
        return None

    required_cols = ['page_number', 'font_size', 'text']
    if not all(col in df_lines.columns for col in required_cols):
        return None

    for page_num in (1, 2, 3):
        page_lines = df_lines[df_lines['page_number'] == page_num].copy()
        if page_lines.empty:
            continue

        page_lines = page_lines[
            page_lines['text'].notna() & (page_lines['text'].str.len() > 5)
        ]
        if page_lines.empty:
            continue

        max_font_size = page_lines['font_size'].max()
        if pd.isna(max_font_size):
            continue

        max_font_lines = page_lines[page_lines['font_size'] == max_font_size]
        if max_font_lines.empty:
            continue

        title = max_font_lines.iloc[0]['text']
        if isinstance(title, str) and title.strip():
            title = title.strip()
            if len(title) > 5 and not title.lower().startswith(('page ', 'chapter ')):
                return title

    return None


def extract_language_from_text(df_lines: pd.DataFrame) -> Optional[str]:
    """Detect the body language with langdetect over the first ~10k chars.

    Table content is excluded when a 'block_type' column is present. Returns a
    normalized 2-letter code, or None if langdetect is unavailable / unsure.

    Args:
        df_lines: DataFrame with a 'text' column (and optional 'block_type').

    Returns:
        Language code (e.g. "en") or None.
    """
    try:
        if df_lines.empty or 'text' not in df_lines.columns:
            return None

        if 'block_type' in df_lines.columns:
            text_lines = df_lines[df_lines['block_type'] != 'table']
        else:
            text_lines = df_lines
        if text_lines.empty:
            return None

        text_sample = ' '.join(text_lines['text'].dropna().astype(str))[:10000]
        if len(text_sample.strip()) < 50:
            return None

        try:
            from langdetect import detect
        except ImportError:
            return None

        return _normalize_language_code(detect(text_sample))

    except Exception:
        return None


# ─────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────

def add_text_fallbacks(doc_meta: Dict[str, Any], df_lines: Optional[pd.DataFrame]) -> None:
    """Populate the three text-derived fields on ``doc_meta`` in place.

    Sets ``author_text``, ``title_text`` and ``language_text`` (each None/empty
    when nothing is found). Does not resolve against native metadata or set the
    final title/author/language — that is the consolidation step's job.

    Args:
        doc_meta: Metadata dict, modified in place.
        df_lines: Parsed body lines, or None (all fields left as None).
    """
    if df_lines is None:
        doc_meta["author_text"] = None
        doc_meta["title_text"] = None
        doc_meta["language_text"] = None
        return

    author_text = extract_author_from_text(df_lines)
    doc_meta["author_text"] = author_text or None
    doc_meta["title_text"] = extract_title_from_text(df_lines)
    doc_meta["language_text"] = extract_language_from_text(df_lines)
