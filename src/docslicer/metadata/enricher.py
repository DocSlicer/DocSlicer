"""Orchestration layer — calls all add_* enrichment functions in sequence."""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Dict, Any

import pandas as pd

from .author import add_author_info
from .title import add_title_info
from .language import add_language_info
from .profile import add_profile_info


def add_document_information(
    doc_meta: Dict[str, Any],
    pdf_path: Optional[Path] = None,
    html_content: Optional[str] = None,
    df_lines: Optional[pd.DataFrame] = None,
) -> None:
    """
    Consolidated function to add all document information metadata.

    This function calls all individual metadata extraction functions:
    - add_author_info: Extracts author information
    - add_title_info: Extracts title information
    - add_language_info: Extracts and resolves language information
    - add_profile_info: Detects document profile

    Args:
        doc_meta: Metadata dict to update (modified in-place)
        pdf_path: Path to PDF file (for PDF content_type)
        html_content: HTML content string (for HTML content_type)
        df_lines: DataFrame with text lines (for text-based detection)

    Modifies:
        doc_meta dictionary in-place
    """
    # For HTML, truncate before parsing — metadata is always near the top.
    # Parsing a full 400-page SEC filing 3× is the main performance bottleneck.
    # - If <head> exists: use up to </head> (captures all standard meta tags)
    # - Otherwise: use first 50 KB (enough for <html lang>, <title>, first <h1>)
    _HTML_FALLBACK_CHARS = 50_000
    html_head = html_content
    if html_content:
        head_end = html_content.lower().find('</head>')
        if head_end != -1:
            html_head = html_content[:head_end + 7]
        elif len(html_content) > _HTML_FALLBACK_CHARS:
            html_head = html_content[:_HTML_FALLBACK_CHARS]

    add_author_info(doc_meta, pdf_path=pdf_path, html_content=html_head, df_lines=df_lines)
    add_title_info(doc_meta, pdf_path=pdf_path, html_content=html_head, df_lines=df_lines)
    add_language_info(doc_meta, pdf_path=pdf_path, html_content=html_head, df_lines=df_lines)
    add_profile_info(doc_meta, df_lines=df_lines)
