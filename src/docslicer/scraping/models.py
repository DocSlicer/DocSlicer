# SPDX-FileCopyrightText: 2026 Market Framer Inc.
# SPDX-License-Identifier: AGPL-3.0-only
"""Scraping data models: SourceType and ScrapedPage."""

# backend/app/services/scraping/models.py

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class SourceType(str, Enum):
    """
    Classification of the source being scraped.

    This allows the dispatcher to decide which fetcher to use,
    and gives downstream consumers context about the origin.
    """

    # Default for everything else (CNBC, GNW, Congress, etc.)
    GENERIC_HTTP = "generic_http"

    # SEC HTML (EDGAR filings, exhibits, etc.)
    SEC_EDGAR_HTML = "sec_edgar_html"

    # Hosted Documents
    PDF_DOCUMENT   = "pdf_document"
    EXCEL_DOCUMENT = "excel_document"

    # (add more later, e.g.)
    # CONGRESS = "congress_bill"
    # JS_HEAVY = "js_heavy" - render with playwright
    # API_JSON = "api_json"


@dataclass
class ScrapedPage:
    """
    Result of a network fetch.

    NOTE:
    - raw_bytes is the authoritative payload (kept exactly as delivered).
    - encoding is the *best guess* from the response headers or HTML meta.
    - html decoding should usually happen in the parsing pipeline.
    """

    # Original URL requested
    url: str

    # Final resolved URL after redirects
    final_url: str

    # Which logical source this belonged to (SEC vs generic, etc.)
    source_type: SourceType

    # HTTP status code (e.g., 200)
    status_code: int

    # Raw response bytes (full body, subject to size limits)
    raw_bytes: bytes

    # Content-Type header (may include charset)
    content_type: str | None

    # httpx-detected encoding (may be None)
    encoding: str | None

    # Which fetcher was used to retrieve this page (for debugging)
    fetcher_used: str | None = None
