# SPDX-FileCopyrightText: 2026 Market Framer Inc.
# SPDX-License-Identifier: AGPL-3.0-only
"""Route a URL to the right fetcher (SEC vs. generic HTTP) and return a ScrapedPage."""

from __future__ import annotations

from urllib.parse import urlparse

from .models import ScrapedPage, SourceType
from .fetchers.http_fetcher import GenericHttpFetcher
from .fetchers.sec_fetcher import SecHttpFetcher
from . import config

_HTTP_FETCHER = GenericHttpFetcher()
_SEC_FETCHER = SecHttpFetcher()


def _is_sec_url(url: str) -> bool:
    host = (urlparse(url).hostname or "").lower()
    return any(host == s or host.endswith("." + s) for s in config.SEC_HOSTS)


def fetch_url(url: str) -> ScrapedPage:
    """
    Fetch a URL and return raw bytes as a ScrapedPage.

    Routing:
    - SEC / Congress URLs → SecHttpFetcher (rate-limited, SEC-compliant UA)
    - All others → GenericHttpFetcher (static HTML, httpx)

    Note: for non-SEC URLs, Playwright rendering (JS-heavy sites, full box
    extraction) is handled by the HTML parser layer, not here.
    """
    if _is_sec_url(url):
        return _SEC_FETCHER.fetch(url, source_type=SourceType.SEC_EDGAR_HTML)
    return _HTTP_FETCHER.fetch(url, source_type=SourceType.GENERIC_HTTP)
