# SPDX-FileCopyrightText: 2026 Market Framer Inc.
# SPDX-License-Identifier: AGPL-3.0-only
"""URL fetching and web-scraping support."""

from .dispatcher import fetch_url
from .models import ScrapedPage, SourceType
from .fetchers.sec_fetcher import SecHttpFetcher

__all__ = ["fetch_url", "ScrapedPage", "SourceType", "SecHttpFetcher"]
