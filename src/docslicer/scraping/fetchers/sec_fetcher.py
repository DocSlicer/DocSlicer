# SPDX-FileCopyrightText: 2026 Market Framer Inc.
# SPDX-License-Identifier: AGPL-3.0-only
"""SecHttpFetcher — fetch SEC EDGAR documents with the required headers and throttling."""

from __future__ import annotations

import logging
import random
import threading
import time
from typing import Mapping
from urllib.parse import urljoin

import httpx

from ..models import ScrapedPage, SourceType
from .. import config

logger = logging.getLogger(__name__)


class SecHttpFetcher:
    """
    SEC-specific HTTP fetcher.

    Handles SEC EDGAR and similar government document hosts with:
    - SEC-compliant User-Agent (required by SEC fair-access policy)
    - Global rate limiting (~8 req/s, backing off on errors)
    - Exponential backoff for 429/502/503/504
    - Simple in-memory cache

    SEC fair-access policy requires a descriptive User-Agent in the format:
        "Company Name contact@example.com"

    Pass your user_agent on construction:
        SecHttpFetcher(user_agent="Acme Research contact@acme.com")
    """

    _RETRIABLE_STATUS_CODES = (429, 502, 503, 504)

    def __init__(
        self,
        user_agent: str = "docslicer contact@example.com",
        headers: Mapping[str, str] | None = None,
        timeout: float | None = None,
        max_bytes: int | None = None,
        max_redirects: int = config.MAX_REDIRECTS,
        use_memory_cache: bool = True,
    ) -> None:
        self._timeout = timeout or config.DEFAULT_TIMEOUT_SECONDS
        self._max_bytes = max_bytes or config.MAX_RESPONSE_BYTES
        self._max_redirects = max_redirects
        self._use_memory_cache = use_memory_cache

        sec_headers = {
            "User-Agent": user_agent,
            "Accept-Encoding": "gzip, deflate",
            "Connection": "keep-alive",
        }
        if headers:
            sec_headers.update(headers)

        self._client = httpx.Client(
            headers=sec_headers,
            timeout=self._timeout,
            follow_redirects=True,
            max_redirects=self._max_redirects,
        )

        # Rate limiting
        self._lock = threading.Lock()
        self._last_request_ts: float = 0.0
        self._normal_interval_sec: float = 0.12   # ~8.3 rps (under SEC 10 rps limit)
        self._reduced_interval_sec: float = 0.30  # ~3.3 rps when server is unhappy
        self._current_interval_sec: float = self._normal_interval_sec

        # Backoff config
        self._base_backoff_sec: float = 1.0
        self._max_backoff_sec: float = 30.0
        self._backoff_multiplier: float = 2.0
        self._jitter_factor: float = 0.1

        # Error tracking for adaptive rate limiting
        self._recent_errors: list[float] = []
        self._error_window_sec: float = 60.0
        self._error_threshold: int = 3
        self._recovery_time_sec: float = 300.0
        self._last_error_ts: float = 0.0

        self._memory_cache: dict[str, bytes] = {}

    def fetch(self, url: str, source_type: SourceType) -> ScrapedPage:
        # Cache check
        if self._use_memory_cache and url in self._memory_cache:
            content = self._memory_cache[url]
            content_str = content.decode("utf-8", errors="ignore")
            if "Undeclared Automated Tool" in content_str or "Your Request Originates from" in content_str:
                del self._memory_cache[url]
            else:
                return ScrapedPage(
                    url=url,
                    final_url=url,
                    source_type=source_type,
                    status_code=200,
                    raw_bytes=content,
                    content_type=None,
                    encoding=None,
                    fetcher_used="SecHttpFetcher (cached)",
                )

        max_retries = 3
        last_exc: Exception | None = None

        for attempt in range(max_retries + 1):
            try:
                final_url, status_code, raw_bytes, content_type, encoding = (
                    self._do_request(url)
                )

                # Detect SEC block page (returns 200 but with block content)
                content_str = raw_bytes.decode(encoding or "utf-8", errors="ignore")
                if "Undeclared Automated Tool" in content_str or "Your Request Originates from" in content_str:
                    raise RuntimeError(
                        f"SEC blocked request for {url} — check your User-Agent header"
                    )

                if self._use_memory_cache:
                    self._memory_cache[url] = raw_bytes

                return ScrapedPage(
                    url=url,
                    final_url=final_url,
                    source_type=source_type,
                    status_code=status_code,
                    raw_bytes=raw_bytes,
                    content_type=content_type,
                    encoding=encoding,
                    fetcher_used="SecHttpFetcher",
                )

            except httpx.HTTPStatusError as e:
                last_exc = e
                if e.response.status_code in self._RETRIABLE_STATUS_CODES and attempt < max_retries:
                    self._record_error()
                    time.sleep(self._backoff(attempt))
                    continue
                raise

            except httpx.RequestError as e:
                last_exc = e
                self._record_error()
                if attempt < max_retries:
                    time.sleep(self._backoff(attempt))
                    continue
                raise

        if last_exc:
            raise last_exc
        raise RuntimeError(f"SEC fetch failed for {url}")

    def _do_request(self, url: str) -> tuple[str, int, bytes, str | None, str | None]:
        with self._lock:
            self._update_rate_limiting()
            elapsed = time.monotonic() - self._last_request_ts
            if elapsed < self._current_interval_sec:
                time.sleep(self._jitter(self._current_interval_sec - elapsed))

            with self._client.stream("GET", url) as resp:
                resp.raise_for_status()
                chunks: list[bytes] = []
                total = 0
                for chunk in resp.iter_bytes():
                    total += len(chunk)
                    if total > self._max_bytes:
                        raise ValueError(f"Response exceeded {self._max_bytes} bytes for {url}")
                    chunks.append(chunk)
                raw_bytes = b"".join(chunks)
                result = (
                    str(resp.url),
                    resp.status_code,
                    raw_bytes,
                    resp.headers.get("Content-Type"),
                    resp.encoding,
                )

            self._last_request_ts = time.monotonic()
            return result

    def _jitter(self, delay: float) -> float:
        return max(0.05, delay + delay * self._jitter_factor * (random.random() * 2 - 1))

    def _backoff(self, attempt: int) -> float:
        base = self._base_backoff_sec * (self._backoff_multiplier ** attempt)
        return self._jitter(min(base, self._max_backoff_sec))

    def _record_error(self) -> None:
        now = time.monotonic()
        self._last_error_ts = now
        self._recent_errors.append(now)
        cutoff = now - self._error_window_sec
        self._recent_errors = [ts for ts in self._recent_errors if ts > cutoff]
        if len(self._recent_errors) >= self._error_threshold:
            self._current_interval_sec = self._reduced_interval_sec

    def _update_rate_limiting(self) -> None:
        now = time.monotonic()
        cutoff = now - self._error_window_sec
        self._recent_errors = [ts for ts in self._recent_errors if ts > cutoff]
        if (
            len(self._recent_errors) < self._error_threshold
            and self._last_error_ts > 0
            and (now - self._last_error_ts) > self._recovery_time_sec
        ):
            self._current_interval_sec = self._normal_interval_sec

    def clear_cache(self) -> None:
        with self._lock:
            self._memory_cache.clear()

    def __del__(self) -> None:
        if hasattr(self, "_client"):
            try:
                self._client.close()
            except Exception:
                pass
