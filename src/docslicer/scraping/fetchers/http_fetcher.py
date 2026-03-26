from __future__ import annotations

from typing import Mapping

import httpx

from ..models import ScrapedPage, SourceType
from .. import config


class GenericHttpFetcher:
    """
    Lightweight HTTP fetcher using httpx.

    Used as a static-HTML fallback when Playwright is not installed.
    Not suitable for JS-rendered pages or bot-protected sites.
    """

    def __init__(
        self,
        headers: Mapping[str, str] | None = None,
        timeout: float | None = None,
        max_bytes: int | None = None,
    ) -> None:
        self._max_bytes = max_bytes or config.MAX_RESPONSE_BYTES
        self._client = httpx.Client(
            headers={**config.DEFAULT_HTTP_HEADERS, **(headers or {})},
            timeout=timeout or config.DEFAULT_TIMEOUT_SECONDS,
            follow_redirects=True,
            max_redirects=config.MAX_REDIRECTS,
        )

    def fetch(self, url: str, source_type: SourceType) -> ScrapedPage:
        with self._client.stream("GET", url) as resp:
            resp.raise_for_status()
            chunks: list[bytes] = []
            total = 0
            for chunk in resp.iter_bytes():
                total += len(chunk)
                if total > self._max_bytes:
                    raise ValueError(
                        f"Response exceeded {self._max_bytes} bytes for {url}"
                    )
                chunks.append(chunk)
            raw_bytes = b"".join(chunks)

        return ScrapedPage(
            url=url,
            final_url=str(resp.url),
            source_type=source_type,
            status_code=resp.status_code,
            raw_bytes=raw_bytes,
            content_type=resp.headers.get("Content-Type"),
            encoding=resp.encoding,
            fetcher_used="GenericHttpFetcher",
        )

    def __del__(self) -> None:
        if hasattr(self, "_client"):
            try:
                self._client.close()
            except Exception:
                pass
