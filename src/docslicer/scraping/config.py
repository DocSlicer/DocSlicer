from __future__ import annotations


# =============================
# General Networking Defaults
# =============================

BROWSER_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)

DEFAULT_HTTP_HEADERS = {
    "User-Agent": BROWSER_USER_AGENT,
    "Accept": (
        "text/html,application/xhtml+xml,application/xml;q=0.9,"
        "image/avif,image/webp,image/apng,*/*;q=0.8,"
        "application/signed-exchange;v=b3;q=0.7"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br, zstd",
    "Cache-Control": "no-cache",
    "Pragma": "no-cache",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
    "Sec-Ch-Ua": '"Chromium";v="120", "Not_A Brand";v="24"',
    "Sec-Ch-Ua-Mobile": "?0",
    "Sec-Ch-Ua-Platform": '"macOS"',
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "same-origin",
    "Sec-Fetch-User": "?1",
}

DEFAULT_TIMEOUT_SECONDS: float = 30.0

# Maximum response body size.
MAX_RESPONSE_BYTES: int = 50 * 1024 * 1024  # 50 MB

MAX_REDIRECTS: int = 5


# =============================
# SEC Routing
# =============================

# Hostnames routed to SecHttpFetcher (rate-limited, SEC-compliant UA).
SEC_HOSTS = {
    "sec.gov",
    "congress.gov",
}


# =============================
# PDF Detection
# =============================

PDF_CONTENT_TYPES = {
    "application/pdf",
    "application/x-pdf",
    "application/acrobat",
}


def is_likely_pdf(content_type: str | None) -> bool:
    if not content_type:
        return False
    ct = content_type.split(";")[0].strip().lower()
    return ct in PDF_CONTENT_TYPES
