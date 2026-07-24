# SPDX-FileCopyrightText: 2026 Market Framer Inc.
# SPDX-License-Identifier: AGPL-3.0-only
"""safe_enrich — run an optional enrichment step, logging and swallowing its failures."""

# _utils/safe_call.py
from __future__ import annotations

import logging
from typing import Any, Callable


def safe_enrich(
    fn: Callable[..., None],
    discovered_metadata: dict[str, Any],
    *args: Any,
    fallback: dict[str, Any],
    logger: logging.Logger,
    **kwargs: Any,
) -> None:
    """Run a metadata-enrichment call that mutates `discovered_metadata` in place.

    Enrichment (author/title/language/page-count detection, ...) is best-effort:
    a failure there shouldn't sink the whole parse. On exception, log it and
    backfill `fallback` values via setdefault so downstream code still finds
    the keys it expects.
    """
    try:
        fn(discovered_metadata, *args, **kwargs)
    except Exception as e:
        logger.error(f"Error in {fn.__name__}: {e}", exc_info=True)
        for key, value in fallback.items():
            discovered_metadata.setdefault(key, value)
