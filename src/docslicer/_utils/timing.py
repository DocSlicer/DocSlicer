# _utils/timing.py
from __future__ import annotations

import logging
from contextlib import contextmanager
from time import perf_counter
from typing import Any, Iterator, Optional


# ==================================================
# STEP TIMING
# ==================================================
#
# Reusable across pipelines: any orchestrator (pdf, ocr, docx, ...) can time a
# step without hand-rolling t0 = perf_counter() / append(...) pairs, and
# without threading a `timings` list through its return signature. Timing is
# reported as a side effect (structured log record), not a return value, so a
# step can be timed without changing what the function returns to its callers.
#
# On exception, the elapsed time is still logged (at WARNING, with the
# exception noted) before the exception propagates — a step that ran for 40s
# before crashing is exactly the kind of thing you want a timing trail for.

def _log(logger: logging.Logger, level: int, step_name: str, duration: float,
          extra_meta: Optional[dict[str, Any]], failed: bool) -> None:
    log_data: dict[str, Any] = {
        "event": "pipeline_step_failed" if failed else "pipeline_step_completed",
        "step_name": step_name,
        "duration_sec": round(duration, 4),
    }
    if extra_meta:
        # extra_meta wins on key collision — caller-supplied metadata is more
        # specific than our generic fields.
        log_data.update(extra_meta)

    suffix = " (failed)" if failed else ""
    logger.log(level, "%-35s %7.3fs%s", step_name, duration, suffix, extra=log_data)


@contextmanager
def timed_step(
    step_name: str,
    *,
    logger: Optional[logging.Logger] = None,
    extra_meta: Optional[dict[str, Any]] = None,
) -> Iterator[None]:
    """Context manager that logs how long the wrapped block took to run.

    Usage:
        with timed_step("OCR word extraction", logger=logger):
            df_words = extract_words_from_images(...)

    Pass `logger` so log lines attribute to the calling pipeline (ocr, pdf,
    docx, ...) rather than a shared/generic name; falls back to this module's
    logger if omitted.
    """
    active_logger = logger or logging.getLogger(__name__)
    t0 = perf_counter()
    try:
        yield
    except Exception:
        _log(active_logger, logging.WARNING, step_name, perf_counter() - t0, extra_meta, failed=True)
        raise
    else:
        _log(active_logger, logging.INFO, step_name, perf_counter() - t0, extra_meta, failed=False)
