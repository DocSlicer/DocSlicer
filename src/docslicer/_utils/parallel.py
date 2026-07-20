"""
Shared helpers for fanning page-oriented pipeline stages out over a process pool.

Pairs with cpu.resolve_worker_count(): the caller decides the pool width from
CPU topology, these helpers decide when a pool is worth it and how to split the
work. Any new page-parallel stage should import from here rather than growing
its own copy.

Not every stage gates on a page threshold, and the ones that do don't share a
single break-even point:
  - PDF word extractor: goes parallel at PARALLEL_PAGE_THRESHOLD pages.
  - cell builder: much cheaper per page, so its own higher
    CELL_BUILDER_PARALLEL_PAGE_THRESHOLD.
  - OCR word extractor: no page threshold — Tesseract per page is expensive
    enough that it fans out whenever resolve_worker_count() yields >1 worker
    (i.e. >=2 pages on a multi-core box). It only borrows warn_pool_fell_back.
"""

from __future__ import annotations

import logging
from typing import List, Sequence, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


def warn_pool_fell_back(stage: str) -> None:
    """Log the once-per-stage warning emitted when a process pool can't start.

    A ``BrokenProcessPool`` at pool startup almost always means the calling
    process's ``__main__`` module runs work at import time without an
    ``if __name__ == "__main__":`` guard: under the ``spawn`` start method
    (macOS/Windows default) each worker re-imports ``__main__`` and crashes.
    Rather than fail the whole parse, callers catch it and re-run the stage
    single-process; this explains the (recoverable) slowdown in the logs.
    """
    logger.warning(
        "%s: process pool could not start (BrokenProcessPool); falling back to "
        "single-process. If you call docslicer from a script, guard the entry "
        "point with `if __name__ == \"__main__\":` to restore parallelism.",
        stage,
    )

# Below this many pages a stage runs single-process: the pool's fixed costs
# (process spawn + interpreter/pandas import per worker, pickling work in and
# results out) outweigh the parallel speedup on small documents.
PARALLEL_PAGE_THRESHOLD = 50

# The cell builder's per-page work is far cheaper than the word extractor's, so
# the same fixed pool costs only pay off on much larger documents. Its own,
# higher break-even lives here rather than in the step so all the parallelism
# thresholds stay together.
CELL_BUILDER_PARALLEL_PAGE_THRESHOLD = 1500


def chunk_evenly(items: Sequence[T], n_chunks: int) -> List[List[T]]:
    """Split ``items`` into up to ``n_chunks`` contiguous, near-equal chunks.

    Chunk sizes differ by at most one. Empty chunks are omitted, so fewer than
    ``n_chunks`` lists come back when ``len(items) < n_chunks``.
    """
    k, rem = divmod(len(items), n_chunks)
    chunks: List[List[T]] = []
    start = 0
    for i in range(n_chunks):
        end = start + k + (1 if i < rem else 0)
        if start < end:
            chunks.append(list(items[start:end]))
        start = end
    return chunks
