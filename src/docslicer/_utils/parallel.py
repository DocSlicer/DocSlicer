"""
Shared helpers for fanning page-oriented pipeline stages out over a process pool.

Pairs with cpu.resolve_worker_count(): the caller decides the pool width from
CPU topology, these helpers decide when a pool is worth it and how to split the
work. Used by the PDF word extractor, the OCR word extractor and the cell
builder; any new page-parallel stage should import from here rather than
growing its own copy.
"""

from __future__ import annotations

from typing import List, Sequence, TypeVar

T = TypeVar("T")

# Below this many pages a stage runs single-process: the pool's fixed costs
# (process spawn + interpreter/pandas import per worker, pickling work in and
# results out) outweigh the parallel speedup on small documents.
PARALLEL_PAGE_THRESHOLD = 50


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
