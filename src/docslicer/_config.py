# SPDX-FileCopyrightText: 2026 Market Framer Inc.
# SPDX-License-Identifier: AGPL-3.0-only
"""ParseConfig — user-facing knobs for parsing, chunking, OCR, and concurrency."""

from __future__ import annotations

import uuid as _uuid
from dataclasses import dataclass, field


@dataclass
class ParseConfig:
    """User-facing configuration for a parse.

    Chunking: ``max_chunk_size`` / ``optimal_chunk_size`` / ``min_chunk_size``
    bound chunk length in characters (always — ``exact_tokens`` changes how the
    resulting chunks are *counted*, not where they are cut);
    ``chunking`` toggles chunking entirely and ``merge_small_chunks`` folds
    undersized chunks into neighbours. ``table_representation`` selects how
    tables are serialized ("markdown", "jsonl", or "melted"). ``extra_fields``
    surfaces extra pipeline columns on the result, and ``debug`` retains the
    intermediate ``pipeline_steps`` DataFrames.

    Concurrency: ``max_workers`` sets the intra-document process-pool width
    (None auto-sizes to performance cores; 1 disables it). Format-specific knobs
    (``password``, ``use_browser``, ``include_*``) are documented inline below.
    Invalid combinations raise ``ValueError`` at construction.
    """

    max_chunk_size: int = 3200
    optimal_chunk_size: int = 1500
    min_chunk_size: int = 700
    chunking: bool = True
    merge_small_chunks: bool = True
    exact_tokens: bool = False
    table_representation: str = "markdown"
    debug: bool = False
    extra_fields: list[str] = field(default_factory=list)
    run_id: str = field(default_factory=lambda: str(_uuid.uuid4()))
    password: str | None = None
    # Process-pool width for CPU-bound steps within this single document (PDF
    # word extraction, cell building, OCR). None -> auto (performance-core
    # count, see _utils.cpu.resolve_worker_count). Set to 1 to disable
    # intra-document parallelism entirely — e.g. when a caller is already
    # parallelizing across documents (DocumentParser(workers=N)) and wants to
    # avoid oversubscribing the machine with nested process pools.
    max_workers: int | None = None
    # HTML only: when False, skip Playwright and use the static (BeautifulSoup)
    # box extractor even if Playwright is installed. ~15x faster and needs no
    # browser, but loses layout coordinates and CSS-class-resolved styling —
    # see docslicer.html.step_01_static_box_extractor for the tradeoffs.
    use_browser: bool = True
    # PPTX only: when False, speaker notes are excluded from extraction.
    include_speaker_notes: bool = True
    # DOCX only: when True, header/footer content is surfaced in df_paragraphs
    # and df_lines with block_type "header" / "footer".
    include_headers_footers: bool = False
    # DOCX only: footnotes/endnotes are document content, included by default.
    include_footnotes: bool = True
    # DOCX only: reviewer comments are annotations, not content — excluded by default.
    include_comments: bool = False

    def __post_init__(self) -> None:
        valid_representations = {"markdown", "jsonl", "melted"}
        if self.table_representation not in valid_representations:
            raise ValueError(
                f"table_representation must be one of "
                f"{sorted(valid_representations)}, got {self.table_representation!r}."
            )

        for name in ("max_chunk_size", "optimal_chunk_size", "min_chunk_size"):
            value = getattr(self, name)
            if not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer, got {value!r}.")
        if not (self.min_chunk_size <= self.optimal_chunk_size <= self.max_chunk_size):
            raise ValueError(
                "chunk sizes must satisfy min_chunk_size <= optimal_chunk_size <= "
                f"max_chunk_size, got min={self.min_chunk_size}, "
                f"optimal={self.optimal_chunk_size}, max={self.max_chunk_size}."
            )

        if self.max_workers is not None and (
            not isinstance(self.max_workers, int) or self.max_workers < 1
        ):
            raise ValueError(
                f"max_workers must be None or a positive integer, got {self.max_workers!r}."
            )

        if not isinstance(self.extra_fields, list) or not all(
            isinstance(f, str) for f in self.extra_fields
        ):
            raise ValueError("extra_fields must be a list of column-name strings.")


DEFAULT_CONFIG = ParseConfig()
