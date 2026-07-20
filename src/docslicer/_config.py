from __future__ import annotations

import uuid as _uuid
from dataclasses import dataclass, field


@dataclass
class ParseConfig:
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


DEFAULT_CONFIG = ParseConfig()
