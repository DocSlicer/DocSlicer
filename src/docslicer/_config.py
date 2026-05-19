from __future__ import annotations

import uuid as _uuid
from dataclasses import dataclass, field


@dataclass
class ParseConfig:
    max_chunk_size: int = 3200
    optimal_chunk_size: int = 1500
    min_chunk_size: int = 400
    extract_tables: bool = True
    regions: list[str] | None = None
    debug: bool = False
    extra_fields: list[str] = field(default_factory=list)
    run_id: str = field(default_factory=lambda: str(_uuid.uuid4()))


DEFAULT_CONFIG = ParseConfig()


@dataclass(frozen=True)
class LayoutConfig:
    """Immutable configuration for layout detection."""
    min_column_gap: float = 30.0
    table_score_threshold: float = 0.7
    heading_score_threshold: float = 1.5
    max_chunk_chars: int = 3200
    min_chunk_chars: int = 400
