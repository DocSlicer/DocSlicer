from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ParseConfig:
    max_chunk_size: int = 3200
    optimal_chunk_size: int = 1500
    min_chunk_size: int = 400
    extract_tables: bool = True
    regions: list[str] | None = None
    debug: bool = False


DEFAULT_CONFIG = ParseConfig()


@dataclass(frozen=True)
class LayoutConfig:
    """Immutable configuration for layout detection."""
    min_column_gap: float = 30.0
    table_score_threshold: float = 0.7
    heading_score_threshold: float = 1.5
    max_chunk_chars: int = 3200
    min_chunk_chars: int = 400
