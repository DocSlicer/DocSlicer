from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd


@dataclass
class Chunk:
    id: str
    text: str
    page: int
    hierarchy: list[str]
    region: str
    chunk_index: int
    char_count: int
    bbox: tuple[float, float, float, float] | None  # (x_left, y_top, x_right, y_bottom), PDF only

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "text": self.text,
            "page": self.page,
            "hierarchy": self.hierarchy,
            "region": self.region,
            "chunk_index": self.chunk_index,
            "char_count": self.char_count,
            "bbox": list(self.bbox) if self.bbox else None,
        }


@dataclass
class Block:
    id: str
    text: str
    page: int
    role: str                                        # paragraph | heading | table | toc | exhibits | navigation | …
    region: str                                      # body | toc | exhibit | header | footer
    chunk_id: str | None                             # which chunk this block belongs to
    char_count: int
    bbox: tuple[float, float, float, float] | None  # (x_left, y_top, x_right, y_bottom), PDF only

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "text": self.text,
            "page": self.page,
            "role": self.role,
            "region": self.region,
            "chunk_id": self.chunk_id,
            "char_count": self.char_count,
            "bbox": list(self.bbox) if self.bbox else None,
        }


@dataclass
class Table:
    id: str
    caption: str | None
    page: int
    markdown: str
    chunk_id: str
    bbox: tuple[float, float, float, float] | None  # (x_left, y_top, x_right, y_bottom), PDF only

    def to_dataframe(self) -> "pd.DataFrame":
        import pandas as pd
        import io
        lines = [l for l in self.markdown.splitlines() if l.strip()]
        if len(lines) < 2:
            return pd.DataFrame()
        # Strip leading/trailing pipes and split on |
        rows = [[c.strip() for c in l.strip().strip("|").split("|")] for l in lines if not set(l.replace("|", "").replace("-", "").replace(" ", "")) == set()]
        if not rows:
            return pd.DataFrame()
        return pd.DataFrame(rows[1:], columns=rows[0])

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "caption": self.caption,
            "page": self.page,
            "markdown": self.markdown,
            "chunk_id": self.chunk_id,
            "bbox": list(self.bbox) if self.bbox else None,
        }


@dataclass
class DocMetadata:
    title: str | None
    author: str | None
    page_count: int
    language: str | None
    has_ocr: bool
    source_url: str | None

    def to_dict(self) -> dict:
        return {
            "title": self.title,
            "author": self.author,
            "page_count": self.page_count,
            "language": self.language,
            "has_ocr": self.has_ocr,
            "source_url": self.source_url,
        }


@dataclass
class ParseResult:
    chunks: list[Chunk]
    blocks: list[Block]
    tables: list[Table]
    metadata: DocMetadata
    pipeline_steps: dict[str, "pd.DataFrame"] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "metadata": self.metadata.to_dict(),
            "chunks": [c.to_dict() for c in self.chunks],
            "blocks": [b.to_dict() for b in self.blocks],
            "tables": [t.to_dict() for t in self.tables],
        }

    def save(self, path: str | Path) -> None:
        path = Path(path)
        suffix = path.suffix.lower()
        if suffix == ".json":
            path.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False))
        elif suffix == ".jsonl":
            with path.open("w", encoding="utf-8") as f:
                for chunk in self.chunks:
                    f.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n")
        elif suffix == ".csv":
            import pandas as pd
            pd.DataFrame([c.to_dict() for c in self.chunks]).to_csv(path, index=False)
        elif suffix == ".parquet":
            import pandas as pd
            pd.DataFrame([c.to_dict() for c in self.chunks]).to_parquet(path, index=False)
        else:
            raise ValueError(f"Unsupported format: {suffix!r}. Use .json, .jsonl, .csv, or .parquet")

    def save_debug(self, path: str | Path) -> None:
        if not self.pipeline_steps:
            raise RuntimeError("No pipeline steps recorded. Re-run with debug=True.")
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        for name, df in self.pipeline_steps.items():
            df.to_csv(path / f"{name}.csv", index=False)
