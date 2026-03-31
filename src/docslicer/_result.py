from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

try:
    from importlib.metadata import version as _pkg_version
    _SCHEMA_VERSION = _pkg_version("docslicer")
except Exception:
    _SCHEMA_VERSION = "0.1.0"

if TYPE_CHECKING:
    import pandas as pd


@dataclass
class BBox:
    x_left: float
    y_top: float
    x_right: float
    y_bottom: float


@dataclass
class Chunk:
    id: str
    parent_chunk_id: str | None                      # id of parent chunk in hierarchy
    chunk_index: int
    page_number: int
    page_label: str | None                           # e.g. "A-6", "iv" — distinct from page_number
    document_region: str                             # body | toc | exhibit | header | footer | coverpage
    heading: str | None                              # active heading text for this chunk
    path: list[str]                                  # full heading path from root, e.g. ["## Section 1", "### 1.1"]
    text: str
    char_count: int
    bbox: BBox | None                                 # PDF only
    link_url: list[str]                              # unique URLs found in chunk
    ixbrl_ids: list[str]                             # unique iXBRL IDs found in chunk
    table_ids: list[str]                             # table IDs referenced in chunk

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Block:
    id: str
    role: str                                        # paragraph | heading | table | toc | exhibits | navigation | …
    page_number: int
    page_label: str | None                           # e.g. "A-6", "iv" — distinct from page_number
    document_region: str                             # body | toc | exhibit | header | footer | coverpage
    text: str
    chunk_id: str | None                             # which chunk this block belongs to
    char_count: int
    bbox: BBox | None                                 # PDF only
    link_url: list[str]                              # unique URLs found in block
    ixbrl_ids: list[str]                             # unique iXBRL IDs found in block
    table_ids: list[str]                             # table IDs referenced in block

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class TableCell:
    row: int                                          # 0-indexed row position
    col: int                                          # 0-indexed column position
    rowspan: int                                      # rows spanned (>=1)
    colspan: int                                      # columns spanned (>=1)
    role: str                                         # header | row_label | value_numeric | value_text | footnote
    text: str
    bbox: BBox | None                                 # PDF only

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Table:
    id: str
    caption: str | None
    page_number: int
    page_label: str | None
    chunk_id: str
    bbox: BBox | None                                 # PDF only; union of all cell bboxes
    markdown: str                                     # convenience — full table as markdown
    cells: list[TableCell]

    def to_dataframe(self) -> "pd.DataFrame":
        import pandas as pd
        return pd.DataFrame([asdict(c) for c in self.cells])

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class DocMetadata:
    title: str | None
    author: str | None
    page_count: int
    language: str | None
    has_ocr: bool
    source_url: str | None

    def to_dict(self) -> dict:
        return asdict(self)



def _to_parquet(df: "pd.DataFrame", path: Path) -> None:
    try:
        df.to_parquet(path, index=False)
    except ImportError:
        csv_path = path.with_suffix(".csv")
        df.to_csv(csv_path, index=False)
        import warnings
        warnings.warn(
            f"pyarrow/fastparquet not installed — saved as CSV instead: {csv_path}\n"
            "Install with: pip install pyarrow",
            stacklevel=3,
        )


@dataclass
class ParseResult:
    chunks: list[Chunk]
    blocks: list[Block]
    tables: list[Table]
    metadata: DocMetadata
    pipeline_steps: dict[str, "pd.DataFrame"] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "schema": "DocSlicerResult",
            "version": _SCHEMA_VERSION,
            "metadata": self.metadata.to_dict(),
            "chunks": [c.to_dict() for c in self.chunks],
            "blocks": [b.to_dict() for b in self.blocks],
            "tables": [t.to_dict() for t in self.tables],
        }



    def chunks_df(self) -> "pd.DataFrame":
        import pandas as pd
        return pd.DataFrame([c.to_dict() for c in self.chunks])

    def blocks_df(self) -> "pd.DataFrame":
        import pandas as pd
        return pd.DataFrame([b.to_dict() for b in self.blocks])

    def tables_df(self) -> "pd.DataFrame":
        import pandas as pd
        return pd.DataFrame([t.to_dict() for t in self.tables])

    def save(self, path: str | Path) -> None:
        """Save parse results to disk.

        Directory path (no suffix): saves all levels as separate parquet files
        plus metadata.json::

            result.save("output/")
            # → output/chunks.parquet, blocks.parquet, tables.parquet, metadata.json

        File path: saves the level implied by the stem, in the requested format::

            result.save("chunks.csv")               # chunks as CSV
            result.save("output/tables.parquet")    # tables as parquet
            result.save("metadata.json")            # metadata only

        Supported stems: chunks, blocks, tables, metadata.
        Supported formats: .json, .jsonl, .csv, .parquet.
        Unknown stems fall back to chunks.
        """
        import pandas as pd

        path = Path(path)

        # Directory: save all levels
        if not path.suffix or path.is_dir():
            path.mkdir(parents=True, exist_ok=True)
            _to_parquet(self.chunks_df(), path / "chunks.parquet")
            _to_parquet(self.blocks_df(), path / "blocks.parquet")
            _to_parquet(self.tables_df(), path / "tables.parquet")
            (path / "metadata.json").write_text(
                json.dumps(self.metadata.to_dict(), indent=2, ensure_ascii=False)
            )
            return

        # File path: determine level from stem, format from suffix
        path.parent.mkdir(parents=True, exist_ok=True)
        stem = path.stem.lower()
        suffix = path.suffix.lower()

        level_map = {
            "chunks": lambda: [c.to_dict() for c in self.chunks],
            "blocks": lambda: [b.to_dict() for b in self.blocks],
            "tables": lambda: [t.to_dict() for t in self.tables],
        }

        if stem == "metadata":
            if suffix != ".json":
                raise ValueError("metadata can only be saved as .json")
            path.write_text(json.dumps(self.metadata.to_dict(), indent=2, ensure_ascii=False))
            return

        # Unknown stem (e.g. "result") → full document export
        if stem not in level_map:
            if suffix != ".json":
                raise ValueError(f"Unknown stem {stem!r}: use chunks/blocks/tables/metadata, or a .json path for a full export")
            path.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False))
            return

        rows_fn = level_map[stem]

        if suffix == ".json":
            path.write_text(json.dumps(rows_fn(), indent=2, ensure_ascii=False))
        elif suffix == ".jsonl":
            with path.open("w", encoding="utf-8") as f:
                for row in rows_fn():
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
        elif suffix == ".csv":
            pd.DataFrame(rows_fn()).to_csv(path, index=False, encoding="utf-8")
        elif suffix == ".parquet":
            _to_parquet(pd.DataFrame(rows_fn()), path)
        else:
            raise ValueError(f"Unsupported format: {suffix!r}. Use .json, .jsonl, .csv, or .parquet")

    def save_debug(self, path: str | Path) -> None:
        if not self.pipeline_steps:
            raise RuntimeError("No pipeline steps recorded. Re-run with debug=True.")
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        for name, df in self.pipeline_steps.items():
            df.to_csv(path / f"{name}.csv", index=False)
