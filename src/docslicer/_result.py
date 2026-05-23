from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from .metadata.schema import DocumentMetadata

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
    section: str                                     # body | toc | exhibit | header | footer | coverpage
    heading: str | None                              # active heading text for this chunk
    path: list[str]                                  # full heading path from root, e.g. ["## Section 1", "### 1.1"]
    text: str
    char_count: int
    bbox: BBox | None                                 # PDF only
    link_url: list[str]                              # unique URLs found in chunk
    ixbrl_ids: list[str]                             # unique iXBRL IDs found in chunk
    table_ids: list[str]                             # table IDs referenced in chunk
    extra: dict = field(default_factory=dict)        # caller-requested extra fields from the pipeline df

    def to_dict(self) -> dict:
        d = asdict(self)
        if not d["ixbrl_ids"]:
            del d["ixbrl_ids"]
        return d


@dataclass
class Block:
    id: str
    role: str                                        # paragraph | heading | table | toc | exhibits | navigation | …
    page_number: int
    page_label: str | None                           # e.g. "A-6", "iv" — distinct from page_number
    section: str                                     # body | toc | exhibit | header | footer | coverpage
    text: str
    chunk_id: str | None                             # which chunk this block belongs to
    char_count: int
    bbox: BBox | None                                 # PDF only
    link_url: list[str]                              # unique URLs found in block
    ixbrl_ids: list[str]                             # unique iXBRL IDs found in block
    table_ids: list[str]                             # table IDs referenced in block
    extra: dict = field(default_factory=dict)        # caller-requested extra fields from the pipeline df

    def to_dict(self) -> dict:
        d = asdict(self)
        if not d["ixbrl_ids"]:
            del d["ixbrl_ids"]
        return d


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
class HierarchyNode:
    heading_id: int
    text: str
    level: int
    heading_type: str
    page_number: int | None
    page_label: str | None
    chunk_ids: list[str]
    children: list[HierarchyNode]

    def to_dict(self, minimal: bool = False) -> dict:
        if minimal:
            d: dict = {"text": self.text}
            if self.children:
                d["children"] = [c.to_dict(minimal=True) for c in self.children]
            return d
        return asdict(self)


@dataclass
class HierarchyTree:
    roots: list[HierarchyNode]

    def __iter__(self):
        return iter(self.roots)

    def __len__(self) -> int:
        return len(self.roots)

    def flatten(self) -> list[HierarchyNode]:
        result: list[HierarchyNode] = []
        def _visit(node: HierarchyNode) -> None:
            result.append(node)
            for child in node.children:
                _visit(child)
        for root in self.roots:
            _visit(root)
        return result

    def to_dict(self, minimal: bool = False) -> list[dict]:
        return [r.to_dict(minimal=minimal) for r in self.roots]

    def save(self, path: str | Path, minimal: bool = False, indent: int = 2) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.to_dict(minimal=minimal), indent=indent), encoding="utf-8")

    def save_outline(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(self.to_outline(), encoding="utf-8")

    def to_outline(self) -> str:
        lines: list[str] = []
        def _visit(node: HierarchyNode, depth: int) -> None:
            lines.append("  " * depth + "- " + node.text)
            for child in node.children:
                _visit(child, depth + 1)
        for root in self.roots:
            _visit(root, 0)
        return "\n".join(lines)


def _prettify_table(markdown: str) -> str:
    """Reformat a markdown table so pipe characters are vertically aligned."""
    lines = markdown.strip().splitlines()
    if not lines:
        return markdown

    def _split_row(line: str) -> list[str]:
        cells = line.strip().split("|")
        if cells and cells[0].strip() == "":
            cells = cells[1:]
        if cells and cells[-1].strip() == "":
            cells = cells[:-1]
        return [c.strip() for c in cells]

    def _is_sep(cell: str) -> bool:
        return bool(cell) and all(c in "-:" for c in cell)

    rows = [_split_row(l) for l in lines if l.strip().startswith("|")]
    if not rows:
        return markdown

    n_cols = max(len(r) for r in rows)
    for row in rows:
        while len(row) < n_cols:
            row.append("")

    sep_indices = {i for i, row in enumerate(rows) if all(_is_sep(c) for c in row if c)}
    col_widths = [3] * n_cols
    for i, row in enumerate(rows):
        if i not in sep_indices:
            for j, cell in enumerate(row):
                col_widths[j] = max(col_widths[j], len(cell))

    out: list[str] = []
    for i, row in enumerate(rows):
        if i in sep_indices:
            parts = []
            for j, cell in enumerate(row):
                w = col_widths[j]
                lc, rc = cell.startswith(":"), cell.endswith(":") and len(cell) > 1
                if lc and rc:
                    parts.append(":" + "-" * (w - 2) + ":")
                elif lc:
                    parts.append(":" + "-" * (w - 1))
                elif rc:
                    parts.append("-" * (w - 1) + ":")
                else:
                    parts.append("-" * w)
        else:
            parts = [cell.ljust(col_widths[j]) for j, cell in enumerate(row)]
        out.append("| " + " | ".join(parts) + " |")

    return "\n".join(out)


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
    metadata: DocumentMetadata
    hierarchy: HierarchyTree = field(default_factory=lambda: HierarchyTree(roots=[]))
    pipeline_steps: dict[str, "pd.DataFrame"] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "schema": "DocSlicerResult",
            "version": _SCHEMA_VERSION,
            "metadata": self.metadata.to_dict(),
            "chunks": [c.to_dict() for c in self.chunks],
            "blocks": [b.to_dict() for b in self.blocks],
            "tables": [t.to_dict() for t in self.tables],
            "hierarchy": self.hierarchy.to_dict(),
        }

    def export_to_dict(self) -> dict:
        return self.to_dict()

    def export_to_markdown(
        self,
        include_page_markers: bool = True,
        include_tables: bool = True,
        include_headers_footers: bool = False,
        include_toc: bool = False,
        prettify: bool = False,
    ) -> str:
        """Render the document as Markdown using blocks as the source of truth."""
        _HEADING_ROLES = {
            "heading", "toc_heading", "exhibit_heading", "hybrid_heading_paragraph",
        }
        _SKIP_ROLES = {"navigation", "suppressed_repeated_heading"}
        excluded_sections: set[str] = set()
        if not include_headers_footers:
            excluded_sections |= {"header", "footer"}
        if not include_toc:
            excluded_sections.add("toc")

        level_map: dict[str, int] = {
            node.text: node.level
            for node in reversed(self.hierarchy.flatten())
        }
        tables_by_id = {t.id: t for t in self.tables}

        parts: list[str] = []
        prev_page: int | None = None

        for block in self.blocks:
            if block.section in excluded_sections:
                continue
            if block.role in _SKIP_ROLES:
                continue

            text = block.text.strip()

            if include_page_markers and block.page_number != prev_page:
                prev_page = block.page_number
                label = block.page_label or str(block.page_number)
                parts.append(f"<!-- page {label} -->")

            if block.role in _HEADING_ROLES:
                level = level_map.get(text, 2)
                prefix = "#" * max(1, min(6, level))
                parts.append(f"{prefix} {text}")
            elif block.role == "table":
                if include_tables:
                    table = tables_by_id.get(block.table_ids[0]) if block.table_ids else None
                    raw = table.markdown if table else text
                    parts.append(_prettify_table(raw) if prettify else raw)
            elif text:
                parts.append(text)

        return "\n\n".join(parts)

    def export_to_text(
        self,
        include_tables: bool = True,
        include_headers_footers: bool = False,
        include_toc: bool = False,
    ) -> str:
        """Render the document as plain text (no Markdown formatting)."""
        _HEADING_ROLES = {
            "heading", "toc_heading", "exhibit_heading", "hybrid_heading_paragraph",
        }
        _SKIP_ROLES = {"navigation", "suppressed_repeated_heading"}
        excluded_sections: set[str] = set()
        if not include_headers_footers:
            excluded_sections |= {"header", "footer"}
        if not include_toc:
            excluded_sections.add("toc")

        tables_by_id = {t.id: t for t in self.tables}

        parts: list[str] = []

        for block in self.blocks:
            if block.section in excluded_sections:
                continue
            if block.role in _SKIP_ROLES:
                continue

            text = block.text.strip()

            if block.role in _HEADING_ROLES:
                if text:
                    parts.append(text)
            elif block.role == "table":
                if include_tables:
                    table = tables_by_id.get(block.table_ids[0]) if block.table_ids else None
                    parts.append(table.markdown if table else text)
            elif text:
                parts.append(text)

        return "\n\n".join(parts)

    def export_tables_csv(self, path: str | Path, encoding: str = "utf-8") -> None:
        """Save all tables as CSV.

        Each table is preceded by a one-row header (table id + page) and a
        caption row when present. Rowspan/colspan are expanded by duplicating
        the cell text into every covered grid position. Tables are separated by
        two blank rows.

        Pass encoding="utf-8-sig" for Excel-friendly output (adds BOM).
        """
        import csv as _csv
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with path.open("w", newline="", encoding=encoding) as f:
            writer = _csv.writer(f)
            for table in self.tables:
                label = table.page_label or str(table.page_number)
                writer.writerow([f"Table {table.id} | Page {label}"])
                if table.caption:
                    writer.writerow([table.caption])

                if not table.cells:
                    writer.writerow(["(no cells)"])
                    writer.writerow([])
                    writer.writerow([])
                    continue

                max_rows = max(c.row + c.rowspan for c in table.cells)
                max_cols = max(c.col + c.colspan for c in table.cells)
                grid: list[list[str]] = [[""] * max_cols for _ in range(max_rows)]

                for cell in table.cells:
                    text = cell.text.replace("\n", " ").strip()
                    for r in range(cell.row, min(cell.row + cell.rowspan, max_rows)):
                        for c in range(cell.col, min(cell.col + cell.colspan, max_cols)):
                            grid[r][c] = text

                for row in grid:
                    writer.writerow(row)

                writer.writerow([])
                writer.writerow([])

    def export_chunks_csv(self, path: str | Path, encoding: str = "utf-8") -> None:
        """Save chunks as CSV. Pass encoding="utf-8-sig" for Excel-friendly output (adds BOM)."""
        import pandas as pd
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([c.to_dict() for c in self.chunks]).to_csv(path, index=False, encoding=encoding)

    def chunks_to_jsonl(self) -> str:
        """Return chunks as a newline-delimited JSON string (one chunk per line), no file I/O."""
        return "\n".join(json.dumps(c.to_dict(), ensure_ascii=False) for c in self.chunks)

    def export_chunks_jsonl(self, path: str | Path) -> None:
        """Save chunks as newline-delimited JSON (one chunk per line)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            for chunk in self.chunks:
                f.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n")

    def export_chunks_parquet(self, path: str | Path) -> None:
        """Save chunks as Parquet (falls back to CSV if pyarrow is not installed)."""
        import pandas as pd
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        _to_parquet(pd.DataFrame([c.to_dict() for c in self.chunks]), path)

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
