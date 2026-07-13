from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING


def _format_hierarchy_json(nodes: list[dict], indent: int = 2) -> str:
    def _fmt(node: dict, level: int) -> str:
        pad = " " * (level * indent)
        if not node.get("children"):
            return f'{pad}{{"text": {json.dumps(node["text"])}}}'
        child_lines = ",\n".join(_fmt(c, level + 1) for c in node["children"])
        return (
            f'{pad}{{"text": {json.dumps(node["text"])}, "children": [\n'
            f'{child_lines}\n'
            f'{pad}]}}'
        )
    return "[\n" + ",\n".join(_fmt(n, 1) for n in nodes) + "\n]\n"


def _norm_id(v: object) -> str:
    """Normalise a table/block id that may have been serialised as '1.0' → '1'."""
    s = str(v).strip()
    if s.endswith(".0") and s[:-2].lstrip("-").isdigit():
        return s[:-2]
    return s

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

    @classmethod
    def from_dict(cls, d: dict | None) -> "BBox | None":
        if d is None:
            return None
        return cls(x_left=d["x_left"], y_top=d["y_top"], x_right=d["x_right"], y_bottom=d["y_bottom"])


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
    token_count: int
    bbox: BBox | None                                 # PDF only
    link_url: list[str]                              # unique URLs found in chunk
    table_ids: list[str]                             # table IDs referenced in chunk
    chart_ids: list[str] = field(default_factory=list)  # chart IDs referenced in chunk (docx/pptx)
    extra: dict = field(default_factory=dict)        # caller-requested extra fields from the pipeline df

    @classmethod
    def from_dict(cls, d: dict) -> "Chunk":
        return cls(
            id=d["id"],
            parent_chunk_id=d.get("parent_chunk_id"),
            chunk_index=d.get("chunk_index", 0),
            page_number=d.get("page_number", 0),
            page_label=d.get("page_label"),
            section=d.get("section", ""),
            heading=d.get("heading"),
            path=d.get("path", []),
            text=d.get("text", ""),
            char_count=d.get("char_count", 0),
            token_count=d.get("token_count", 0),
            bbox=BBox.from_dict(d.get("bbox")),
            link_url=d.get("link_url", []),
            table_ids=[_norm_id(v) for v in d.get("table_ids", [])],
            chart_ids=[_norm_id(v) for v in d.get("chart_ids", [])],
            extra=d.get("extra", {}),
        )

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Block:
    id: str
    type: str                                        # paragraph | heading | table | toc | exhibits | navigation | …
    page_number: int
    page_label: str | None                           # e.g. "A-6", "iv" — distinct from page_number
    section: str                                     # body | toc | exhibit | header | footer | coverpage
    text: str
    chunk_id: str | None                             # which chunk this block belongs to
    char_count: int
    bbox: BBox | None                                 # PDF only
    link_url: list[str]                              # unique URLs found in block
    table_ids: list[str]                             # table IDs referenced in block
    chart_ids: list[str] = field(default_factory=list)  # chart IDs referenced in block (docx/pptx)
    extra: dict = field(default_factory=dict)        # caller-requested extra fields from the pipeline df

    @classmethod
    def from_dict(cls, d: dict) -> "Block":
        return cls(
            id=d["id"],
            type=d.get("type", ""),
            page_number=d.get("page_number", 0),
            page_label=d.get("page_label"),
            section=d.get("section", ""),
            text=d.get("text", ""),
            chunk_id=d.get("chunk_id"),
            char_count=d.get("char_count", 0),
            bbox=BBox.from_dict(d.get("bbox")),
            link_url=d.get("link_url", []),
            table_ids=[_norm_id(v) for v in d.get("table_ids", [])],
            chart_ids=[_norm_id(v) for v in d.get("chart_ids", [])],
            extra=d.get("extra", {}),
        )

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

    @classmethod
    def from_dict(cls, d: dict) -> "TableCell":
        return cls(
            row=d.get("row", 0),
            col=d.get("col", 0),
            rowspan=d.get("rowspan", 1),
            colspan=d.get("colspan", 1),
            role=d.get("role", ""),
            text=d.get("text", ""),
            bbox=BBox.from_dict(d.get("bbox")),
        )

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

    @classmethod
    def from_dict(cls, d: dict) -> "Table":
        return cls(
            id=d["id"],
            caption=d.get("caption"),
            page_number=d.get("page_number", 0),
            page_label=d.get("page_label"),
            chunk_id=d.get("chunk_id", ""),
            bbox=BBox.from_dict(d.get("bbox")),
            markdown=d.get("markdown", ""),
            cells=[TableCell.from_dict(c) for c in d.get("cells", [])],
        )

    def to_dataframe(self) -> "pd.DataFrame":
        import pandas as pd
        return pd.DataFrame([asdict(c) for c in self.cells])

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ChartPoint:
    series_index: int                                 # 0-indexed series position
    series_name: str | None                           # e.g. "Revenue 2024"
    point_index: int                                  # 0-indexed point position within the series
    category: str | None                              # category-axis label, e.g. "Q1"
    label: str | None                                 # explicit data label attached to the point
    value: float | None                               # plotted value (None for non-numeric caches)
    x_value: float | None                             # scatter/bubble charts only
    y_value: float | None                             # scatter/bubble charts only
    bubble_size: float | None                         # bubble charts only
    percent: float | None                             # share of series total; pie/doughnut only

    @classmethod
    def from_dict(cls, d: dict) -> "ChartPoint":
        return cls(
            series_index=d.get("series_index", 0),
            series_name=d.get("series_name"),
            point_index=d.get("point_index", 0),
            category=d.get("category"),
            label=d.get("label"),
            value=d.get("value"),
            x_value=d.get("x_value"),
            y_value=d.get("y_value"),
            bubble_size=d.get("bubble_size"),
            percent=d.get("percent"),
        )

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Chart:
    id: str
    chart_type: str                                   # barChart | lineChart | pieChart | scatterChart | …
    title: str | None
    axis_x_title: str | None
    axis_y_title: str | None
    page_number: int
    page_label: str | None
    chunk_id: str
    is_stacked: bool
    bbox: BBox | None                                 # PPTX only — shape geometry on the slide
    markdown: str                                     # convenience — chart data as a markdown table
    points: list[ChartPoint]

    @classmethod
    def from_dict(cls, d: dict) -> "Chart":
        return cls(
            id=d["id"],
            chart_type=d.get("chart_type", ""),
            title=d.get("title"),
            axis_x_title=d.get("axis_x_title"),
            axis_y_title=d.get("axis_y_title"),
            page_number=d.get("page_number", 0),
            page_label=d.get("page_label"),
            chunk_id=d.get("chunk_id", ""),
            is_stacked=d.get("is_stacked", False),
            bbox=BBox.from_dict(d.get("bbox")),
            markdown=d.get("markdown", ""),
            points=[ChartPoint.from_dict(p) for p in d.get("points", [])],
        )

    @property
    def series_names(self) -> list[str]:
        """Unique series names in plot order (None-named series excluded)."""
        seen: dict[str, None] = {}
        for p in sorted(self.points, key=lambda p: p.series_index):
            if p.series_name is not None:
                seen.setdefault(p.series_name, None)
        return list(seen)

    def to_dataframe(self) -> "pd.DataFrame":
        import pandas as pd
        return pd.DataFrame([asdict(p) for p in self.points])

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
    path: list[str] = field(default_factory=list)  # ancestor texts; populated by .level() and .find_heading()
    block_ids: list[str] = field(default_factory=list)  # all block ids under this heading section

    @classmethod
    def from_dict(cls, d: dict) -> "HierarchyNode":
        return cls(
            heading_id=d["heading_id"],
            text=d.get("text", ""),
            level=d.get("level", 1),
            heading_type=d.get("heading_type", "free_form"),
            page_number=d.get("page_number"),
            page_label=d.get("page_label"),
            chunk_ids=d.get("chunk_ids", []),
            children=[cls.from_dict(c) for c in d.get("children", [])],
            path=d.get("path", []),
            block_ids=d.get("block_ids", []),
        )

    def to_dict(self, minimal: bool = False) -> dict:
        if minimal:
            d: dict = {"text": self.text}
            if self.children:
                d["children"] = [c.to_dict(minimal=True) for c in self.children]
            return d
        d = {
            "heading_id": self.heading_id,
            "text": self.text,
            "level": self.level,
            "heading_type": self.heading_type,
            "page_number": self.page_number,
            "page_label": self.page_label,
            "chunk_ids": self.chunk_ids,
            "children": [c.to_dict() for c in self.children],
        }
        if self.path:
            d["path"] = self.path
        if self.block_ids:
            d["block_ids"] = self.block_ids
        return d


@dataclass
class HierarchyTree:
    roots: list[HierarchyNode]

    @classmethod
    def from_dict(cls, data: list[dict]) -> "HierarchyTree":
        return cls(roots=[HierarchyNode.from_dict(n) for n in data])

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
        data = self.to_dict(minimal=minimal)
        text = _format_hierarchy_json(data) if minimal else json.dumps(data, indent=indent)
        Path(path).write_text(text, encoding="utf-8")

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

    def _locate(self, heading_id: int) -> tuple[HierarchyNode | None, list[str], int]:
        """Return (node, ancestor_path, depth) for the given heading_id, or (None, [], 0)."""
        def _search(node: HierarchyNode, ancestors: list[str], depth: int):
            if node.heading_id == heading_id:
                return node, ancestors, depth
            for child in node.children:
                result = _search(child, ancestors + [node.text], depth + 1)
                if result[0] is not None:
                    return result
            return None, [], 0
        for root in self.roots:
            result = _search(root, [], 1)
            if result[0] is not None:
                return result
        return None, [], 0

    def level(self, n: int, parent: HierarchyNode | int | None = None) -> list[HierarchyNode]:
        """Return all nodes at depth n with their ancestor path populated.

        Args:
            n: 1-based depth (1 = top-level headings).
            parent: Optional HierarchyNode or heading_id to restrict results to
                descendants of that node.
        """
        results: list[HierarchyNode] = []

        def _collect(node: HierarchyNode, ancestors: list[str], depth: int) -> None:
            if depth == n:
                results.append(replace(node, path=list(ancestors)))
                return
            if depth > n:
                return
            for child in node.children:
                _collect(child, ancestors + [node.text], depth + 1)

        if parent is None:
            for root in self.roots:
                _collect(root, [], 1)
        else:
            parent_id = parent.heading_id if isinstance(parent, HierarchyNode) else int(parent)
            parent_node, parent_ancestors, parent_depth = self._locate(parent_id)
            if parent_node is None:
                return []
            for child in parent_node.children:
                _collect(child, parent_ancestors + [parent_node.text], parent_depth + 1)

        return results

    def find_heading(self, text: str, case_sensitive: bool = False) -> list[HierarchyNode]:
        """Return all nodes whose text contains ``text``, with ancestor path populated.

        Args:
            text: Substring to search for.
            case_sensitive: If False (default), search is case-insensitive.
        """
        results: list[HierarchyNode] = []
        needle = text if case_sensitive else text.lower()

        def _visit(node: HierarchyNode, ancestors: list[str]) -> None:
            haystack = node.text if case_sensitive else node.text.lower()
            if needle in haystack:
                results.append(replace(node, path=list(ancestors)))
            for child in node.children:
                _visit(child, ancestors + [node.text])

        for root in self.roots:
            _visit(root, [])

        return results


def _collect_chunk_ids(node: HierarchyNode, recursive: bool) -> set[str]:
    ids = set(node.chunk_ids)
    if recursive:
        for child in node.children:
            ids |= _collect_chunk_ids(child, recursive)
    return ids


def _collect_block_ids(node: HierarchyNode, recursive: bool) -> set[str]:
    ids = set(node.block_ids)
    if recursive:
        for child in node.children:
            ids |= _collect_block_ids(child, recursive)
    return ids


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


def _prettify_embedded_tables(text: str) -> str:
    """Prettify pipe-table sections within mixed text, passing other lines through.

    Chart blocks may carry a title line before the table, or a non-table
    representation (jsonl/melted) with no pipe lines at all — only contiguous
    runs of pipe lines are realigned.
    """
    out: list[str] = []
    buf: list[str] = []

    def _flush() -> None:
        if buf:
            out.append(_prettify_table("\n".join(buf)))
            buf.clear()

    for line in text.splitlines():
        if line.strip().startswith("|"):
            buf.append(line)
        else:
            _flush()
            out.append(line)
    _flush()
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
    charts: list[Chart] = field(default_factory=list)  # docx/pptx only for now
    hierarchy: HierarchyTree = field(default_factory=lambda: HierarchyTree(roots=[]))
    pipeline_steps: dict[str, "pd.DataFrame"] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict) -> "ParseResult":
        """Reconstruct a ParseResult from a dict produced by to_dict()."""
        return cls(
            chunks=[Chunk.from_dict(c) for c in d.get("chunks", [])],
            blocks=[Block.from_dict(b) for b in d.get("blocks", [])],
            tables=[Table.from_dict(t) for t in d.get("tables", [])],
            metadata=DocumentMetadata.from_dict(d.get("metadata", {})),
            charts=[Chart.from_dict(c) for c in d.get("charts", [])],
            hierarchy=HierarchyTree.from_dict(d.get("hierarchy", [])),
        )

    @classmethod
    def load(cls, path: str | Path) -> "ParseResult":
        """Load a ParseResult from a JSON file saved with .save() or .to_json()."""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(data)

    @property
    def text(self) -> str:
        """Plain text of the document body, excluding headers, footers, and TOC."""
        return self.export_to_text()

    def to_dict(self) -> dict:
        return {
            "schema": "DocSlicerResult",
            "version": _SCHEMA_VERSION,
            "metadata": self.metadata.to_dict(),
            "chunks": [c.to_dict() for c in self.chunks],
            "blocks": [b.to_dict() for b in self.blocks],
            "tables": [t.to_dict() for t in self.tables],
            "charts": [c.to_dict() for c in self.charts],
            "hierarchy": self.hierarchy.to_dict(),
        }

    def to_json(self, indent: int = 2) -> str:
        """Return the full parse result as a JSON string."""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    def export_to_dict(self) -> dict:
        return self.to_dict()

    def export_to_markdown(
        self,
        include_page_markers: bool = True,
        include_tables: bool = True,
        include_toc: bool = True,
        include_furniture: bool = True,
        prettify: bool = True,
    ) -> str:
        """Render the document as Markdown using blocks as the source of truth."""
        _HEADING_ROLES = {
            "heading", "toc_heading", "exhibit_heading", "hybrid_heading_paragraph",
        }
        _FURNITURE_ROLES = {"navigation", "suppressed_repeated_heading", "page_label", "vertical_text", "hr"}
        excluded_sections: set[str] = set()
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
            if not include_furniture and block.type in _FURNITURE_ROLES:
                continue

            text = block.text.strip()

            if include_page_markers and block.page_number != prev_page:
                prev_page = block.page_number
                label = block.page_label or str(block.page_number)
                parts.append(f"<!-- page {label} -->")

            if block.type in _HEADING_ROLES:
                level = level_map.get(text, 2)
                prefix = "#" * max(1, min(6, level))
                parts.append(f"{prefix} {text}")
            elif block.type == "table" or (block.type in ("toc", "toc_heading") and block.table_ids):
                if include_tables:
                    table = tables_by_id.get(block.table_ids[0]) if block.table_ids else None
                    raw = table.markdown if table else text
                    parts.append(_prettify_table(raw) if prettify else raw)
            elif block.type == "chart":
                if text:
                    parts.append(_prettify_embedded_tables(text) if prettify else text)
            elif text:
                parts.append(text)

        return "\n\n".join(parts)

    def export_to_text(
        self,
        include_tables: bool = True,
        include_toc: bool = False,
        include_furniture: bool = True,
    ) -> str:
        """Render the document as plain text (no Markdown formatting)."""
        _HEADING_ROLES = {
            "heading", "toc_heading", "exhibit_heading", "hybrid_heading_paragraph",
        }
        _FURNITURE_ROLES = {"navigation", "suppressed_repeated_heading", "page_label", "vertical_text", "hr"}
        excluded_sections: set[str] = set()
        if not include_toc:
            excluded_sections.add("toc")

        tables_by_id = {t.id: t for t in self.tables}

        parts: list[str] = []

        for block in self.blocks:
            if block.section in excluded_sections:
                continue
            if not include_furniture and block.type in _FURNITURE_ROLES:
                continue

            text = block.text.strip()

            if block.type in _HEADING_ROLES:
                if text:
                    parts.append(text)
            elif block.type == "table" or (block.type in ("toc", "toc_heading") and block.table_ids):
                if include_tables:
                    table = tables_by_id.get(block.table_ids[0]) if block.table_ids else None
                    parts.append(table.markdown if table else text)
            elif text:
                parts.append(text)

        return "\n\n".join(parts)

    def export_toc(self) -> str:
        """Return the document's table of contents as a Markdown string.

        Renders blocks with type ``toc_heading`` or ``toc`` in document order.
        Returns an empty string when no TOC was detected.
        """
        tables_by_id = {t.id: t for t in self.tables}
        parts: list[str] = []
        for block in self.blocks:
            if block.type == "toc_heading" and not block.table_ids:
                text = block.text.strip()
                if text:
                    parts.append(f"# {text}")
            elif block.type in ("toc", "toc_heading"):
                if block.table_ids:
                    table = tables_by_id.get(block.table_ids[0])
                    raw = table.markdown if table else block.text.strip()
                    parts.append(_prettify_table(raw))
                else:
                    text = block.text.strip()
                    if text:
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

    def export_charts_csv(self, path: str | Path, encoding: str = "utf-8") -> None:
        """Save all chart datapoints as one flat CSV (one row per point).

        Pass encoding="utf-8-sig" for Excel-friendly output (adds BOM).
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.charts_df().to_csv(path, index=False, encoding=encoding)

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

    def charts_df(self) -> "pd.DataFrame":
        """One row per chart datapoint, with chart-level metadata repeated."""
        import pandas as pd
        rows = []
        for chart in self.charts:
            for point in chart.points:
                rows.append({
                    "chart_id": chart.id,
                    "chart_type": chart.chart_type,
                    "title": chart.title,
                    "page_number": chart.page_number,
                    "page_label": chart.page_label,
                    **point.to_dict(),
                })
        return pd.DataFrame(rows)

    # ── Hierarchy navigation ──────────────────────────────────────────────────

    def find_heading(self, text: str, case_sensitive: bool = False) -> list[HierarchyNode]:
        """Return all hierarchy nodes whose text contains ``text``."""
        return self.hierarchy.find_heading(text, case_sensitive=case_sensitive)

    def _resolve_heading(self, heading: HierarchyNode | int) -> HierarchyNode | None:
        if isinstance(heading, HierarchyNode):
            return heading
        for node in self.hierarchy.flatten():
            if node.heading_id == heading:
                return node
        return None

    def chunks_under(self, heading: HierarchyNode | int, recursive: bool = True) -> list[Chunk]:
        """Return all chunks under a heading.

        Args:
            heading: A HierarchyNode or heading_id int.
            recursive: Include chunks from descendant headings (default True).
        """
        node = self._resolve_heading(heading)
        if node is None:
            return []
        chunk_ids = _collect_chunk_ids(node, recursive)
        return [c for c in self.chunks if c.id in chunk_ids]

    def blocks_under(self, heading: HierarchyNode | int, recursive: bool = True) -> list[Block]:
        """Return all blocks under a heading.

        Prefers direct block_ids when available (exact). Falls back to the
        page-range approximation derived from chunks when block_ids are absent.

        Args:
            heading: A HierarchyNode or heading_id int.
            recursive: Include blocks from descendant headings (default True).
        """
        node = self._resolve_heading(heading)
        if node is None:
            return []

        # Prefer direct block_ids (populated by _build_hierarchy)
        block_ids = _collect_block_ids(node, recursive)
        if block_ids:
            return [b for b in self.blocks if b.id in block_ids]

        # Fall back to page-range approximation via chunks
        relevant_chunks = self.chunks_under(node, recursive=recursive)
        if not relevant_chunks:
            return []
        pages = {c.page_number for c in relevant_chunks}
        sections = {c.section for c in relevant_chunks}
        return [b for b in self.blocks if b.page_number in pages and b.section in sections]

    def tables_under(self, heading: HierarchyNode | int, recursive: bool = True) -> list[Table]:
        """Return all tables referenced under a heading.

        Uses chunk.table_ids when chunks are available; falls back to
        block.table_ids when chunking was disabled.

        Args:
            heading: A HierarchyNode or heading_id int.
            recursive: Include tables from descendant headings (default True).
        """
        node = self._resolve_heading(heading)
        if node is None:
            return []
        relevant_chunks = self.chunks_under(node, recursive=recursive)
        if relevant_chunks:
            table_ids = {tid for c in relevant_chunks for tid in c.table_ids}
        else:
            relevant_blocks = self.blocks_under(node, recursive=recursive)
            table_ids = {tid for b in relevant_blocks for tid in b.table_ids}
        if not table_ids:
            return []
        return [t for t in self.tables if t.id in table_ids]

    def charts_under(self, heading: HierarchyNode | int, recursive: bool = True) -> list[Chart]:
        """Return all charts referenced under a heading.

        Uses chunk.chart_ids when chunks are available; falls back to
        block.chart_ids when chunking was disabled.

        Args:
            heading: A HierarchyNode or heading_id int.
            recursive: Include charts from descendant headings (default True).
        """
        node = self._resolve_heading(heading)
        if node is None:
            return []
        relevant_chunks = self.chunks_under(node, recursive=recursive)
        if relevant_chunks:
            chart_ids = {cid for c in relevant_chunks for cid in c.chart_ids}
        else:
            relevant_blocks = self.blocks_under(node, recursive=recursive)
            chart_ids = {cid for b in relevant_blocks for cid in b.chart_ids}
        if not chart_ids:
            return []
        return [c for c in self.charts if c.id in chart_ids]

    # ── Page navigation ───────────────────────────────────────────────────────

    def chunks_by_page(self, page: int | str) -> list[Chunk]:
        """Return chunks on a given page. Pass int for page_number, str for page_label."""
        if isinstance(page, str):
            return [c for c in self.chunks if c.page_label == page]
        return [c for c in self.chunks if c.page_number == page]

    def blocks_by_page(self, page: int | str) -> list[Block]:
        """Return blocks on a given page. Pass int for page_number, str for page_label."""
        if isinstance(page, str):
            return [b for b in self.blocks if b.page_label == page]
        return [b for b in self.blocks if b.page_number == page]

    def tables_by_page(self, page: int | str) -> list[Table]:
        """Return tables on a given page. Pass int for page_number, str for page_label."""
        if isinstance(page, str):
            return [t for t in self.tables if t.page_label == page]
        return [t for t in self.tables if t.page_number == page]

    def charts_by_page(self, page: int | str) -> list[Chart]:
        """Return charts on a given page. Pass int for page_number, str for page_label."""
        if isinstance(page, str):
            return [c for c in self.charts if c.page_label == page]
        return [c for c in self.charts if c.page_number == page]

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

        Supported stems: chunks, blocks, tables, charts, metadata.
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
            if self.charts:
                _to_parquet(self.charts_df(), path / "charts.parquet")
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
            "charts": lambda: [c.to_dict() for c in self.charts],
        }

        if stem == "metadata":
            if suffix != ".json":
                raise ValueError("metadata can only be saved as .json")
            path.write_text(json.dumps(self.metadata.to_dict(), indent=2, ensure_ascii=False))
            return

        # Unknown stem (e.g. "result") → full document export
        if stem not in level_map:
            if suffix != ".json":
                raise ValueError(f"Unknown stem {stem!r}: use chunks/blocks/tables/charts/metadata, or a .json path for a full export")
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
