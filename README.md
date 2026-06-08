# DocSlicer

Deterministic hierarchical document parser and chunker for business documents. No LLM calls.

DocSlicer turns PDFs, Word documents, HTML pages, and PowerPoint files into clean chunks, structured blocks, tables, and a navigable heading hierarchy — preserving the document's own structure instead of guessing at it.

```python
import docslicer

result = docslicer.parse_document("annual_report.pdf")

# Inspect the outline first
result.hierarchy.to_outline()
# - Executive Summary
# - Risk Factors
#   - Market Risk
#   - Credit Risk
#   - Liquidity Risk
# - Financial Statements
#   - Consolidated Balance Sheet
#   - Notes to Financial Statements

# Pull only the chunks you need
risk_section = result.find_heading("Risk Factors")[0]
chunks = result.chunks_under(risk_section)

# Tables come back structured, not as flat text
for table in result.tables_under(risk_section):
    print(table.markdown)
```

---

## Install

```bash
pip install docslicer
```

HTML parsing and OCR are optional extras:

```bash
pip install 'docslicer[html]'    # HTML / URL parsing via Playwright
playwright install                # one-time browser install

pip install 'docslicer[ocr]'     # scanned PDF support via Tesseract
# also requires: apt install tesseract-ocr  (or brew install tesseract)

pip install 'docslicer[parquet]' # Parquet export support
```

**Requires Python 3.10+**

---

## What you get back

`parse_document` returns a `ParseResult`:

```python
result.chunks      # list[Chunk]   — heading-aware text chunks, ready for embedding
result.blocks      # list[Block]   — paragraph/heading/table blocks before chunking
result.tables      # list[Table]   — structured tables with cells, spans, and markdown
result.metadata    # DocumentMetadata — title, author, language, page count, OCR flag
result.hierarchy   # HierarchyTree — navigable tree of all headings
```

Each `Chunk` carries:

```python
chunk.text          # str   — chunk text
chunk.path          # list  — ["## Risk Factors", "### Market Risk"]
chunk.heading       # str   — nearest heading above this chunk
chunk.section       # str   — body | toc | exhibit | header | footer | coverpage | …
chunk.page_number   # int   — 1-based physical page
chunk.page_label    # str   — "A-6", "iv", "F-3" — as printed on the page
chunk.table_ids     # list  — IDs of tables referenced in this chunk
chunk.link_url      # list  — URLs found in this chunk
chunk.bbox          # BBox  — bounding box (PDF only)
```

---

## Supported formats

| Format | Extension | Notes |
|--------|-----------|-------|
| PDF | `.pdf` | Text-based and scanned (OCR extra required for scanned) |
| Word | `.docx` | Full style and outline hierarchy |
| HTML | `.html`, URLs | Static files and JS-rendered pages (html extra required for URLs) |
| PowerPoint | `.pptx` | Slides, speaker notes, charts |

Not supported: `.doc`, `.ppt` (legacy Office formats), `.xlsx`.

---

## Usage

### Parse a document

`parse_document` auto-detects the format from the file extension or magic bytes. Pass a file path, URL, raw `bytes`, or a file-like object:

```python
result = docslicer.parse_document("contract.docx")
result = docslicer.parse_document("report.pdf")
result = docslicer.parse_document("https://www.sec.gov/Archives/edgar/data/.../10-K.htm")
result = docslicer.parse_document(file_bytes)
```

### Chunking options

```python
result = docslicer.parse_document(
    "report.pdf",
    max_chunk_size=2000,          # hard cap, default 3200
    optimal_chunk_size=800,       # target size, default 1500
    min_chunk_size=400,           # soft floor, default 700
    chunking=False,               # skip chunking, return blocks only (faster)
    merge_small_chunks=True,      # merge chunks below min_chunk_size (default True)
    table_representation="jsonl", # "markdown" (default) | "jsonl" | "melted"
)
```

### Table representation formats

`table_representation` controls how tables are serialised into chunk text. Given a
financial table with multi-row column headers:

**`"markdown"` (default)** — preserves the original 2D layout:

```
|           | Three Months Ended    | Three Months Ended    |
|           | December 27, 2025     | December 28, 2024     |
|-----------|----------------------:|----------------------:|
| iPhone ®  |              $85,269  |              $69,138  |
| Mac ®     |               8,386   |               8,987   |
| iPad ®    |               8,595   |               8,088   |
| …         |                   …   |                   …   |
```

**`"melted"`** — one row per cell, headers joined with ` > `. Good for sparse or
pivot-style tables where individual cell retrieval matters:

```
iPhone ® | Three Months Ended > December 27, 2025 | $85,269
iPhone ® | Three Months Ended > December 28, 2024 | $69,138
Mac ® | Three Months Ended > December 27, 2025 | 8,386
Mac ® | Three Months Ended > December 28, 2024 | 8,987
iPad ® | Three Months Ended > December 27, 2025 | 8,595
iPad ® | Three Months Ended > December 28, 2024 | 8,088
…
```

**`"jsonl"`** — one JSON object per row, multi-row headers joined with `_`. Useful
when chunks are fed into structured extraction or tool-use pipelines:

```jsonl
{"Metric": "iPhone ®", "Three Months Ended_December 27, 2025": "$85,269", "Three Months Ended_December 28, 2024": "$69,138"}
{"Metric": "Mac ®", "Three Months Ended_December 27, 2025": "8,386", "Three Months Ended_December 28, 2024": "8,987"}
{"Metric": "iPad ®", "Three Months Ended_December 27, 2025": "8,595", "Three Months Ended_December 28, 2024": "8,088"}
…
```

### Navigate by heading

```python
# Find a heading
nodes = result.find_heading("Financial Statements")
node = nodes[0]

# Retrieve content under it (recursive by default)
chunks = result.chunks_under(node)
blocks = result.blocks_under(node)
tables = result.tables_under(node)

# Walk the outline
for node in result.hierarchy.level(1):
    print(node.text, "→", len(result.chunks_under(node)), "chunks")
```

### Navigate by page

```python
result.chunks_by_page(14)        # by page number
result.chunks_by_page("F-3")     # by printed page label
result.blocks_by_page(14)
result.tables_by_page(14)
```

### Batch processing

```python
for path, result in docslicer.parse_all("documents/", recursive=True):
    if isinstance(result, Exception):
        print(f"Failed {path}: {result}")
    else:
        print(f"{path}: {len(result.chunks)} chunks")
```

### Export

```python
# Save everything
result.save("output/")
# → output/chunks.parquet, blocks.parquet, tables.parquet, metadata.json

# Specific formats
result.save("chunks.csv")
result.save("result.json")        # full parse result as JSON
result.export_chunks_jsonl("chunks.jsonl")

# Render as Markdown or plain text
md = result.export_to_markdown(include_tables=True)
txt = result.export_to_text()

# DataFrames
df = result.chunks_df()
```

### Reuse config across documents

```python
from docslicer import DocumentParser, ParseConfig

parser = DocumentParser(ParseConfig(max_chunk_size=1500, optimal_chunk_size=600))

for path in paths:
    result = parser.parse(path)
```

### Debug mode

```python
result = docslicer.parse_document("report.pdf", debug=True)
result.save_debug("debug_steps/")
# → one CSV per pipeline step, showing intermediate DataFrames
```

---

## OCR

`parse_document` automatically detects scanned pages and falls back to OCR when the
`[ocr]` extra is installed. No configuration needed — `result.metadata.has_ocr`
tells you whether OCR was used.

```bash
pip install 'docslicer[ocr]'
# Linux:  apt install tesseract-ocr
# macOS:  brew install tesseract
```

---

## How chunking works

DocSlicer chunks at heading and paragraph boundaries — never in the middle of a
sentence, never with character overlap. Each chunk tracks its full heading path
(`chunk.path`) so downstream retrieval can filter by section without re-parsing.

Heading hierarchy is extracted deterministically from font size, bold weight,
numbering patterns, and document structure — not from LLM inference. The algorithm
handles nested numbered sections (`1.`, `1.2.`, `1.2.3`), re-entry after exhibit
breaks, and repeated navigation headings across pages.

---

## Format-specific functions

If you know the format upfront and want explicit failure on unexpected input, use the
format-specific variants. They accept the same arguments as `parse_document`:

```python
docslicer.parse_pdf("report.pdf")
docslicer.parse_docx("contract.docx")
docslicer.parse_pptx("deck.pptx")
docslicer.parse_html("filing.html")
```

---

## License

MIT
