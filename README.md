# DocSlicer

Deterministic hierarchical document parser and chunker for business documents. No LLM calls.

DocSlicer turns PDFs, Word documents, HTML pages, and PowerPoint files into clean chunks, structured blocks, tables, and a navigable heading hierarchy — preserving the document's own structure instead of guessing at it.

```python
import docslicer

result = docslicer.parse_document("annual_report.pdf")

# Inspect the outline first
result.hierarchy.to_outline()
# - PART I — FINANCIAL INFORMATION
#   - Item 1. Financial Statements
#     - Notes to Condensed Consolidated Financial Statements
#       - Note 4 – Financial Instruments
#         - Derivative Instruments and Hedging
#           - Foreign Exchange Rate Risk
#           - Interest Rate Risk
#         - Accounts Receivable
#           - Trade Receivables
#   - Item 2. Management's Discussion and Analysis
#     - Liquidity and Capital Resources
# - PART II — OTHER INFORMATION
#   ...

# Pull only the chunks you need
risk_section = result.find_heading("Risk Factors")[0]
chunks = result.chunks_under(risk_section)

# Tables come back structured, not as flat text
for table in result.tables_under(risk_section):
    print(table.markdown)
```

---

## Features

- **No LLM, VLM, or ML models** — fully deterministic; no model weights to download, no GPU required, no cold-start delay
- **Lightweight** — ~1.5 MB wheel with no heavy ML dependencies
- **Deep hierarchy extraction** — works for both numbered (`1.`, `1.2.`, `1.2.3`) and free-form headings; uses font size, bold weight, and document structure — not inference; handles re-entry after exhibit breaks and repeated navigation headings across pages
- **Structure-aware chunking** — splits at heading and paragraph boundaries, preserving semantic coherence
- **Zero character overlap** — chunks are non-overlapping by default; no duplicated tokens in your context window
- **Agentic-friendly** — reduces token spend on long documents: have the agent inspect the outline first, then pull only the relevant chunks into context instead of feeding a 500-page document verbatim; well-suited for legal texts, technical SOPs, financial filings, and compliance documents
- **Unified result object** — `chunks`, `blocks`, `tables`, `metadata`, and `hierarchy` in one place
- **Structured tables** — tables come back as cells, not flat text; export as Markdown, JSONL, or melted format
- **Multiple export formats** — CSV, Markdown, JSONL, Parquet, JSON, plain text, and DataFrames
- **Reading order preserved** — including multi-column PDF layouts
- **Supports `pdf`, `docx`, `pptx`, and `html`** — including JS-rendered pages via Playwright
- **Robust URL fetching** — always renders pages in a real browser, handling cookie banners and bot protection out of the box; also preserves styling signals like boldness that raw HTML omits, producing sharper heading detection and chunk quality
- **OCR fallback** — auto-detects scanned pages and falls back to Tesseract when the extra is installed

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

## What you get back (ParseResult)

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
chunk.path          # list  — full heading breadcrumb from root to nearest heading
chunk.heading       # str   — nearest heading above this chunk
chunk.section       # str   — body | toc | exhibit | header | footer | coverpage | …
chunk.page_number   # int   — 1-based physical page
chunk.page_label    # str   — "A-6", "iv", "F-3" — as printed on the page
chunk.table_ids     # list  — IDs of tables referenced in this chunk
chunk.link_url      # list  — URLs found in this chunk
chunk.bbox          # BBox  — bounding box (PDF only)
```

Every chunk carries its full heading breadcrumb, no matter how deeply nested. For example, a paragraph six levels deep in a financial filing:

```python
chunk.path == [
    "# PART I — FINANCIAL INFORMATION",
    "## Item 1. Financial Statements",
    "### Notes to Condensed Consolidated Financial Statements (Unaudited)",
    "#### Note 4 – Financial Instruments",
    "##### Accounts Receivable",
    "###### Trade Receivables",
]
```

This lets downstream code filter or group chunks by any level of the hierarchy without re-parsing the document.

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

## Parsing

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
    extra_fields=["is_bold", "font_size", "font_name"],  # surface internal pipeline columns on each chunk/block via .extra
)
```

### How merge_small_chunks works

Because DocSlicer is structure-aware, it initially produces one chunk per heading or paragraph boundary. For documents with many short sections this can yield a lot of small chunks. With `merge_small_chunks=True` (the default), sibling sections under the same parent heading are merged together until they reach `min_chunk_size` — but never across heading boundaries into a different parent.

For example, these five short sections all fall under `## Products and Services Performance`:

```
### Mac          → "Mac net sales decreased …"            (~120 chars)
### iPad         → "iPad net sales increased …"           (~180 chars)
### Wearables    → "Wearables net sales decreased …"      (~130 chars)
### Services     → "Services net sales increased …"       (~160 chars)
```

Instead of four tiny chunks, they get merged into one coherent chunk that still carries the correct `path` for each paragraph. Set `merge_small_chunks=False` if you need one chunk per section regardless of size.

---

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

### Batch processing

```python
for path, result in docslicer.parse_all("documents/", recursive=True):
    if isinstance(result, Exception):
        print(f"Failed {path}: {result}")
    else:
        print(f"{path}: {len(result.chunks)} chunks")
```

### Reuse config across documents

```python
from docslicer import DocumentParser, ParseConfig

parser = DocumentParser(ParseConfig(max_chunk_size=1500, optimal_chunk_size=600))

for path in paths:
    result = parser.parse(path)
```

---

## Navigating the hierarchy

Most chunking libraries give you a flat list of text segments. DocSlicer also gives you a navigable tree of the document's heading structure, extracted deterministically from the document itself.

This is particularly useful for agents and retrieval pipelines working with long documents: rather than feeding the entire document into context, the agent can inspect the outline first to understand the structure, decide which sections are relevant, and then pull only those chunks — keeping token usage proportional to the task.

### Inspect the outline

```python
# Print the full heading tree
result.hierarchy.to_outline()

# Walk all top-level sections and see how much content each contains
for node in result.hierarchy.level(1):
    print(node.text, "→", len(result.chunks_under(node)), "chunks")
```

### Drill into a section

`.level(n)` returns all headings at depth `n`. Pass a `parent` to scope it to a
specific subtree — the typical pattern for an agent navigating a long document:

```python
# All top-level headings
l1 = result.hierarchy.level(1)

# Pick one, then list its subsections
section = result.find_heading("Financial Statements")[0]
for node in result.hierarchy.level(2, parent=section):
    print(node.text, f"(p.{node.page_number})")

# Drill one level deeper
subsection = result.hierarchy.level(2, parent=section)[0]
for node in result.hierarchy.level(3, parent=subsection):
    print(node.text)
```

### Retrieve content under a heading

`find_heading` matches any node whose text contains the search term (case-insensitive).
All retrieval methods recurse into subsections by default.

```python
node = result.find_heading("Financial Instruments")[0]

chunks = result.chunks_under(node)              # text chunks, ready for embedding or prompting
chunks = result.chunks_under(node, recursive=False)  # direct heading only, no subsections
tables = result.tables_under(node)              # structured tables in this section
blocks = result.blocks_under(node)              # raw paragraph/heading blocks
```

### Navigate by page

```python
result.chunks_by_page(14)        # by page number
result.chunks_by_page("F-3")     # by printed page label
result.blocks_by_page(14)
result.tables_by_page(14)
```

---

## Export

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

### Debug mode

```python
result = docslicer.parse_document("report.pdf", debug=True)

# result.pipeline_steps is an ordered dict of step name → DataFrame
for name, df in result.pipeline_steps.items():
    print(name, df.shape)
    df.to_csv(f"debug/{name}.csv", index=False)

# PDF steps:  words → shapes → cells → lines → table_cells → blocks → chunks
# DOCX steps: runs → paragraphs → lines → table_cells → blocks → chunks
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

Apache 2.0
