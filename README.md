# DocSlicer

[![PyPI](https://img.shields.io/pypi/v/docslicer.svg)](https://pypi.org/project/docslicer/) [![License: AGPL v3](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](LICENSE) [![Commercial license available](https://img.shields.io/badge/License-Commercial-green.svg)](LICENSE-COMMERCIAL.md)

Lightning-fast (31 pages/sec), deterministic document parser and chunker for business documents. No LLM calls or heavy ML models.

DocSlicer turns PDFs, Word documents, HTML pages, and PowerPoint files into clean chunks, structured blocks, tables, charts, markdown and a navigable heading hierarchy.

Top score on [BizDocBench](https://github.com/DocSlicer/BizDocBench) (0.88 overall vs 0.70 for the next-best tool). 0.80 table accuracy, 0.98 content faithfulness, 0.85 heading recognition and hierarchy preservation, and 0.76 RAG retrieval performance.

**Add DocSlicer to your AI pipeline:**
- Classic RAG: the layout-aware chunker gives you clean non-overlapping chunks, each carrying its full heading breadcrumb, ready to embed.
- Vectorless RAG: for when you want an answer out of a document right now. The agent pulls the outline, picks the section it needs, and navigates to the correct section of text without embedding the whole document

```python
import docslicer

def main():
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

if __name__ == "__main__":
    main()
```

---

## Features

- **No LLM, VLM, or ML models** — fully deterministic; no model weights to download, no GPU required, no cold-start delay
- **Lightweight** — ~630 KB wheel with no heavy ML dependencies
- **Agentic-friendly** — reduces token spend on long documents: have the agent inspect the outline first, then pull only the relevant chunks into context instead of feeding a 500-page document verbatim; well-suited for legal texts, technical SOPs, financial filings, and compliance documents
- **Deep hierarchy extraction** — works for both numbered (`1.`, `1.2.`, `1.2.3`) and free-form headings; uses font size, bold weight, and document structure — not inference; handles re-entry after exhibit breaks and repeated navigation headings across pages
- **Structure-aware chunking** — splits at heading and paragraph boundaries, preserving semantic coherence
- **Zero character overlap** — chunks are non-overlapping by default; no duplicated tokens in your context window
- **Unified result object** — `chunks`, `blocks`, `tables`, `charts`, `metadata`, and `hierarchy` in one place
- **Structured tables** — tables come back as cells, not flat text; export as Markdown, JSONL, or melted format
- **Multiple export formats** — CSV, Markdown, JSONL, Parquet, JSON, plain text, and DataFrames
- **Reading order preserved** — including multi-column PDF layouts
- **Supports `pdf`, `docx`, `pptx`, and `html`** — including JS-rendered pages via Playwright
- **Robust URL fetching** — always renders pages in a real browser, handling cookie banners and bot protection out of the box; also preserves styling signals like boldness that raw HTML omits, producing sharper heading detection and chunk quality
- **OCR fallback** — auto-detects scanned pages and falls back to Tesseract when the extra is installed

---

## Benchmarks

Measured with [BizDocBench](https://github.com/DocSlicer/BizDocBench) — an open benchmark for multi-format business document parsing. All scores are 0–1 (higher is better); `pages_per_sec_aggregate` is throughput across the full corpus.

| Tool | Score | Coverage | Speed | Hierarchy | Faithfulness | Tables | Retrieval | Pages/sec |
|---|---|---|---|---|---|---|---|---|
| **docslicer** | **0.8796** | 1.0000 | 0.8836 | 0.8466 | 0.9824 | 0.8047 | 0.7601 | 31.27 |
| docling | 0.7036 | 1.0000 | 0.3805 | 0.4905 | 0.8927 | 0.7467 | 0.7111 | 3.46 |
| markitdown | 0.5838 | 1.0000 | 0.8513 | 0.0604 | 0.7972 | 0.2584 | 0.5357 | 27.42 |
| unstructured | 0.5798 | 0.9091 | 0.1073 | 0.4327 | 0.9057 | 0.4812 | 0.6430 | 0.52 |
| opendataloader | 0.5359 | 0.5844 | 1.0000 | 0.3853 | 0.6484 | 0.2655 | 0.3317 | 117.26 |
| pymupdf4llm | 0.4519 | 0.5974 | 0.6492 | 0.1089 | 0.6456 | 0.3551 | 0.3552 | 11.84 |
| mineru | 0.4107 | 0.5974 | 0.1353 | 0.4220 | 0.6176 | 0.3012 | 0.3910 | 0.70 |
| marker | 0.3735 | 0.5974 | 0.1598 | 0.1926 | 0.6121 | 0.3012 | 0.3778 | 0.87 |

---

## Install

```bash
pip install docslicer
```

The core install is dependency-light. Optional features are available as extras:

```bash
pip install 'docslicer[html]'    # HTML / URL parsing via Playwright
playwright install chromium       # one-time browser install (Chromium only)

pip install 'docslicer[ocr]'     # scanned PDF support via Tesseract + OpenCV
# The tesserocr wheel bundles libtesseract but NOT the language models,
# so install the Tesseract engine to provide them (docslicer auto-detects the path):
# Linux:  apt install tesseract-ocr
# macOS:  brew install tesseract

pip install 'docslicer[llm]'     # exact token counts via tiktoken (exact_tokens=True)
pip install 'docslicer[crypto]'  # password-protected Office files (msoffcrypto-tool)
pip install 'docslicer[parquet]' # Parquet export support
```

Extras can be combined: `pip install 'docslicer[html,ocr,llm]'`.

**Requires Python 3.10+**

---

## What you get back (ParseResult)

`parse_document` returns a `ParseResult`:

```python
result.chunks      # list[Chunk]   — heading-aware text chunks, ready for embedding
result.blocks      # list[Block]   — paragraph/heading/table blocks before chunking
result.tables      # list[Table]   — structured tables with cells, spans, and markdown
result.charts      # list[Chart]   — charts as extracted data points (docx/pptx)
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
chunk.chart_ids     # list  — IDs of charts referenced in this chunk (docx/pptx)
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

### Parsing & content options

`parse_document` (and the format-specific functions) accept options that control what
gets parsed and how, before it's chunked. Format-specific toggles are accepted everywhere
for a uniform API but only take effect for the relevant format.

```python
result = docslicer.parse_document(
    "contract.docx",
    password="admin123",           # decrypt password-protected files; .docx and .pptx needs [crypto] extra
    max_workers=4,                 # process-pool width for PDF extraction/OCR (default: auto by CPU cores)
    include_headers_footers=True,  # docx: include header/footer content (default False)
    include_footnotes=True,        # docx: include footnotes (default True)
    include_comments=True,         # docx: include review comments (default False)
    include_speaker_notes=True,    # pptx: include slide speaker notes (default True)
    use_browser=True,              # html/URL: render in a real browser (default True)
)
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
    exact_tokens=True,            # exact tiktoken (cl100k_base) counts; needs [llm] extra, else char/4 estimate
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

Point `parse_all` at a folder (or pass a list of paths/URLs). It yields `(source, result)`
pairs, and a file that fails to parse yields the `Exception` instead of aborting the batch.
Any `parse_document` keyword — chunk sizes, `include_*`, etc. — is forwarded per document.

```python
for path, result in docslicer.parse_all("documents/", recursive=True, max_chunk_size=2000):
    if isinstance(result, Exception):
        print(f"Failed {path}: {result}")
    else:
        print(f"{path}: {len(result.chunks)} chunks")
```

### Reuse config across documents

`DocumentParser` holds a fixed `ParseConfig` across many documents and keeps a single
browser open across HTML/URL inputs (launched lazily on the first HTML parse), so a batch
of URLs starts Chromium once instead of once per document. Use it as a context manager so
that browser is always released:

```python
from docslicer import DocumentParser, ParseConfig

config = ParseConfig(max_chunk_size=1500, optimal_chunk_size=600)

with DocumentParser(config) as parser:
    for path, result in parser.parse_all(paths):   # or parser.parse(path) for one
        ...
```

### Two levels of parallelism

There are two independent knobs, and they compose:

- **`ParseConfig(max_workers=N)`** — *within* a single document: parallelizes PDF word
  extraction, cell building, and OCR across processes (default: auto, sized to CPU cores).
  Best when documents are large.
- **`DocumentParser(config, workers=N)`** — *across* documents: fans whole documents out
  over `N` worker processes, each with its own config and browser. Best when you have many
  documents. Results arrive in submission order (this path isn't lazy per-document).

Setting `workers` alone defaults each worker's `max_workers` to `1`, so nested pools don't
oversubscribe the machine; set both explicitly to run both levels at once. The `workers`
path can't forward a browser session or `on_stage` callback across processes — leave
`workers` unset when you need those.

> **Guard your entry point.** DocSlicer uses a `ProcessPoolExecutor` whenever there's
> real CPU work to fan out — **any PDF over ~50 pages, any scanned/OCR PDF of any length**,
> and both parallelism knobs above. This is not opt-in: a plain
> `docslicer.parse_document("big.pdf")` triggers it too. On macOS and Windows, Python
> spawns workers by re-importing your script top to bottom, so a parse that runs at module
> level makes each worker re-run it and spawn again — raising `RuntimeError: An attempt has
> been made to start a new process before the current process ... bootstrapping phase`.
> Put your parsing code inside a function behind an `if __name__ == "__main__":` guard:
>
> ```python
> def main():
>     with DocumentParser(config, workers=4) as parser:
>         for path, result in parser.parse_all(paths):
>             ...
>
> if __name__ == "__main__":
>     main()
> ```

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
charts = result.charts_under(node)              # charts (with extracted data points) in this section
blocks = result.blocks_under(node)              # raw paragraph/heading blocks
```

### Navigate by page

```python
result.chunks_by_page(14)        # by page number
result.chunks_by_page("F-3")     # by printed page label
result.blocks_by_page(14)
result.tables_by_page(14)
result.charts_by_page(14)
```

### Parse once, navigate many times

A parsed result is plain data, so you can persist it and reload it later. When an
agent asks many questions about the same document, there's no need to parse it
again on every question:

```python
from pathlib import Path
import docslicer

cache = Path("annual_report.json")

if cache.exists():
    result = docslicer.ParseResult.load(cache)
else:
    result = docslicer.parse_document("annual_report.pdf")
    result.save(cache)
```

A reloaded result supports the full API — `hierarchy`, `find_heading`,
`chunks_under`, `tables` — so a long-running agent session or document server can
keep documents open across requests without re-parsing.

---

## Export

`save()` decides what to write from the path you give it.

```python
# Save the whole result and reload it later — keeps the heading hierarchy
result.save("result.json")                     # same output as result.to_json()
result = docslicer.ParseResult.load("result.json")

# A single collection, in the format you name
result.save("chunks.csv")
result.save("charts.jsonl")       # stems: chunks | blocks | tables | charts | metadata
result.export_chunks_jsonl("chunks.jsonl")

# One file per collection
result.save("output/")
# → output/chunks.parquet, blocks.parquet, tables.parquet, metadata.json
#   (+ charts.parquet when the document has charts)
#   Falls back to .csv unless the [parquet] extra is installed.

# Render as Markdown or plain text
md = result.export_to_markdown(include_tables=True)
txt = result.export_to_text()

# DataFrames
df = result.chunks_df()
```

Only `result.json` round-trips — the collection and directory forms write flat rows
without the heading hierarchy, so `ParseResult.load()` can't read them back.

### Debug mode

```python
result = docslicer.parse_document("report.pdf", debug=True)

# result.pipeline_steps is an ordered dict of step name → DataFrame
for name, df in result.pipeline_steps.items():
    print(name, df.shape)
    df.to_csv(f"debug/{name}.csv", index=False)

# PDF steps:        words → shapes → cells → lines → table_cells → blocks → chunks
# DOCX/PPTX steps:  runs → chart_points → paragraphs → lines → table_cells → blocks → chunks
```

---

## OCR

`parse_document` automatically detects scanned pages and falls back to OCR when the
`[ocr]` extra is installed. No configuration needed — `result.metadata.has_ocr`
tells you whether OCR was used.

```bash
pip install 'docslicer[ocr]'
# tesserocr binds libtesseract directly, so install the Tesseract dev libraries first:
# Linux:  apt install tesseract-ocr libtesseract-dev libleptonica-dev pkg-config
# macOS:  brew install tesseract leptonica
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

DocSlicer is **dual-licensed**:

- **[AGPL-3.0](LICENSE)** — free to use, modify, and distribute, provided you comply with the AGPL's terms, including making the complete source of any application that uses DocSlicer available to its users (including over a network).
- **[Commercial license](LICENSE-COMMERCIAL.md)** — for embedding DocSlicer in a closed-source or proprietary product, or offering it as part of a hosted/SaaS service without releasing your source.

See [LICENSE-COMMERCIAL.md](LICENSE-COMMERCIAL.md) for details, or reach out about a commercial license.
