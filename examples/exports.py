"""
exports.py — All export formats and output options.

Shows every way to save or convert a ParseResult: JSON, CSV, JSONL,
Parquet, Markdown, plain text, tables, hierarchy, and DataFrames.

Usage:
    python examples/exports.py
    python examples/exports.py path/to/your/document.pdf
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import docslicer

SOURCE = sys.argv[1] if len(sys.argv) > 1 else Path(__file__).parent / "sample_docs" / "financial_report.pdf"
OUTPUT = Path(__file__).parent / "output"
OUTPUT.mkdir(exist_ok=True)

print(f"Parsing: {SOURCE}")
result = docslicer.parse_document(SOURCE)
print(f"  {len(result.chunks)} chunks  {len(result.blocks)} blocks  {len(result.tables)} tables\n")

# ── Full result ────────────────────────────────────────────────────────────────
# Saves chunks + blocks + tables + metadata + hierarchy as a single JSON file.
# Reload later without re-parsing: docslicer.ParseResult.load("output/result.json")

result.save(OUTPUT / "result.json")
print(f"Saved full result  → {OUTPUT / 'result.json'}")

# ── Chunks ─────────────────────────────────────────────────────────────────────

result.save(OUTPUT / "chunks.json")
result.save(OUTPUT / "chunks.csv")
result.export_chunks_jsonl(OUTPUT / "chunks.jsonl")
print(f"Saved chunks       → chunks.json / .csv / .jsonl")

# ── Parquet (requires: pip install 'docslicer[parquet]') ──────────────────────

# result.save(OUTPUT / "chunks.parquet")
# result.save(OUTPUT / "")   # → chunks.parquet + blocks.parquet + tables.parquet + metadata.json

# ── Markdown and plain text ────────────────────────────────────────────────────

md = result.export_to_markdown(include_tables=True, prettify=True)
(OUTPUT / "result.md").write_text(md, encoding="utf-8")
print(f"Saved markdown     → {OUTPUT / 'result.md'}")

txt = result.export_to_text()
(OUTPUT / "result.txt").write_text(txt, encoding="utf-8")
print(f"Saved plain text   → {OUTPUT / 'result.txt'}")

# ── Tables ─────────────────────────────────────────────────────────────────────

result.export_tables_csv(OUTPUT / "tables.csv", encoding="utf-8-sig")
print(f"Saved tables       → {OUTPUT / 'tables.csv'}  (Excel-friendly UTF-8 BOM)")

# ── Hierarchy ─────────────────────────────────────────────────────────────────

result.hierarchy.save(OUTPUT / "hierarchy.json")
result.hierarchy.save(OUTPUT / "hierarchy_minimal.json", minimal=True)
result.hierarchy.save_outline(OUTPUT / "outline.md")
print(f"Saved hierarchy    → hierarchy.json / hierarchy_minimal.json / outline.md")

# ── Metadata ──────────────────────────────────────────────────────────────────

result.save(OUTPUT / "metadata.json")
print(f"Saved metadata     → {OUTPUT / 'metadata.json'}")

# ── DataFrames ────────────────────────────────────────────────────────────────

chunks_df = result.chunks_df()
blocks_df = result.blocks_df()
print(f"\nDataFrames:  chunks {chunks_df.shape}   blocks {blocks_df.shape}")
print(chunks_df[["id", "page_number", "section", "char_count", "heading"]].head(3).to_string(index=False))

# ── Extra fields ──────────────────────────────────────────────────────────────
# Attach raw pipeline columns (font_name, is_bold, font_size, …) to each chunk/block.

rich = docslicer.parse_document(SOURCE, extra_fields=["is_bold", "font_name", "font_size"])
if rich.chunks:
    print(f"\nExtra fields sample: {rich.chunks[0].extra}")

print("\nDone.")
