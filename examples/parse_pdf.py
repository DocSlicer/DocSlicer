"""
parse_pdf.py — Parse a PDF with DocSlicer and inspect the result.

Usage:
    python examples/parse_pdf.py
    python examples/parse_pdf.py path/to/your.pdf
    python examples/parse_pdf.py path/to/your.pdf --save output.json
"""

import sys
from pathlib import Path

# If running from the repo without installing, add src/ to the path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from docslicer import parse_pdf

# ── Input ──────────────────────────────────────────────────────────────────────
pdf_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / "sample_docs" / "financial_report.pdf"
save_path = None
if "--save" in sys.argv:
    save_path = Path(sys.argv[sys.argv.index("--save") + 1])

# ── Parse ──────────────────────────────────────────────────────────────────────
print(f"Parsing {pdf_path.name} …")
result = parse_pdf(pdf_path)

# ── Metadata ───────────────────────────────────────────────────────────────────
meta = result.metadata
print(f"\n── Metadata ──────────────────────────")
print(f"  Title   : {meta.title}")
print(f"  Author  : {meta.author}")
print(f"  Pages   : {meta.page_count}")
print(f"  Language: {meta.language}")
print(f"  OCR     : {meta.has_ocr}")

# ── Chunks ─────────────────────────────────────────────────────────────────────
print(f"\n── Chunks ({len(result.chunks)} total) ────────────────────")
for chunk in result.chunks[:3]:
    hierarchy = " > ".join(chunk.hierarchy) if chunk.hierarchy else "(no heading)"
    print(f"\n  [{chunk.chunk_index}] p.{chunk.page}  {chunk.region}  {chunk.char_count} chars")
    print(f"  Hierarchy : {hierarchy}")
    print(f"  Text      : {chunk.text[:120].replace(chr(10), ' ')} …")

if len(result.chunks) > 3:
    print(f"\n  … and {len(result.chunks) - 3} more chunks")

# ── Blocks ─────────────────────────────────────────────────────────────────────
print(f"\n── Blocks ({len(result.blocks)} total) ────────────────────")
for block in result.blocks[:3]:
    print(f"  [{block.id}] p.{block.page}  role={block.role}  region={block.region}  {block.char_count} chars")

if len(result.blocks) > 3:
    print(f"  … and {len(result.blocks) - 3} more blocks")

# ── Tables ─────────────────────────────────────────────────────────────────────
print(f"\n── Tables ({len(result.tables)} total) ────────────────────")
for table in result.tables:
    print(f"\n  [{table.id}] p.{table.page}  caption={table.caption}")
    # Print first 3 lines of markdown
    lines = table.markdown.splitlines()
    for line in lines[:4]:
        print(f"  {line}")
    if len(lines) > 4:
        print(f"  … ({len(lines) - 4} more rows)")

# ── Save ───────────────────────────────────────────────────────────────────────
if save_path:
    result.save(save_path)
    print(f"\nSaved to {save_path}")
