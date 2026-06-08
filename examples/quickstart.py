"""
quickstart.py — Parse a document and explore the result.

Runs out of the box against the included sample PDF.

Usage:
    python examples/quickstart.py
    python examples/quickstart.py path/to/your/document.pdf
    python examples/quickstart.py https://example.com/report.html
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import docslicer

# ── Input ──────────────────────────────────────────────────────────────────────

source = sys.argv[1] if len(sys.argv) > 1 else Path(__file__).parent / "sample_docs" / "financial_report.pdf"

print(f"Parsing: {source}")
result = docslicer.parse_document(source)

# ── Metadata ───────────────────────────────────────────────────────────────────

meta = result.metadata
print(f"\n── Metadata ────────────────────────────────────")
print(f"  Title    : {meta.title}")
print(f"  Author   : {meta.author}")
print(f"  Language : {meta.language}")
print(f"  Pages    : {meta.page_count}")
print(f"  OCR used : {meta.has_ocr}")
print(f"  Tokens   : {meta.estimated_tokens}")

# ── Hierarchy ──────────────────────────────────────────────────────────────────

print(f"\n── Document outline ────────────────────────────")
print(result.hierarchy.to_outline() or "  (no headings detected)")

# ── Chunks ─────────────────────────────────────────────────────────────────────

print(f"\n── Chunks ({len(result.chunks)} total) ─────────────────────")
for chunk in result.chunks[:3]:
    path = " > ".join(chunk.path) if chunk.path else "(no heading)"
    print(f"\n  [{chunk.chunk_index}] p.{chunk.page_number}  {chunk.section}  {chunk.char_count} chars")
    print(f"  Path : {path}")
    print(f"  Text : {chunk.text[:120].replace(chr(10), ' ')} …")

if len(result.chunks) > 3:
    print(f"\n  … and {len(result.chunks) - 3} more chunks")

# ── Heading navigation ─────────────────────────────────────────────────────────

# Find the first top-level heading and pull everything under it
top_level = result.hierarchy.level(1)
if top_level:
    node = top_level[0]
    under = result.chunks_under(node)
    print(f"\n── chunks_under('{node.text}') → {len(under)} chunks ──")
    for chunk in under[:2]:
        print(f"  p.{chunk.page_number}  {chunk.char_count} chars  {chunk.text[:80].replace(chr(10), ' ')} …")

# ── Tables ─────────────────────────────────────────────────────────────────────

print(f"\n── Tables ({len(result.tables)} total) ─────────────────────")
for table in result.tables[:2]:
    label = table.page_label or str(table.page_number)
    print(f"\n  [{table.id}] p.{label}  {len(table.cells)} cells  caption={table.caption!r}")
    for line in table.markdown.splitlines()[:4]:
        print(f"  {line}")
    if len(table.markdown.splitlines()) > 4:
        print(f"  … ({len(table.markdown.splitlines()) - 4} more rows)")

if not result.tables:
    print("  (no tables detected)")

print()
