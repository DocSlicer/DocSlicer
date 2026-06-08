"""
hierarchy.py — Navigate the document structure programmatically.

Shows how to traverse the heading tree, search headings, and pull
chunks / blocks / tables from any node or page.

Usage:
    python examples/hierarchy.py
    python examples/hierarchy.py path/to/your/document.pdf
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import docslicer

SOURCE = sys.argv[1] if len(sys.argv) > 1 else Path(__file__).parent / "sample_docs" / "financial_report.pdf"

print(f"Parsing: {SOURCE}")
result = docslicer.parse_document(SOURCE)
print(f"  {len(result.chunks)} chunks  {len(result.blocks)} blocks  {len(result.tables)} tables")

# ── Outline ────────────────────────────────────────────────────────────────────

print(f"\n── Outline ──────────────────────────────────────")
print(result.hierarchy.to_outline() or "  (no headings detected)")

# ── Level listings ─────────────────────────────────────────────────────────────

l1 = result.hierarchy.level(1)
print(f"\n── hierarchy.level(1) — {len(l1)} nodes ──────────────")
for n in l1:
    print(f"  [{n.heading_id}] p.{n.page_number}  {n.text}")

l2 = result.hierarchy.level(2)
print(f"\n── hierarchy.level(2) — {len(l2)} nodes (with ancestor path) ──")
for n in l2[:6]:
    path_str = " > ".join(n.path) if n.path else "(root)"
    print(f"  [{n.heading_id}] p.{n.page_number}  {path_str} > {n.text}")
if len(l2) > 6:
    print(f"  … and {len(l2) - 6} more")

# ── Children of a node ────────────────────────────────────────────────────────

if l1:
    parent = l1[0]
    children = result.hierarchy.level(2, parent=parent)
    print(f"\n── level(2, parent='{parent.text}') — {len(children)} children ──")
    for n in children:
        print(f"  [{n.heading_id}] p.{n.page_number}  {n.text}")

# ── find_heading() ─────────────────────────────────────────────────────────────

SEARCH = "financial"
found = result.find_heading(SEARCH)
print(f"\n── find_heading('{SEARCH}') — {len(found)} matches ──────")
for n in found[:5]:
    path_str = " > ".join(n.path) if n.path else "(root)"
    print(f"  [{n.heading_id}] p.{n.page_number}  {path_str} > {n.text}")

# ── chunks_under() / blocks_under() ───────────────────────────────────────────

if found:
    node = found[0]
    chunks = result.chunks_under(node)
    chunks_direct = result.chunks_under(node, recursive=False)
    print(f"\n── chunks_under('{node.text}') ──────────────────")
    print(f"  recursive=True  → {len(chunks)} chunks")
    print(f"  recursive=False → {len(chunks_direct)} chunks")
    for c in chunks[:2]:
        print(f"\n  [{c.chunk_index}] p.{c.page_number}  {c.char_count} chars")
        print(f"  {c.text[:120].replace(chr(10), ' ')} …")

    blocks = result.blocks_under(node)
    print(f"\n── blocks_under('{node.text}') → {len(blocks)} blocks ───")
    for b in blocks[:3]:
        print(f"  [{b.id}] role={b.role}  p.{b.page_number}  {b.text[:80].replace(chr(10), ' ')}")

# ── tables_under() ────────────────────────────────────────────────────────────

all_nodes = result.hierarchy.flatten()
node_with_tables = next((n for n in all_nodes if result.tables_under(n)), None)
if node_with_tables:
    tables = result.tables_under(node_with_tables)
    print(f"\n── tables_under('{node_with_tables.text}') → {len(tables)} tables ──")
    for t in tables[:2]:
        print(f"\n  [{t.id}] p.{t.page_number}  {len(t.cells)} cells  caption={t.caption!r}")
        for line in t.markdown.splitlines()[:4]:
            print(f"  {line}")

# ── chunks_by_page() / blocks_by_page() ───────────────────────────────────────

if result.chunks:
    page = result.chunks[0].page_number
    label = result.chunks[0].page_label

    page_chunks = result.chunks_by_page(page)
    print(f"\n── chunks_by_page({page}) → {len(page_chunks)} chunks ────────")
    for c in page_chunks[:3]:
        print(f"  [{c.id}]  [{c.section}]  {c.char_count} chars")

    if label and label != str(page):
        label_chunks = result.chunks_by_page(label)
        print(f"  (chunks_by_page('{label}') by label → same {len(label_chunks)} chunks)")

    page_blocks = result.blocks_by_page(page)
    print(f"\n── blocks_by_page({page}) → {len(page_blocks)} blocks ───────")
    for b in page_blocks[:3]:
        print(f"  [{b.id}]  role={b.role}  {b.text[:60].replace(chr(10), ' ')}")

print()
