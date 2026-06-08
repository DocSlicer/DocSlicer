"""API contract tests — hierarchy navigation, chunk/block retrieval, page lookups."""

import pytest
from docslicer import HierarchyNode, Chunk, Block


# ── Chunk / Block fields ───────────────────────────────────────────────────────

def test_chunk_fields(pdf_result):
    chunk = pdf_result.chunks[0]
    assert isinstance(chunk.id, str)
    assert isinstance(chunk.chunk_index, int)
    assert isinstance(chunk.page_number, int)
    assert isinstance(chunk.section, str)
    assert isinstance(chunk.text, str)
    assert isinstance(chunk.char_count, int)
    assert isinstance(chunk.path, list)
    assert isinstance(chunk.table_ids, list)
    assert isinstance(chunk.extra, dict)


def test_block_fields(pdf_result):
    block = pdf_result.blocks[0]
    assert isinstance(block.id, str)
    assert isinstance(block.type, str)
    assert isinstance(block.page_number, int)
    assert isinstance(block.section, str)
    assert isinstance(block.text, str)


# ── Hierarchy ─────────────────────────────────────────────────────────────────

def test_hierarchy_level1(pdf_result):
    l1 = pdf_result.hierarchy.level(1)
    assert len(l1) > 0
    for node in l1:
        assert isinstance(node, HierarchyNode)
        assert node.heading_id
        assert node.text
        assert isinstance(node.path, list)


def test_hierarchy_level2_has_path(pdf_result):
    l2 = pdf_result.hierarchy.level(2)
    nodes_with_path = [n for n in l2 if n.path]
    assert len(nodes_with_path) > 0, "level(2) nodes should have ancestor path populated"


def test_hierarchy_level2_parent_filter(pdf_result):
    l1 = pdf_result.hierarchy.level(1)
    if not l1:
        pytest.skip("no level-1 headings")
    parent = l1[0]
    by_node = pdf_result.hierarchy.level(2, parent=parent)
    by_id = pdf_result.hierarchy.level(2, parent=parent.heading_id)
    assert len(by_node) == len(by_id)
    assert all(isinstance(n, HierarchyNode) for n in by_node)


def test_find_heading(pdf_result):
    l1 = pdf_result.hierarchy.level(1)
    if not l1:
        pytest.skip("no headings")
    term = l1[0].text.split()[0].lower()
    found = pdf_result.find_heading(term)
    assert len(found) > 0
    for node in found:
        assert term in node.text.lower()
        assert isinstance(node.path, list)


def test_find_heading_no_match(pdf_result):
    assert pdf_result.find_heading("xyzzy_no_match_42") == []


def test_hierarchy_flatten(pdf_result):
    nodes = pdf_result.hierarchy.flatten()
    assert len(nodes) > 0
    assert all(isinstance(n, HierarchyNode) for n in nodes)


# ── chunks_under / blocks_under ───────────────────────────────────────────────

def test_chunks_under(pdf_result):
    l1 = pdf_result.hierarchy.level(1)
    if not l1:
        pytest.skip("no level-1 headings")
    node = l1[0]
    chunks = pdf_result.chunks_under(node)
    chunks_direct = pdf_result.chunks_under(node, recursive=False)
    assert all(isinstance(c, Chunk) for c in chunks)
    assert len(chunks) >= len(chunks_direct)


def test_blocks_under(pdf_result):
    l1 = pdf_result.hierarchy.level(1)
    if not l1:
        pytest.skip("no level-1 headings")
    blocks = pdf_result.blocks_under(l1[0])
    assert all(isinstance(b, Block) for b in blocks)


def test_tables_under(pdf_result):
    from docslicer import Table
    nodes = pdf_result.hierarchy.flatten()
    node_with_tables = next((n for n in nodes if pdf_result.tables_under(n)), None)
    if node_with_tables is None:
        pytest.skip("no hierarchy node with tables")
    tables = pdf_result.tables_under(node_with_tables)
    assert len(tables) > 0
    assert all(isinstance(t, Table) for t in tables)


# ── by_page ───────────────────────────────────────────────────────────────────

def test_chunks_by_page(pdf_result):
    page = pdf_result.chunks[0].page_number
    chunks = pdf_result.chunks_by_page(page)
    assert len(chunks) > 0
    assert all(c.page_number == page for c in chunks)


def test_blocks_by_page(pdf_result):
    page = pdf_result.blocks[0].page_number
    blocks = pdf_result.blocks_by_page(page)
    assert len(blocks) > 0
    assert all(b.page_number == page for b in blocks)


def test_chunks_by_page_label(pdf_result):
    chunk = next((c for c in pdf_result.chunks if c.page_label), None)
    if chunk is None:
        pytest.skip("no page labels in this document")
    by_label = pdf_result.chunks_by_page(chunk.page_label)
    by_number = pdf_result.chunks_by_page(chunk.page_number)
    assert len(by_label) == len(by_number)
