"""Export tests — save/load round-trip, text formats, CSV, hierarchy serialisation."""

import json
import pytest
from pathlib import Path

import docslicer
from docslicer import ParseResult


# ── JSON round-trip ───────────────────────────────────────────────────────────

def test_to_json_is_valid(pdf_result):
    raw = pdf_result.to_json()
    assert isinstance(raw, str)
    data = json.loads(raw)
    assert "chunks" in data
    assert "blocks" in data
    assert "tables" in data
    assert "metadata" in data
    assert "hierarchy" in data


def test_save_and_load_roundtrip(pdf_result, tmp_path):
    path = tmp_path / "result.json"
    pdf_result.save(path)
    assert path.exists()
    loaded = ParseResult.load(path)
    assert len(loaded.chunks) == len(pdf_result.chunks)
    assert len(loaded.tables) == len(pdf_result.tables)
    assert loaded.chunks[0].id == pdf_result.chunks[0].id
    assert loaded.chunks[0].text == pdf_result.chunks[0].text


# ── Text exports ──────────────────────────────────────────────────────────────

def test_export_to_markdown(pdf_result):
    md = pdf_result.export_to_markdown()
    assert isinstance(md, str)
    assert len(md) > 0


def test_export_to_markdown_with_tables(pdf_result):
    md = pdf_result.export_to_markdown(include_tables=True, prettify=True)
    assert isinstance(md, str)
    assert len(md) > 0


def test_export_to_text(pdf_result):
    txt = pdf_result.export_to_text()
    assert isinstance(txt, str)
    assert len(txt) > 0


# ── CSV exports ───────────────────────────────────────────────────────────────

def test_save_chunks_csv(pdf_result, tmp_path):
    path = tmp_path / "chunks.csv"
    pdf_result.save(path)
    assert path.exists()
    assert path.stat().st_size > 0


def test_export_chunks_jsonl(pdf_result, tmp_path):
    path = tmp_path / "chunks.jsonl"
    pdf_result.export_chunks_jsonl(path)
    assert path.exists()
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == len(pdf_result.chunks)
    assert json.loads(lines[0])["id"] == pdf_result.chunks[0].id


def test_export_tables_csv(pdf_result, tmp_path):
    if not pdf_result.tables:
        pytest.skip("no tables in this document")
    path = tmp_path / "tables.csv"
    pdf_result.export_tables_csv(path)
    assert path.exists()
    assert path.stat().st_size > 0


# ── DataFrames ────────────────────────────────────────────────────────────────

def test_chunks_df(pdf_result):
    df = pdf_result.chunks_df()
    assert len(df) == len(pdf_result.chunks)
    assert "id" in df.columns
    assert "text" in df.columns
    assert "page_number" in df.columns


def test_blocks_df(pdf_result):
    df = pdf_result.blocks_df()
    assert len(df) == len(pdf_result.blocks)
    assert "id" in df.columns
    assert "type" in df.columns


# ── Hierarchy serialisation ───────────────────────────────────────────────────

def test_hierarchy_save(pdf_result, tmp_path):
    path = tmp_path / "hierarchy.json"
    pdf_result.hierarchy.save(path)
    assert path.exists()
    data = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(data, list)


def test_hierarchy_save_minimal(pdf_result, tmp_path):
    path = tmp_path / "hierarchy_minimal.json"
    pdf_result.hierarchy.save(path, minimal=True)
    assert path.exists()
    data = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(data, list)
    if data:
        assert set(data[0].keys()) <= {"text", "children"}


def test_hierarchy_save_outline(pdf_result, tmp_path):
    path = tmp_path / "outline.md"
    pdf_result.hierarchy.save_outline(path)
    assert path.exists()
    content = path.read_text(encoding="utf-8")
    assert isinstance(content, str)


# ── Extra fields ──────────────────────────────────────────────────────────────

def test_extra_fields(tmp_path):
    from pathlib import Path
    samples = Path(__file__).parent.parent / "examples" / "sample_docs"
    result = docslicer.parse_document(
        samples / "financial_report.pdf",
        extra_fields=["is_bold", "font_size"],
    )
    assert result.chunks[0].extra is not None
    assert "is_bold" in result.chunks[0].extra or result.chunks[0].extra == {}
