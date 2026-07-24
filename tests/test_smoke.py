"""Smoke tests — every sample file parses without error and returns a usable result."""

import docslicer
from docslicer import ParseResult


def _assert_result(result: ParseResult, label: str) -> None:
    assert isinstance(result, ParseResult), f"{label}: expected ParseResult"
    assert len(result.chunks) > 0, f"{label}: no chunks"
    assert len(result.blocks) > 0, f"{label}: no blocks"
    assert result.metadata is not None, f"{label}: no metadata"
    assert result.metadata.page_count > 0, f"{label}: page_count is 0"
    for chunk in result.chunks:
        assert chunk.id, f"{label}: chunk missing id"
        assert chunk.text, f"{label}: chunk missing text"
        assert chunk.page_number > 0, f"{label}: chunk page_number <= 0"
        assert chunk.section, f"{label}: chunk missing section"
        assert chunk.char_count == len(chunk.text), f"{label}: char_count mismatch"


def test_pdf(pdf_result):
    _assert_result(pdf_result, "PDF")


def test_scanned_pdf(scanned_result):
    _assert_result(scanned_result, "scanned PDF")


def test_html(html_result):
    _assert_result(html_result, "HTML")


def test_docx(docx_result):
    _assert_result(docx_result, "DOCX")


def test_pptx(pptx_result):
    _assert_result(pptx_result, "PPTX")


def test_scanned_pdf_ocr_flag(scanned_result):
    assert scanned_result.metadata.has_ocr is True


def test_pdf_has_tables(pdf_result):
    assert len(pdf_result.tables) > 0, "financial_report.pdf should have tables"
    for table in pdf_result.tables:
        assert table.id
        assert table.markdown
        assert len(table.cells) > 0
