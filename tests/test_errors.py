"""Error-path tests — the failure modes a public API is judged on.

These lock the *observable* contract for bad input: which calls raise (and with
what exception), which degrade gracefully, and that batch parsing isolates a
failing document instead of aborting. Behaviours here were verified against the
implementation, not assumed.
"""
from pathlib import Path

import pytest

import docslicer
from docslicer import ParseConfig, ParseResult

SAMPLES = Path(__file__).resolve().parent.parent / "examples" / "sample_docs"
PDF = SAMPLES / "financial_report.pdf"

# Minimal OLE compound-file header — enough to make a file that exists, isn't a
# PDF, and isn't an Office Open XML (ZIP) package, so detection reaches the
# legacy-format branch.
_OLE_JUNK = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1 legacy ole payload"


# ── Missing files ─────────────────────────────────────────────────────────────

def test_parse_pdf_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        docslicer.parse_pdf("/no/such/file.pdf")


def test_parse_docx_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        docslicer.parse_docx("/no/such/file.docx")


def test_parse_document_missing_path_with_extension_raises():
    """A nonexistent path with a document extension is a typo, not HTML content:
    parse_document must raise rather than silently parse the filename as HTML.
    """
    with pytest.raises(FileNotFoundError):
        docslicer.parse_document("/no/such/file.pdf")


def test_parse_document_extensionless_string_still_html():
    """The HTML-string fallback is preserved: a bare string with no document
    extension is still treated as markup, not a missing file.
    """
    result = docslicer.parse_document("just some plain text, not a path")
    assert isinstance(result, ParseResult)


# ── Unsupported / legacy formats ──────────────────────────────────────────────

def test_legacy_ppt_raises_value_error(tmp_path):
    p = tmp_path / "deck.ppt"
    p.write_bytes(_OLE_JUNK)
    with pytest.raises(ValueError, match="legacy Office format"):
        docslicer.parse_document(p)


def test_legacy_doc_raises_value_error(tmp_path):
    p = tmp_path / "memo.doc"
    p.write_bytes(_OLE_JUNK)
    with pytest.raises(ValueError, match="legacy Office format"):
        docslicer.parse_document(p)


# ── Corrupt payloads ──────────────────────────────────────────────────────────

def test_corrupt_pdf_bytes_raises():
    """Bytes with a %PDF magic number but no valid structure must raise, not
    return a bogus empty result that looks like a successful parse.
    """
    with pytest.raises(Exception):
        docslicer.parse_pdf(b"%PDF-1.4 this is not a valid pdf payload")


# ── parse_all input validation ────────────────────────────────────────────────

def test_parse_all_non_directory_raises():
    with pytest.raises(ValueError, match="is not a directory"):
        list(docslicer.parse_all("/no/such/dir"))


def test_parse_all_isolates_a_broken_item():
    """A broken source yields (source, Exception); the good one still parses.
    parse_all must never let one bad document abort the batch.
    """
    broken = b"%PDF-1.4 this is not a valid pdf payload"
    out = list(docslicer.parse_all([PDF, broken]))
    assert len(out) == 2
    by_id = {id(src): res for src, res in out}
    assert isinstance(by_id[id(PDF)], ParseResult)
    assert isinstance(by_id[id(broken)], Exception)


# ── ParseConfig validation ────────────────────────────────────────────────────
# Invalid config must fail fast at construction, not silently mis-parse later.

def test_default_config_is_valid():
    ParseConfig()  # must not raise


def test_invalid_table_representation_raises():
    with pytest.raises(ValueError, match="table_representation"):
        ParseConfig(table_representation="fancy")


def test_non_positive_chunk_size_raises():
    with pytest.raises(ValueError, match="positive integer"):
        ParseConfig(max_chunk_size=0)


def test_contradictory_chunk_size_ordering_raises():
    with pytest.raises(ValueError, match="min_chunk_size <= optimal_chunk_size"):
        ParseConfig(min_chunk_size=2000, optimal_chunk_size=1500, max_chunk_size=3200)


def test_non_positive_max_workers_raises():
    with pytest.raises(ValueError, match="max_workers"):
        ParseConfig(max_workers=0)


def test_extra_fields_must_be_list_of_strings():
    with pytest.raises(ValueError, match="extra_fields"):
        ParseConfig(extra_fields="is_bold")  # a bare string, not a list


def test_invalid_config_surfaces_through_parse_functions():
    """The top-level parse_* helpers build a ParseConfig, so a bad value there is
    rejected up front rather than deep in the pipeline.
    """
    with pytest.raises(ValueError, match="table_representation"):
        docslicer.parse_pdf(PDF, table_representation="fancy")
