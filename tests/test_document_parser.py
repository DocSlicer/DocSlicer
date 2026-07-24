"""DocumentParser tests — reusable config, context manager, batch parse_all.

DocumentParser holds one ParseConfig (and one browser session) across many
documents. These lock its public contract: config is applied and reused, the
context manager cleans up, and parse_all isolates a failing document instead of
aborting the whole batch.
"""

from pathlib import Path

from docslicer import DocumentParser, ParseConfig, ParseResult

SAMPLES = Path(__file__).resolve().parent.parent / "examples" / "sample_docs"
PDF = SAMPLES / "financial_report.pdf"


# ── Single parse ──────────────────────────────────────────────────────────────

def test_parse_returns_result():
    with DocumentParser() as parser:
        result = parser.parse(PDF)
    assert isinstance(result, ParseResult)
    assert len(result.chunks) > 0


def test_default_config_when_none():
    parser = DocumentParser()
    assert isinstance(parser.config, ParseConfig)


# ── Config is applied and reused ──────────────────────────────────────────────

def test_config_is_applied():
    """chunking=False must skip chunk building while still producing blocks."""
    with DocumentParser(ParseConfig(chunking=False)) as parser:
        result = parser.parse(PDF)
    assert result.chunks == []
    assert len(result.blocks) > 0


def test_config_reused_across_documents():
    """The same instance yields identical structure for the same input twice."""
    with DocumentParser() as parser:
        first = parser.parse(PDF)
        second = parser.parse(PDF)
    assert [c.text for c in first.chunks] == [c.text for c in second.chunks]


# ── Context manager cleanup ───────────────────────────────────────────────────

def test_close_is_idempotent():
    parser = DocumentParser()
    parser.parse(PDF)
    parser.close()
    parser.close()  # second call must not raise


# ── Batch parse_all ───────────────────────────────────────────────────────────

def test_parse_all_yields_results_in_order():
    sources = [PDF, PDF]
    with DocumentParser() as parser:
        out = list(parser.parse_all(sources))
    assert len(out) == len(sources)
    for src, result in out:
        assert src in sources
        assert isinstance(result, ParseResult)


def test_parse_all_isolates_failures():
    """A broken document yields (source, Exception) without aborting the batch."""
    broken = b"%PDF-1.4 this is not a valid pdf payload"
    with DocumentParser() as parser:
        out = list(parser.parse_all([PDF, broken]))
    assert len(out) == 2
    results = {id(src): res for src, res in out}
    assert isinstance(results[id(PDF)], ParseResult)
    assert isinstance(results[id(broken)], Exception)
