"""
inputs.py — All supported input types.

docslicer.parse_document() accepts local files (PDF, DOCX, PPTX, HTML)
and remote URLs. This file demonstrates each format.

The three local examples run by default. Uncomment any URL or file path
section to try that input type.

The work runs under `if __name__ == "__main__":` — docslicer parses CPU-bound
steps (PDF word extraction, OCR, ...) across a process pool, and on spawn-based
platforms (macOS, Windows) each worker re-imports this file. The guard keeps that
re-import from re-running the parses in every worker.

Usage:
    python examples/inputs.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import docslicer

SAMPLE = Path(__file__).parent / "sample_docs"


def show(label: str, source) -> None:
    print(f"\nParsing: {label}")
    try:
        result = docslicer.parse_document(source)
    except Exception as e:
        print(f"  FAILED: {e}")
        return
    print(f"  {len(result.chunks)} chunks  {len(result.blocks)} blocks  {len(result.tables)} tables")
    print(f"  Title  : {result.metadata.title or '(none)'}")
    print(f"  Pages  : {result.metadata.page_count}")
    print(f"  OCR    : {result.metadata.has_ocr}")
    outline = result.hierarchy.to_outline()
    if outline:
        print(f"  Outline: {outline.splitlines()[0]} …")


def main() -> None:
    # ── 1. Digital PDF ─────────────────────────────────────────────────────────

    show("Digital PDF", SAMPLE / "financial_report.pdf")

    # ── 2. Scanned PDF — OCR is applied automatically ──────────────────────────

    show("Scanned PDF (OCR)", SAMPLE / "letter_scanned.pdf")

    # ── 3. HTML — local file ───────────────────────────────────────────────────

    show("HTML (local file)", SAMPLE / "sec_10q.html")

    # ── 4. DOCX ────────────────────────────────────────────────────────────────

    show("DOCX", SAMPLE / "infosec_policy.docx")

    # ── 5. PPTX ────────────────────────────────────────────────────────────────

    show("PPTX", SAMPLE / "financial_review.pptx")

    # ── 6. Non-SEC HTML URL ────────────────────────────────────────────────────

    show(
        "Non-SEC URL",
        "https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:32006L0112",
    )

    # ── 7. SEC EDGAR filing URL ────────────────────────────────────────────────

    show(
        "SEC EDGAR filing",
        "https://www.sec.gov/Archives/edgar/data/1561861/000119312525270444/d11281d424b1.htm",
    )

    # ── 8. URL that resolves to a PDF ──────────────────────────────────────────
    # docslicer follows the link and parses the underlying PDF directly.

    show(
        "URL → PDF",
        "https://www.gsk.com/media/hpgfxwxv/q1-2026-results-announcement.pdf",
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
