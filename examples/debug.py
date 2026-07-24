"""
debug.py — Dump all intermediate pipeline DataFrames as CSV for inspection.

Parses a document with debug=True, then writes one CSV per pipeline step
with every column intact — nothing filtered.

PDF steps:   words → shapes → cells → lines → table_cells → blocks → chunks
DOCX:        runs → paragraphs → lines → table_cells → blocks → chunks
PPTX:        runs → chart_points → paragraphs → lines → table_cells → blocks → chunks
HTML:        boxes → lines → table_cells → blocks → chunks

The work runs under `if __name__ == "__main__":`. docslicer parses CPU-bound
steps (PDF word extraction, cell building, OCR) across a process pool, and on
spawn-based platforms (macOS, Windows) each worker re-imports this file — without
the guard, that re-import would re-run the parse in every worker. A scanned PDF
(OCR) trips this reliably; a small digital PDF may resolve to a single worker and
slip through, but the guard is needed either way.

Usage:
    python examples/debug.py
    python examples/debug.py path/to/your/document.pdf
    python examples/debug.py path/to/your/document.pdf output/debug/
"""

import sys
from pathlib import Path
from time import perf_counter

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import docslicer
from docslicer._utils.df_export.export_debug import export_debug


def main() -> None:
    # ── Args ─────────────────────────────────────────────────────────────────
    SOURCE = sys.argv[1] if len(sys.argv) > 1 else Path(__file__).parent / "sample_docs" / "financial_report.pdf"
    OUT_DIR = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(__file__).parent / "output" / "debug"

    # ── Parse ────────────────────────────────────────────────────────────────

    print(f"Parsing: {SOURCE}")
    t0 = perf_counter()
    result = docslicer.parse_document(SOURCE, debug=True)
    elapsed = perf_counter() - t0
    print(f"Done in {elapsed:.2f}s — {len(result.chunks)} chunks  {len(result.blocks)} blocks  {len(result.tables)} tables\n")

    # ── Dump pipeline steps ──────────────────────────────────────────────────

    if not result.pipeline_steps:
        print("No pipeline steps recorded (debug=True required).")
        sys.exit(1)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"{'Step':<16}  {'Rows':>6}  {'Cols':>5}  File")
    print("-" * 70)

    for name, df in result.pipeline_steps.items():
        out_path = OUT_DIR / f"{name}.csv"
        export_debug(df).to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"{name:<16}  {len(df):>6}  {len(df.columns):>5}  {out_path}")

    # ── Column listing per step ──────────────────────────────────────────────

    print()
    for name, df in result.pipeline_steps.items():
        print(f"── {name} columns ({'  '.join(df.columns)})")

    print(f"\nAll CSVs written to: {OUT_DIR}/")


if __name__ == "__main__":
    main()
