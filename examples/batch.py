"""
batch.py — Process multiple documents efficiently.

Shows two approaches:
  - DocumentParser: reusable parser with a fixed config, avoids re-initialising
    on each call (useful in long-running processes or pipelines).
  - parse_all(): iterate over a folder, with per-file error handling.

Usage:
    python examples/batch.py
    python examples/batch.py path/to/folder/
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import docslicer
from docslicer import DocumentParser, ParseConfig

FOLDER = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / "sample_docs"

# ── DocumentParser — reusable config ──────────────────────────────────────────
# Create once, call .parse() many times. The parser holds its configuration
# so you don't repeat ParseConfig arguments across calls.

parser = DocumentParser(ParseConfig(
    max_chunk_size=1500,
    optimal_chunk_size=600,
))

print(f"── DocumentParser  (folder: {FOLDER})")
for path in sorted(FOLDER.glob("*.*")):
    if path.suffix.lower() in {".pdf", ".docx", ".pptx", ".html", ".htm"}:
        result = parser.parse(path)
        print(f"  {path.name:40s}  {len(result.chunks):>4} chunks  {len(result.tables):>2} tables")

# ── parse_all() — folder iteration with error handling ────────────────────────
# Yields (path, ParseResult | Exception) for every supported file in the folder.
# Failed files surface as exceptions instead of stopping the loop.

print(f"\n── parse_all  (folder: {FOLDER})")
for path, result in docslicer.parse_all(FOLDER):
    if isinstance(result, Exception):
        print(f"  FAILED  {path.name}: {result}")
    else:
        print(f"  OK      {path.name:40s}  {len(result.chunks):>4} chunks  {len(result.tables):>2} tables")

print("\nDone.")
