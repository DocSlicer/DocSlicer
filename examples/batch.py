"""
batch.py — Process multiple documents, and the two axes of concurrency.

docslicer has two independent parallelism knobs:

  * ParseConfig.max_workers — intra-document. Fans a single document's
    CPU-bound steps (PDF word extraction, cell building, OCR) across a process
    pool. None (default) auto-sizes to the machine's performance cores; set 1
    to keep a document single-process.

  * DocumentParser(workers=N) — inter-document. Fans whole documents across N
    worker processes. Each worker builds its own DocumentParser (and its own
    browser, lazily, if it hits HTML).

They compose, so it's easy to oversubscribe the machine with nested process
pools. To avoid that, when you set workers=N a worker's own config.max_workers
defaults to 1 unless you picked a value explicitly. Rule of thumb: parse many
small docs → inter-document (workers=N); parse a few large docs → intra-document
(max_workers=None) one at a time.

Note the `if __name__ == "__main__":` guard below — DocumentParser(workers=N)
uses a ProcessPoolExecutor, and on spawn-based platforms (macOS, Windows) each
worker re-imports this file. Without the guard, that re-import would re-launch
the pool in every worker. Any script that uses workers=N needs it.

This file shows three approaches:
  1. DocumentParser — reusable config + shared browser, sequential.
  2. DocumentParser(workers=N) — same, but fanned across processes.
  3. parse_all() — one-shot folder iteration with per-file error handling.

Usage:
    python examples/batch.py
    python examples/batch.py path/to/folder/
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import docslicer
from docslicer import DocumentParser, ParseConfig

SUPPORTED = {".pdf", ".docx", ".pptx", ".html", ".htm", ".xhtml"}


def name_of(source) -> str:
    return Path(str(source)).name


def main() -> None:
    folder = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / "sample_docs"
    paths = sorted(p for p in folder.glob("*.*") if p.suffix.lower() in SUPPORTED)

    config = ParseConfig(max_chunk_size=1500, optimal_chunk_size=1000)

    # ── 1. DocumentParser — reusable config, sequential ───────────────────────
    # Create once, parse many. The parser holds its configuration (so you don't
    # repeat ParseConfig args) and keeps a single browser open across HTML
    # inputs. Use it as a context manager so that browser is released at the end.

    print(f"── DocumentParser — sequential  (folder: {folder})")
    with DocumentParser(config) as parser:
        for path in paths:
            result = parser.parse(path)
            print(f"  {path.name:40s}  {len(result.chunks):>4} chunks  {len(result.tables):>2} tables")

    # ── 2. DocumentParser(workers=N) — documents fanned across processes ──────
    # parse_all() yields (source, ParseResult | Exception), so a failed file
    # never aborts the batch. With workers set, results arrive once the whole
    # batch is scheduled (not lazily per-document). Each worker's
    # config.max_workers defaults to 1 here, since we didn't set it — that
    # avoids nested pools oversubscribing cores.

    print(f"\n── DocumentParser — workers=4")
    with DocumentParser(config, workers=4) as parser:
        for source, result in parser.parse_all(paths):
            if isinstance(result, Exception):
                print(f"  FAILED  {name_of(source)}: {result}")
            else:
                print(f"  OK      {name_of(source):40s}  {len(result.chunks):>4} chunks  {len(result.tables):>2} tables")

    # ── 3. parse_all() — one-shot folder iteration ────────────────────────────
    # No parser to hold: discovers every supported file in a folder and yields
    # (path, ParseResult | Exception). Good for a quick single pass; pass
    # recursive=True to descend subdirectories, plus any parse_document kwargs.

    print(f"\n── parse_all  (folder: {folder})")
    for path, result in docslicer.parse_all(folder):
        if isinstance(result, Exception):
            print(f"  FAILED  {path.name}: {result}")
        else:
            print(f"  OK      {path.name:40s}  {len(result.chunks):>4} chunks  {len(result.tables):>2} tables")

    print("\nDone.")


if __name__ == "__main__":
    main()
