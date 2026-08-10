# SPDX-FileCopyrightText: 2026 Market Framer Inc.
# SPDX-License-Identifier: AGPL-3.0-only
"""Command-line entry point: parse a file path or URL to structured JSON on stdout."""

import argparse
import json
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        prog="docslicer",
        description="Parse and chunk documents into structured JSON output.",
    )
    parser.add_argument("source", help="File path or URL to parse")
    parser.add_argument(
        "--output", "-o",
        help="Write output to this file instead of stdout",
    )
    parser.add_argument(
        "--max-chunk-size", type=int, default=3200,
        help="Maximum characters per chunk (default: 3200)",
    )
    parser.add_argument(
        "--optimal-chunk-size", type=int, default=1500,
        help="Target characters per chunk (default: 1500)",
    )
    parser.add_argument(
        "--min-chunk-size", type=int, default=700,
        help="Soft minimum characters per chunk (default: 700)",
    )
    parser.add_argument(
        "--table-representation",
        choices=["markdown", "jsonl", "melted"],
        default="markdown",
        help="How tables are serialized into chunk text (default: markdown)",
    )
    parser.add_argument(
        "--no-chunking", action="store_true",
        help="Skip chunking and return only blocks (faster)",
    )
    parser.add_argument(
        "--no-merge", action="store_true",
        help="Disable merging of small chunks",
    )
    parser.add_argument(
        "--exact-tokens", action="store_true",
        help="Compute exact token counts instead of the fast estimate",
    )
    parser.add_argument(
        "--extra-fields", nargs="+", default=None,
        help="Extra pipeline fields to surface on each chunk/block",
    )
    parser.add_argument(
        "--password",
        help="Password for an encrypted document",
    )
    parser.add_argument(
        "--max-workers", type=int, default=None,
        help="Process-pool width for CPU-bound steps (default: auto)",
    )
    parser.add_argument(
        "--no-browser", action="store_true",
        help="HTML only: skip Playwright and use the static box extractor",
    )
    parser.add_argument(
        "--include-headers-footers", action="store_true",
        help="DOCX only: surface header/footer content as blocks",
    )
    parser.add_argument(
        "--no-footnotes", action="store_true",
        help="DOCX only: exclude footnotes/endnotes",
    )
    parser.add_argument(
        "--include-comments", action="store_true",
        help="DOCX only: include reviewer comments",
    )
    parser.add_argument(
        "--no-speaker-notes", action="store_true",
        help="PPTX only: exclude speaker notes",
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Include pipeline debug info in output",
    )
    args = parser.parse_args()

    from docslicer import parse_document

    try:
        result = parse_document(
            args.source,
            max_chunk_size=args.max_chunk_size,
            optimal_chunk_size=args.optimal_chunk_size,
            min_chunk_size=args.min_chunk_size,
            table_representation=args.table_representation,
            chunking=not args.no_chunking,
            merge_small_chunks=not args.no_merge,
            exact_tokens=args.exact_tokens,
            extra_fields=args.extra_fields,
            password=args.password,
            max_workers=args.max_workers,
            use_browser=not args.no_browser,
            include_headers_footers=args.include_headers_footers,
            include_footnotes=not args.no_footnotes,
            include_comments=args.include_comments,
            include_speaker_notes=not args.no_speaker_notes,
            debug=args.debug,
        )
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    # --no-chunking stops the pipeline before chunks exist, so blocks are the
    # unit it has to return; serializing chunks regardless printed an empty
    # list for the one flag whose whole purpose is to produce output faster.
    records = result.blocks if args.no_chunking else result.chunks
    label = "blocks" if args.no_chunking else "chunks"
    out = json.dumps([r.to_dict() for r in records], indent=2)

    if args.output:
        Path(args.output).write_text(out, encoding="utf-8")
        print(f"Wrote {len(records)} {label} to {args.output}")
    else:
        print(out)
