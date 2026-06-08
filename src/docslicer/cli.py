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
            debug=args.debug,
        )
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    out = json.dumps([c.to_dict() for c in result.chunks], indent=2)

    if args.output:
        Path(args.output).write_text(out, encoding="utf-8")
        print(f"Wrote {len(result.chunks)} chunks to {args.output}")
    else:
        print(out)
