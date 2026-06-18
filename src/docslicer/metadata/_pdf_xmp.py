"""Extract the raw XMP packet from a PDF by scanning its bytes."""
from __future__ import annotations

from pathlib import Path
from typing import Optional


def read_xmp(pdf_path: Path) -> Optional[str]:
    """
    Return the raw XMP XML string embedded in *pdf_path*, or None.

    Scans the raw bytes for the standard <?xpacket …?> delimiters.
    Works for uncompressed XMP streams (the overwhelming majority of PDFs).
    """
    try:
        with open(pdf_path, "rb") as f:
            data = f.read()
        start = data.find(b"<?xpacket begin=")
        if start == -1:
            return None
        end = data.find(b"<?xpacket end=", start)
        if end == -1:
            return None
        end = data.index(b"?>", end) + 2
        return data[start:end].decode("utf-8", errors="replace")
    except Exception:
        return None
