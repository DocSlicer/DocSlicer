# SPDX-FileCopyrightText: 2026 Market Framer Inc.
# SPDX-License-Identifier: AGPL-3.0-only
"""Document handle store backing the MCP server.

Parsing a document is expensive and its full ParseResult is far too large to
hand to a language model in one response. The server therefore parses once,
persists the result to a cache directory, and returns a short ``doc_id``
handle. Follow-up tools re-hydrate the ParseResult from memory (hot) or disk
(cold) and return only the slice the caller asked for.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import re
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

from .._result import ParseResult

_MEMORY_LIMIT = 8  # ParseResults kept hot in RAM

_CACHE_LIMIT_MB = 2048  # Disk the persisted ParseResults may occupy, before pruning


def cache_dir() -> Path:
    """Directory holding persisted ParseResults. Override with DOCSLICER_MCP_CACHE."""
    env = os.environ.get("DOCSLICER_MCP_CACHE")
    base = Path(env) if env else Path.home() / ".cache" / "docslicer-mcp"
    base.mkdir(parents=True, exist_ok=True)
    return base


def cache_limit_bytes() -> int:
    """Cache size ceiling in bytes. Override with DOCSLICER_MCP_CACHE_MAX_MB; 0 disables pruning."""
    env = os.environ.get("DOCSLICER_MCP_CACHE_MAX_MB")
    try:
        megabytes = int(env) if env else _CACHE_LIMIT_MB
    except ValueError:
        megabytes = _CACHE_LIMIT_MB
    return max(0, megabytes) * 1024 * 1024


def allowed_root() -> Path | None:
    """Optional sandbox root. When DOCSLICER_MCP_ROOT is set, only files beneath it parse."""
    env = os.environ.get("DOCSLICER_MCP_ROOT")
    return Path(env).expanduser().resolve() if env else None


def urls_allowed() -> bool:
    """Whether URL sources are accepted. Disable with DOCSLICER_MCP_ALLOW_URLS=0."""
    return os.environ.get("DOCSLICER_MCP_ALLOW_URLS", "1").lower() not in ("0", "false", "no")


class SourceNotAllowed(ValueError):
    """Raised when a requested source falls outside the configured sandbox."""


class UnknownDocument(KeyError):
    """Raised when a doc_id has no entry in memory or on disk."""

    def __str__(self) -> str:  # KeyError repr adds quotes; drop them
        return self.args[0] if self.args else ""


def _slug(value: str) -> str:
    stem = Path(value).stem or "doc"
    stem = re.sub(r"[^a-zA-Z0-9]+", "-", stem).strip("-").lower()
    return (stem[:32] or "doc")


def is_url(value: str) -> bool:
    return value.startswith(("http://", "https://"))


def resolve_source(source: str) -> str:
    """Validate a source string against the sandbox settings.

    Returns the source unchanged for URLs, or the resolved absolute path for
    files. Raises SourceNotAllowed / FileNotFoundError otherwise.
    """
    if is_url(source):
        if not urls_allowed():
            raise SourceNotAllowed(
                "URL sources are disabled on this server (DOCSLICER_MCP_ALLOW_URLS=0)."
            )
        return source

    path = Path(source).expanduser()
    try:
        resolved = path.resolve(strict=True)
    except FileNotFoundError:
        raise FileNotFoundError(f"No such file: {source}") from None

    root = allowed_root()
    if root is not None and root not in resolved.parents and resolved != root:
        raise SourceNotAllowed(
            f"Access to {resolved} is outside the allowed root {root}. "
            "Copy the file inside that directory or adjust DOCSLICER_MCP_ROOT."
        )
    return str(resolved)


def resolve_output(path: str) -> Path:
    """Validate a destination the server is about to write to, against the sandbox.

    The counterpart to ``resolve_source`` for the write direction: a root that
    restricts what may be read but lets a caller write anywhere on the machine
    is not a sandbox. Kept separate because the two validate opposite things —
    a source must already exist, a destination must not have to.

    Only the existing part of the path can be resolved, so the check lands on
    the nearest ancestor that exists. That is the boundary symlinks are
    resolved at, and where an escape would have to be built.
    """
    target = Path(path).expanduser()
    resolved = Path(target if target.is_absolute() else Path.cwd() / target).resolve()

    root = allowed_root()
    if root is not None and root not in resolved.parents:
        raise SourceNotAllowed(
            f"Cannot write to {resolved}: it is outside the allowed root {root}. "
            "Choose an output_path inside that directory or adjust DOCSLICER_MCP_ROOT."
        )
    return resolved


def make_doc_id(source: str, options: dict) -> str:
    """Stable handle for a (source, options, file-version) triple."""
    fingerprint = {"source": source, "options": options}
    if not is_url(source):
        try:
            stat = Path(source).stat()
            fingerprint["size"] = stat.st_size
            fingerprint["mtime"] = int(stat.st_mtime)
        except OSError:
            pass
    blob = json.dumps(fingerprint, sort_keys=True, default=str).encode("utf-8")
    return f"{_slug(source)}-{hashlib.sha256(blob).hexdigest()[:8]}"


@dataclass
class Entry:
    """Sidecar record describing one cached document."""

    doc_id: str
    source: str
    options: dict
    parsed_at: float
    chunks: int
    blocks: int
    tables: int
    charts: int
    pages: int | None
    title: str | None
    content_type: str

    def to_dict(self) -> dict:
        return {
            "doc_id": self.doc_id,
            "source": self.source,
            "options": self.options,
            "parsed_at": self.parsed_at,
            "counts": {
                "chunks": self.chunks,
                "blocks": self.blocks,
                "tables": self.tables,
                "charts": self.charts,
                "pages": self.pages,
            },
            "title": self.title,
            "content_type": self.content_type,
        }


class DocumentStore:
    """LRU of hot ParseResults over a JSON-on-disk cache."""

    def __init__(self) -> None:
        self._hot: "OrderedDict[str, ParseResult]" = OrderedDict()

    # -- paths ---------------------------------------------------------
    def _result_path(self, doc_id: str) -> Path:
        return cache_dir() / f"{doc_id}.json"

    def _meta_path(self, doc_id: str) -> Path:
        return cache_dir() / f"{doc_id}.meta.json"

    # -- writes --------------------------------------------------------
    def put(self, doc_id: str, result: ParseResult, source: str, options: dict) -> Entry:
        self._result_path(doc_id).write_text(result.to_json(indent=None), encoding="utf-8")
        entry = Entry(
            doc_id=doc_id,
            source=source,
            options=options,
            parsed_at=time.time(),
            chunks=len(result.chunks),
            blocks=len(result.blocks),
            tables=len(result.tables),
            charts=len(result.charts),
            pages=result.metadata.page_count,
            title=result.metadata.title,
            content_type=result.metadata.content_type,
        )
        self._meta_path(doc_id).write_text(
            json.dumps(entry.to_dict(), indent=2), encoding="utf-8"
        )
        self._remember(doc_id, result)
        self._prune(keep=doc_id)
        return entry

    def _prune(self, keep: str) -> None:
        """Drop the least recently parsed documents until the cache is under its limit.

        Nothing else evicts: a long-lived server parses documents forever and a
        single filing's serialised ParseResult runs to tens of megabytes, so
        without a ceiling the cache is an unbounded write to the user's disk.

        Size is the axis that matters rather than a document count — one filing
        outweighs a hundred memos. Walks the result files rather than ``list()``
        so that a result whose sidecar failed to write is still pruned; such an
        orphan is invisible to every other read path and would otherwise be a
        leak that never comes back.
        """
        limit = cache_limit_bytes()
        if not limit:
            return

        entries: list[tuple[float, int, str]] = []
        for path in cache_dir().glob("*.json"):
            if path.name.endswith(".meta.json"):
                continue
            doc_id = path.stem
            try:
                size = path.stat().st_size
                age = path.stat().st_mtime
            except OSError:
                continue
            with contextlib.suppress(OSError):
                size += self._meta_path(doc_id).stat().st_size
            entries.append((age, size, doc_id))

        entries.sort(reverse=True)  # newest first; the tail is what goes
        total = 0
        for _, size, doc_id in entries:
            total += size
            # The document just parsed always survives, whatever it weighs — it
            # is the one the caller is about to read from.
            if total > limit and doc_id != keep:
                self.forget(doc_id)

    def _remember(self, doc_id: str, result: ParseResult) -> None:
        self._hot[doc_id] = result
        self._hot.move_to_end(doc_id)
        while len(self._hot) > _MEMORY_LIMIT:
            self._hot.popitem(last=False)

    # -- reads ---------------------------------------------------------
    def get(self, doc_id: str) -> ParseResult:
        if doc_id in self._hot:
            self._hot.move_to_end(doc_id)
            return self._hot[doc_id]

        path = self._result_path(doc_id)
        if not path.exists():
            known = ", ".join(e.doc_id for e in self.list()[:5]) or "none cached"
            raise UnknownDocument(
                f"Unknown doc_id {doc_id!r}. Call parse first. Known documents: {known}"
            )
        result = ParseResult.load(path)
        self._remember(doc_id, result)
        return result

    def find(self, source: str, options: dict) -> str | None:
        """Return an existing doc_id for this exact (source, options) pair, if cached."""
        doc_id = make_doc_id(source, options)
        return doc_id if self._result_path(doc_id).exists() else None

    def list(self) -> list[Entry]:
        entries: list[Entry] = []
        for meta in cache_dir().glob("*.meta.json"):
            try:
                d = json.loads(meta.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            counts = d.get("counts", {})
            entries.append(
                Entry(
                    doc_id=d.get("doc_id", meta.stem.removesuffix(".meta")),
                    source=d.get("source", ""),
                    options=d.get("options", {}),
                    parsed_at=d.get("parsed_at", 0.0),
                    chunks=counts.get("chunks", 0),
                    blocks=counts.get("blocks", 0),
                    tables=counts.get("tables", 0),
                    charts=counts.get("charts", 0),
                    pages=counts.get("pages"),
                    title=d.get("title"),
                    content_type=d.get("content_type", "unknown"),
                )
            )
        entries.sort(key=lambda e: e.parsed_at, reverse=True)
        return entries

    def forget(self, doc_id: str) -> bool:
        self._hot.pop(doc_id, None)
        removed = False
        for path in (self._result_path(doc_id), self._meta_path(doc_id)):
            if path.exists():
                path.unlink()
                removed = True
        return removed
