# SPDX-FileCopyrightText: 2026 Market Framer Inc.
# SPDX-License-Identifier: AGPL-3.0-only
"""MCP (Model Context Protocol) server exposing docslicer to LLM clients.

Install with ``pip install docslicer[mcp]`` and run ``docslicer-mcp``.
"""

from __future__ import annotations

__all__ = ["main"]


def main() -> None:  # pragma: no cover - thin console-script shim
    from .server import main as _main

    _main()
