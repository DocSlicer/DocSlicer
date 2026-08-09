# SPDX-FileCopyrightText: 2026 Market Framer Inc.
# SPDX-License-Identifier: AGPL-3.0-only
"""Entry point for the MCPB bundle.

The `uv` server type names a Python *file*, not a console script, so this
forwards to the same `main()` that the `docslicer-mcp` command runs. Keep it
free of logic: the host runs it with no arguments, so the server takes its
defaults (stdio transport) and reads the rest from the environment that
`mcp_config.env` supplies.
"""

from __future__ import annotations

from docslicer.mcp.server import main

if __name__ == "__main__":
    main()
