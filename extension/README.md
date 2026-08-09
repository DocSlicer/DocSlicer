# DocSlicer desktop extension (.mcpb)

Packages the MCP server for one-click install in Claude Desktop / Cowork.

## How the runtime is resolved

`server.type` is `uv` (manifest 0.4+). Dependencies are declared in
`pyproject.toml` and installed by the host with uv; no `server/lib` and no
`server/venv` ship in the archive. The user needs no Python of their own — uv
provisions an interpreter satisfying `requires-python`.

This replaces an earlier `type: python` manifest that invoked `uvx` against a
bundled wheel. That approach depended on `uv` being on the PATH of a GUI
process, which on macOS does not include `/opt/homebrew/bin`, so it worked from
a terminal and failed when Claude Desktop spawned it. The hand-written `PATH`
in `mcp_config.env` was a workaround for exactly that, and is gone.

`mcp_config` is retained even though the spec calls it optional for this type:
the v0.4 schema lists it in `server.required` and `command` in its own
`required`, so a manifest without it fails validation. It is also what maps
`user_config` into `DOCSLICER_MCP_ROOT` and `DOCSLICER_MCP_ALLOW_URLS`.

`--project ${__dirname}` is load-bearing. `uv run <script>` alone resolves
against the working directory, which the host sets to something other than the
extension folder, so `pyproject.toml` would be ignored and the import of
`docslicer` would fail.

## Dependency source

PyPI currently has 0.2.0 only, while `pyproject.toml` pins `>=0.2.1`, so the
bundle ships the 0.2.1 wheel and `[tool.uv.sources]` points at it with a path
relative to `pyproject.toml` — which resolves inside the installed extension
directory on any machine. `uv.lock` is packed alongside it so installs are
reproducible.

Once 0.2.1 is on PyPI: delete the `[tool.uv.sources]` table, drop the wheel
from `extension/`, re-add `*.whl` to `.mcpbignore`, regenerate `uv.lock`, and
repack. The bundle then falls back to ~20 kB.

## Build

```bash
python -m build                        # refresh dist/docslicer-X.Y.Z-py3-none-any.whl
cp dist/docslicer-X.Y.Z-py3-none-any.whl extension/
cd extension && mcpb pack . ../dist/docslicer-X.Y.Z.mcpb
```

Install: Claude Desktop → Settings → Extensions → Advanced settings →
Extension Developer → "Install Extension…" → pick the `.mcpb`.

Verify the archive contains `src/server.py` before installing — a bundle
missing it installs cleanly and then fails at spawn time with
`No such file or directory (os error 2)`:

```bash
unzip -l ../dist/docslicer-X.Y.Z.mcpb
```

Smoke-test the exact command the manifest runs, without Claude Desktop:

```bash
printf '{"jsonrpc":"2.0","id":0,"method":"initialize","params":{"protocolVersion":"2025-06-18","capabilities":{},"clientInfo":{"name":"t","version":"1"}}}\n' \
  | uv run --project extension extension/src/server.py
```

## HTML rendering

uv resolves the bundle's dependencies into a fresh, isolated environment — a
Playwright installed anywhere else on the machine is invisible to the server.
That is why `pyproject.toml` asks for `docslicer[mcp,html]` and not just
`[mcp]`.

The `html` extra installs the Playwright *package*; browsers are a separate
download, and which build satisfies Playwright is pinned to its version (1.62
wants `chromium-1234`), so an existing `~/…/ms-playwright` cache populated by a
different Playwright will not match. What usually saves the launch is the
`channel="chrome"` preference in `BrowserSession._ensure_browser` — it drives an
installed Google Chrome with nothing to download.

With neither Chrome nor matching browsers, the pipeline logs a warning and falls
back to the static extractor, so the extension still works; HTML fidelity drops
(no layout coordinates, no CSS-class styles). `parse` reports which path ran as
`renderer: "browser" | "static"`.

## Privacy Policy

Full policy: https://docslicer.ai/privacy

**What is collected.** DocSlicer collects nothing. The extension has no
telemetry, analytics, crash reporting, or usage tracking, and no account,
licence key, or registration. Anthropic operates Claude and handles the
conversation itself under its own privacy policy; DocSlicer is not a party to
that and receives nothing from it.

**How your documents are used.** Parsing runs entirely on your machine, in a
local Python process started by Claude Desktop. Document contents are used only
to produce the outline, text slices, search results, and markdown you ask for,
and are returned only to the Claude client that called the tool. Documents are
never uploaded to DocSlicer or to any third party. The `root_dir` setting you
choose at install time bounds which folder the server may read from and write
to.

**Where data is stored, and for how long.** Parsed results are cached on your
own disk, by default under `~/.cache/docslicer-mcp` and configurable with the
`DOCSLICER_MCP_CACHE` environment variable. The cache is pruned to a size
ceiling (2048 MB by default, set with `DOCSLICER_MCP_CACHE_MAX_MB`); apart from
that it persists until you delete it, and deleting the directory removes it
permanently with no copy retained elsewhere. `to_markdown` writes a `.md` file
where you tell it to and replaces any existing file at that path. Nothing else
is written outside the cache directory and the output path you supply.

**Network access and third parties.** No network request is made for a local
file. Requests leave your machine only when you pass an `http(s)` source:

- The URL you supply is fetched directly from that host. For an HTML page,
  Playwright may render it in a local headless browser, which also loads the
  subresources that page references, exactly as visiting it in a browser would.
- Requests to `sec.gov` send a `User-Agent` header identifying the client, as
  the SEC fair-access policy requires.

These hosts are third parties chosen by you, not by DocSlicer, and their own
policies govern what they log. Set `allow_urls` to `0` (or
`DOCSLICER_MCP_ALLOW_URLS=0`) to reject remote sources and restrict the server
to local files only.

**Contact.** Privacy questions: jelle@docslicer.ai. Issues:
https://github.com/DocSlicer/DocSlicer/issues

## Publishing build

Publish `docslicer` 0.2.1 to PyPI, then follow the cleanup in
[Dependency source](#dependency-source) and repack.
