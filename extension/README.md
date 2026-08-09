# DocSlicer desktop extension (.mcpb)

Packages the MCP server for one-click install in Claude Desktop / Cowork.

## Local testing build (no PyPI)

The bundled manifest points `uvx` at a wheel shipped **inside** the archive, so
nothing needs to be published to test it.

```bash
python -m build                       # produces dist/docslicer-X.Y.Z-py3-none-any.whl
cp dist/docslicer-X.Y.Z-py3-none-any.whl extension/
cd extension && mcpb pack . ../dist/docslicer-X.Y.Z.mcpb
```

Install: Claude Desktop → Settings → Extensions → Advanced settings →
Extension Developer → "Install Extension…" → pick the `.mcpb`.

Requires `uv` on the machine (`brew install uv`). Claude Desktop bundles Node,
not Python, so the uvx path is what makes this cross-platform.

## HTML rendering

`uvx` resolves the extras in `--from` into a fresh, isolated environment — a
Playwright installed anywhere else on the machine is invisible to the server.
That is why the manifest asks for `[mcp,html]` and not just `[mcp]`.

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

Once the package is on PyPI, drop the bundled wheel and change the manifest's
`--from` argument to `docslicer[mcp,html]` so users get updates from PyPI:

```jsonc
"args": ["--from", "docslicer[mcp,html]", "docslicer-mcp"]
```
