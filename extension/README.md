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

`pyproject.toml` declares `docslicer[mcp,html]>=0.2.1` and uv resolves it from
PyPI at install time. No wheel is vendored — `.mcpbignore` excludes `*.whl`, and
the `uv` server type forbids shipping a resolved environment.

`uv.lock` ships in the archive and is what makes installs reproducible: it pins
`docslicer` and every transitive dependency to an exact version and hash, so the
open-ended `>=0.2.1` in `pyproject.toml` never floats. Regenerate it whenever the
version changes, and keep it committed.

## Build

`docslicer` X.Y.Z must be on PyPI *before* the bundle is packed — the lock
resolves against the registry.

```bash
cd extension
uv lock                                 # pin the new release by hash
mcpb pack . ../dist/docslicer-X.Y.Z.mcpb
```

The archive holds six files and is ~500 kB, almost all of it `uv.lock`:

```
README.md  icon-512.png  manifest.json  pyproject.toml  src/server.py  uv.lock
```

## Install

Users install a released `.mcpb` any of three ways:

- **Double-click** the `.mcpb` file
- **Drag and drop** it onto the Claude Desktop window
- **Settings** → Extensions → Advanced settings → Install Extension… → pick the file

For a locally built bundle, use Settings → Extensions → Advanced settings →
Extension Developer → "Install Extension…".

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

## Release checklist

1. Bump `version` in the root `pyproject.toml`, `extension/pyproject.toml`, and
   `extension/manifest.json` — all three must match
2. Publish `docslicer` X.Y.Z to PyPI
3. `cd extension && uv lock` — must resolve from
   `registry = "https://pypi.org/simple"`, never a local path. Commit the result.
4. `mcpb pack . ../dist/docslicer-X.Y.Z.mcpb`
5. Verify the packed lock before releasing:

   ```bash
   unzip -p ../dist/docslicer-X.Y.Z.mcpb uv.lock | grep -A2 '^name = "docslicer"$'
   ```

6. Tag, then attach the `.mcpb` to the GitHub release:

   ```bash
   gh release create vX.Y.Z ../dist/docslicer-X.Y.Z.mcpb
   ```

7. Install the released file from its download URL on a machine **without `uv`**
   before announcing
