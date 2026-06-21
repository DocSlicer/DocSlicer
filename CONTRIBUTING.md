# Contributing to Docslicer

Thanks for your interest in improving Docslicer! Contributions of all kinds
are welcome — bug reports, feature requests, documentation, and code.

## Ways to contribute

- **Report a bug** — open an issue with a minimal reproduction (a small input
  file or snippet), the version you're on, and what you expected vs. saw.
- **Suggest a feature** — open an issue describing the use case before writing
  code, so we can agree on the approach.
- **Submit code** — fork the repo, create a branch, and open a pull request.

## Development setup

```bash
git clone https://github.com/<your-org>/docslicer.git
cd docslicer
python -m venv .venv && source .venv/bin/activate
pip install -e '.[dev]'
pytest
```

## Pull request guidelines

- Keep PRs focused — one logical change per PR.
- Add or update tests for any behavior change; `pytest` must pass.
- Run the formatter/linter before pushing (e.g. `ruff` / `black`).
- Update relevant docs and docstrings.
- Describe *what* changed and *why* in the PR description, and link any
  related issue.

## Contributor License Agreement (CLA) — required

Docslicer is **dual-licensed** (AGPL v3.0 for the community, plus a separate
commercial license — see [COMMERCIAL-LICENSE.md](./COMMERCIAL-LICENSE.md)).
To keep that model possible, the project must hold the rights to license
**all** of the code, including your contribution, under both licenses.

Because of this, **we can only accept contributions from people who have
agreed to our Contributor License Agreement.** The CLA does not take your
copyright away — you keep ownership of your work — it simply grants the
maintainers a broad license to use and relicense your contribution
(including commercially). See [CLA.md](./CLA.md) for the full text.

### How to sign

By opening a pull request, you confirm that you have read and agree to the
[CLA](./CLA.md). For your **first** contribution, please also add the
following line to your PR description (replacing the details):

```
I have read and agree to the Docslicer CLA.
Signed-off-by: Your Name <your.email@example.com>
```

> If you are contributing **on behalf of your employer**, please make sure
> you have authority to do so, or have your employer sign the corporate
> version of the CLA — otherwise your employer, not you, may own the code.

Contributions without an agreed CLA cannot be merged. If you'd prefer not to
sign, you're still very welcome to open issues and help with discussions.

## Questions

Open an issue or reach out at **[your email]**.
