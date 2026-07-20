"""Guard against code packages silently dropping out of the built wheel.

pyproject uses `[tool.setuptools.packages.find]` (regular packages), which
excludes any directory lacking an `__init__.py` — and prunes its subpackages
too. A missing `__init__.py` therefore ships a wheel that imports fine until it
hits the missing submodule at runtime. This test fails loudly instead.
"""
from pathlib import Path

PKG_ROOT = Path(__file__).resolve().parent.parent / "src" / "docslicer"

# config/ is data-only (yaml/csv), shipped via [tool.setuptools.package-data],
# so it intentionally has no __init__.py and is not an importable package.
DATA_ONLY_DIRS = {PKG_ROOT / "config"}


def _dirs_with_python_modules():
    """Directories under the package that contain importable .py modules."""
    for py in PKG_ROOT.rglob("*.py"):
        if "__pycache__" in py.parts:
            continue
        yield py.parent


def test_every_code_dir_has_init():
    missing = sorted(
        str(d.relative_to(PKG_ROOT.parent))
        for d in set(_dirs_with_python_modules())
        if d not in DATA_ONLY_DIRS and not (d / "__init__.py").exists()
    )
    assert not missing, (
        "Directory contains .py modules but no __init__.py, so it is dropped "
        f"from the built wheel: {missing}"
    )
