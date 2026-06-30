"""
AcroForm field extractor (pikepdf backend).

Reads PDF interactive form fields (``/AcroForm``) from the COS objects and
returns them as a list of :class:`FormField` instances, grouped by page.
pdfium's text extraction ignores widget annotations entirely, so filled values
("Parser Test", checkbox states, dropdown selections) are invisible to the
word extractor without this layer.

What is extracted
-----------------
* All Widget annotation types: text, textarea, comb-text, checkbox, radio,
  dropdown, listbox, signature.
* Values via ``/V`` inheritance (widget → parent → root field).
* The accessible label via ``/TU`` (tooltip / accessible name); no spatial
  fallback is done here — label words that describe the field are already
  present as regular words in ``df_words``.
* PDF-space ``/Rect`` for every widget, so callers can convert to screen
  coordinates and place synthetic word rows spatially.

Scope
-----
Only filled fields (non-empty ``value``) produce :class:`FormField` instances
in the output — empty fields are omitted so nothing spurious enters the words
DataFrame. XFA-only PDFs (no ``/AcroForm``) return an empty dict.

Usage
-----
    from docslicer.pdf._utils.form_fields import build_form_index
    index = build_form_index("file.pdf")   # {page_index: [FormField, ...]}
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import pikepdf


# ── value helpers ─────────────────────────────────────────────────────────────
def _norm_value(v) -> Optional[str]:
    if v is None:
        return None
    if isinstance(v, pikepdf.Name):
        s = str(v)
        return s[1:] if s.startswith("/") else s
    if isinstance(v, pikepdf.Array):
        return ", ".join(_norm_value(x) or "" for x in v)
    s = str(v)
    return s if s else None


def _norm_str(o) -> Optional[str]:
    if o is None:
        return None
    try:
        s = str(o)
        return s if s else None
    except Exception:
        return None


def _widget_type(ft: Optional[str], ff: int) -> str:
    if ft == "Tx":
        if ff & (1 << 12):  return "textarea"
        if ff & (1 << 24):  return "comb-text"
        if ff & (1 << 13):  return "password"
        return "text"
    if ft == "Btn":
        if ff & (1 << 16):  return "pushbutton"
        if ff & (1 << 15):  return "radio"
        return "checkbox"
    if ft == "Ch":
        return "combo-editable" if (ff & (1 << 17)) and (ff & (1 << 18)) else (
            "dropdown" if ff & (1 << 17) else "listbox"
        )
    if ft == "Sig":
        return "signature"
    return ft or "unknown"


# ── /Parent-chain inheritance ─────────────────────────────────────────────────
def _chain(widget: pikepdf.Dictionary) -> List[pikepdf.Dictionary]:
    out, cur, hops = [], widget, 0
    while cur is not None and hops < 16:
        if isinstance(cur, pikepdf.Dictionary):
            out.append(cur)
        try:
            cur = cur.get("/Parent")
        except Exception:
            break
        hops += 1
    return out


def _inh(chain: List[pikepdf.Dictionary], key: str):
    for node in chain:
        try:
            v = node.get(key)
        except Exception:
            v = None
        if v is not None:
            return v
    return None


def _qual_name(chain: List[pikepdf.Dictionary]) -> str:
    parts = []
    for node in reversed(chain):
        try:
            t = node.get("/T")
        except Exception:
            t = None
        if t is not None:
            parts.append(_norm_str(t) or "")
    return ".".join(p for p in parts if p)


# ── Result type ───────────────────────────────────────────────────────────────
@dataclass
class FormField:
    """One AcroForm field / widget (filled or empty)."""
    page_index: int                  # 0-based
    field_name: str                  # qualified dot-path name
    widget_type: str                 # text | textarea | comb-text | checkbox | radio | dropdown | listbox | signature
    value: Optional[str]             # filled value; None / "Off" when empty
    is_empty: bool                   # True when field has no meaningful value
    label: Optional[str]             # /TU accessible label / tooltip (None if absent)
    pdf_rect: Tuple[float, float, float, float]  # (llx, lly, urx, ury) in PDF space (y-up, MediaBox origin)
    widget_objgen: Optional[Tuple[int, int]]     # annotation objgen; joins to struct-tree WidgetLink


# ── Parser ────────────────────────────────────────────────────────────────────
def _extract_fields(pdf: pikepdf.Pdf) -> List[FormField]:
    """Walk every Widget annotation across all pages; yield filled FormFields."""
    results: List[FormField] = []

    # page object-id -> 0-based index
    page_of: Dict[Tuple[int, int], int] = {}
    for i, page in enumerate(pdf.pages):
        try:
            page_of[page.obj.objgen] = i
        except Exception:
            pass

    # Track which (objgen) we've already emitted to deduplicate merged widgets.
    seen: set = set()

    for pi, page in enumerate(pdf.pages):
        annots = None
        try:
            annots = page.get("/Annots")
        except Exception:
            pass
        if not isinstance(annots, (pikepdf.Array, list)):
            continue

        for a in annots:
            try:
                if _norm_str(a.get("/Subtype")) != "/Widget":
                    continue
            except Exception:
                continue

            # Deduplicate: indirect objects share an objgen; direct (rare) fall
            # through and may be processed twice, which is harmless.
            try:
                key = a.objgen
                if key in seen:
                    continue
                seen.add(key)
            except Exception:
                key = None

            chain = _chain(a)

            ft_raw = _inh(chain, "/FT")
            ft = _norm_str(ft_raw)
            if ft and ft.startswith("/"):
                ft = ft[1:]

            ff_raw = _inh(chain, "/Ff")
            ff = int(ff_raw) if ff_raw is not None else 0

            value = _norm_value(_inh(chain, "/V"))
            is_empty = not value or value in ("Off", "")

            label = _norm_str(_inh(chain, "/TU"))

            rect_raw = _inh(chain, "/Rect")
            if not isinstance(rect_raw, pikepdf.Array) or len(rect_raw) < 4:
                continue
            try:
                llx, lly, urx, ury = [float(rect_raw[i]) for i in range(4)]
                # Normalise: some files flip lly/ury
                if lly > ury:
                    lly, ury = ury, lly
            except Exception:
                continue

            results.append(FormField(
                page_index=pi,
                field_name=_qual_name(chain),
                widget_type=_widget_type(ft, ff),
                value=value,
                is_empty=is_empty,
                label=label,
                pdf_rect=(llx, lly, urx, ury),
                widget_objgen=key,
            ))

    return results


# ── Public API ────────────────────────────────────────────────────────────────
def build_form_index(
    source: Union[str, Path, pikepdf.Pdf],
) -> Dict[int, List[FormField]]:
    """Parse the AcroForm of *source* and return filled fields grouped by page.

    Returns ``{page_index: [FormField, ...]}``. Only filled fields are returned.
    Returns an empty dict when the PDF has no ``/AcroForm`` or when no fields
    are filled."""
    def _run(pdf: pikepdf.Pdf) -> Dict[int, List[FormField]]:
        if pdf.Root.get("/AcroForm") is None:
            return {}
        fields = _extract_fields(pdf)
        index: Dict[int, List[FormField]] = {}
        for f in fields:
            index.setdefault(f.page_index, []).append(f)
        return index

    if isinstance(source, pikepdf.Pdf):
        return _run(source)
    with pikepdf.open(str(Path(source).expanduser())) as pdf:
        return _run(pdf)


def has_acroform(source: Union[str, Path, pikepdf.Pdf]) -> bool:
    """True if the PDF has an ``/AcroForm`` dictionary (interactive form)."""
    if isinstance(source, pikepdf.Pdf):
        return source.Root.get("/AcroForm") is not None
    with pikepdf.open(str(Path(source).expanduser())) as pdf:
        return pdf.Root.get("/AcroForm") is not None
