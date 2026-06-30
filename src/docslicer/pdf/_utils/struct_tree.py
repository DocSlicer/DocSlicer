"""
Structure-tree parser (pikepdf backend).

Parses a PDF's logical structure tree (``/StructTreeRoot``) directly from the COS
objects with **pikepdf**, independent of any word/geometry extraction. The output
is a mapping ``(page_index, mcid) -> StructInfo`` that step_01 joins onto words by
marked-content id.

Why pikepdf instead of pdfium for this layer
---------------------------------------------
pdfium's ``FPDF_StructElement_*`` API cannot expose attribute objects (``/A`` /
``/C``), so it can never return table ``ColSpan`` / ``RowSpan``, ``Headers`` or
``Scope`` — exactly the data table reconstruction needs. pikepdf reads the COS
tree directly, so it recovers:

  * ``/RoleMap`` and ``/ClassMap`` (custom tag -> standard tag; shared attrs),
  * PDF 2.0 namespaced role maps (``/RoleMapNS`` on each ``/Namespace``),
  * attribute dictionaries incl. ``ColSpan`` / ``RowSpan`` / ``Headers`` / ``Scope``,
  * tags pdfium drops on files with no ``/MarkInfo`` flag, without ever crashing.

Robustness goals
----------------
The parser is built to resolve *anything* that carries a structure tree:

  * Works on legacy (PDF 1.7 / ISO 32000-1) and modern (PDF 2.0 / ISO 32000-2)
    tag sets, and on MathML-namespaced sub-trees.
  * Custom tags are resolved through RoleMap/RoleMapNS when a mapping exists, and
    preserved verbatim (``raw_tag``) when it does not — nothing is ever dropped.
  * Cycle-safe tree walk (path-local visited set) and per-step cycle detection in
    role resolution; every COS access is defensive so a malformed object degrades
    to "skip this node" rather than raising.

Limitation
----------
MCIDs are unique per *content stream*, not per page. Content inside a form
XObject has its own MCID space (signalled by ``/Stm`` on the MCR). We key by
``(page_index, mcid)`` to match pdfium's per-word ``GetMarkedContentID`` (which
also collapses streams); the owning stream is recorded on ``StructInfo.stm`` so a
future stream-aware join can disambiguate. On the rare collision (two streams on
one page reusing an mcid) the first wins.

Usage (library)
---------------
    from docslicer.pdf._utils.struct_tree import build_struct_index
    index = build_struct_index("file.pdf")          # {(page, mcid): StructInfo}
    info = index.get((0, 3))
    if info: print(info.tag, info.col_span, info.ancestors)

Usage (CLI probe)
-----------------
    python -m docslicer.pdf._utils.struct_tree FILE.pdf [--page N]
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import pikepdf

# ── Standard structure namespaces (PDF 2.0, ISO 32000-2 §14.7.4) ──────────────
NS_PDF_1_7 = "http://iso.org/pdf/ssn"     # legacy default namespace
NS_PDF_2_0 = "http://iso.org/pdf2/ssn"    # PDF 2.0 standard structure namespace
NS_MATHML = "http://www.w3.org/1998/Math/MathML"

# Standard structure types for the PDF 1.7 namespace (ISO 32000-1 Table 333).
_STD_1_7 = frozenset({
    "Document", "Part", "Art", "Sect", "Div", "BlockQuote", "Caption", "TOC",
    "TOCI", "Index", "NonStruct", "Private", "H", "H1", "H2", "H3", "H4", "H5",
    "H6", "P", "L", "LI", "Lbl", "LBody", "Table", "TR", "TH", "TD", "THead",
    "TBody", "TFoot", "Span", "Quote", "Note", "Reference", "BibEntry", "Code",
    "Link", "Annot", "Figure", "Formula", "Form", "Ruby", "RB", "RT", "RP",
    "Warichu", "WT", "WP",
})

# Standard structure types added / retained in the PDF 2.0 namespace
# (ISO 32000-2 Table 364). Superset-ish; includes 1.7 types that survive plus the
# new ones (Title, Sub, Em, Strong, Aside, FENote, DocumentFragment, Artifact, …).
_STD_2_0 = frozenset({
    "Document", "DocumentFragment", "Part", "Div", "Aside", "Title", "Sub",
    "Sect", "Art", "BlockQuote", "Caption", "TOC", "TOCI", "Index", "NonStruct",
    "Private", "H", "H1", "H2", "H3", "H4", "H5", "H6", "P", "Em", "Strong",
    "L", "LI", "Lbl", "LBody", "Table", "TR", "TH", "TD", "THead", "TBody",
    "TFoot", "Span", "Quote", "Note", "Reference", "BibEntry", "Code", "Link",
    "Annot", "Figure", "Formula", "Form", "Ruby", "RB", "RT", "RP", "Warichu",
    "WT", "WP", "FENote", "Artifact",
})

# Union used purely as the "stop chasing the RoleMap chain" test. A generous set
# is the safe choice: standard types must never be remapped (a file that remaps
# e.g. /Sect is malformed), so stopping early on any recognized type cannot lose
# a legitimate mapping, while it does guard against rolemap cycles into junk.
_STANDARD = _STD_1_7 | _STD_2_0

# Heading variants beyond H6 (PDF 2.0 allows /Hn with arbitrary n) are normalized
# to themselves but recognized as standard so resolution stops.
def _is_standard(tag: Optional[str], ns_uri: Optional[str]) -> bool:
    if tag is None:
        return False
    if ns_uri == NS_MATHML:
        return True  # any MathML element name is a terminal (standard) tag
    if tag in _STANDARD:
        return True
    # PDF 2.0 numbered headings: H7, H8, …
    if len(tag) > 1 and tag[0] == "H" and tag[1:].isdigit():
        return True
    return False


# ── COS scalar helpers ────────────────────────────────────────────────────────
def _name(o: Any) -> Optional[str]:
    """A ``/Name`` (or anything name-ish) without the leading slash."""
    if o is None:
        return None
    try:
        s = str(o)
    except Exception:
        return None
    return s[1:] if s.startswith("/") else s


def _text(o: Any) -> Optional[str]:
    if o is None:
        return None
    try:
        return str(o)
    except Exception:
        return None


def _as_int(o: Any) -> Optional[int]:
    if isinstance(o, bool):
        return None
    if isinstance(o, int):
        return o
    try:
        f = float(o)
        return int(f)
    except Exception:
        return None


def _to_py(o: Any) -> Any:
    """COS value -> plain python (for attribute values)."""
    if isinstance(o, (bool, int, float, str)):
        return o
    if isinstance(o, pikepdf.Array):
        return [_to_py(x) for x in o]
    if isinstance(o, pikepdf.Dictionary):
        return {_name(k): _to_py(v) for k, v in o.items()}
    if isinstance(o, pikepdf.Name):
        return _name(o)
    if isinstance(o, pikepdf.String):
        return str(o)
    try:  # Integer / Real scalar
        f = float(o)
        return int(f) if f.is_integer() else round(f, 6)
    except Exception:
        return _text(o)


def _obj_key(o: Any) -> Tuple[int, int]:
    """Stable identity for cycle detection. Indirect objects use objgen; direct
    (inline) objects can't form reference cycles, so fall back to python id()."""
    try:
        og = o.objgen
        if og and og != (0, 0):
            return og
    except Exception:
        pass
    return (0, id(o) & 0x7FFFFFFF)


# ── Result type ───────────────────────────────────────────────────────────────
@dataclass
class StructInfo:
    """Resolved structure data for one marked-content leaf, keyed by (page, mcid)."""
    tag: Optional[str]               # role-resolved, namespace-aware tag (e.g. "H1")
    raw_tag: Optional[str]           # original /S verbatim (e.g. "CorporateHeader")
    ns: Optional[str]                # namespace URI of the leaf element, if any
    elem_id: int                     # global DFS id of the owning element
    rank: int                        # global DFS reading-order position
    ancestors: List[str]             # resolved tags root -> direct parent
    ancestor_ids: List[int]          # parallel DFS elem_ids for each ancestor
    attrs: Dict[str, Any]            # merged own attributes (/C classes then /A)
    chain_attrs: List[Dict[str, Any]] = field(default_factory=list)  # per-ancestor attrs
    actual_text: Optional[str] = None
    alt: Optional[str] = None
    lang: Optional[str] = None
    stm: Optional[Tuple[int, int]] = None  # owning content stream objgen (XObject case)

    # Convenience accessors for the high-value table attributes ----------------
    #
    # ColSpan/RowSpan/Headers/Scope are attributes of the owning TD/TH *element*,
    # which is usually an ancestor of the text leaf (the leaf is a P inside the
    # cell). We therefore search the ancestor chain from the leaf outward and take
    # the nearest occurrence — these keys are cell-scoped, so the nearest hit is
    # the enclosing cell.
    def _chain_get(self, key: str) -> Any:
        for a in reversed(self.chain_attrs):
            if key in a:
                return a[key]
        return self.attrs.get(key)

    @property
    def col_span(self) -> int:
        return _as_int(self._chain_get("ColSpan")) or 1

    @property
    def row_span(self) -> int:
        return _as_int(self._chain_get("RowSpan")) or 1

    @property
    def headers(self) -> List[str]:
        h = self._chain_get("Headers")
        if isinstance(h, list):
            return [str(x) for x in h]
        return [str(h)] if h is not None else []

    @property
    def scope(self) -> Optional[str]:
        s = self._chain_get("Scope")
        return _name(s) if s is not None else None


@dataclass
class WidgetLink:
    """Structure-tree link to a form widget annotation (an /OBJR leaf).

    ``elem_id`` is the DFS id of the structure element whose /K holds the OBJR —
    i.e. the form field's owning element. ``chain_ids`` is that element's full
    ancestor chain (root → owning element, inclusive). ``rank`` is the widget's
    position in the same global DFS reading-order counter used for MCID leaves
    (:attr:`StructInfo.rank`), so a field's label can be found as the nearest
    *preceding* text leaf by rank — the layout Acrobat/USCIS forms actually use,
    where the widget is a reading-order sibling of its prompt rather than nested
    inside it. ``chain_ids`` still enables the rarer nested-containment case."""
    elem_id: int
    chain_ids: List[int]
    page_index: Optional[int]
    rank: int


# ── The parser ────────────────────────────────────────────────────────────────
class _StructTreeParser:
    def __init__(self, pdf: pikepdf.Pdf):
        self.pdf = pdf
        self.root = pdf.Root.get("/StructTreeRoot")

        # page object -> 0-based index
        self._page_of: Dict[Tuple[int, int], int] = {}
        for i, page in enumerate(pdf.pages):
            try:
                self._page_of[page.obj.objgen] = i
            except Exception:
                pass

        self.role_map: Dict[str, str] = {}            # legacy 1.7 document RoleMap
        self.role_map_ns: Dict[Tuple[int, int], Dict[str, Any]] = {}  # nsdict-key -> RoleMapNS
        self.class_map: Dict[str, List[dict]] = {}    # class name -> [attr dicts]
        self._load_maps()

        self.index: Dict[Tuple[Optional[int], int], StructInfo] = {}
        # widget annotation objgen -> structural link to its owning element.
        self.widget_links: Dict[Tuple[int, int], WidgetLink] = {}
        self._elem_counter = 0
        self._rank_counter = 0

    # — maps ——————————————————————————————————————————————————————————————————
    def _load_maps(self) -> None:
        if self.root is None:
            return
        rm = self.root.get("/RoleMap")
        if isinstance(rm, pikepdf.Dictionary):
            for k, v in rm.items():
                key = _name(k)
                tgt = _name(v)
                if key and tgt:
                    self.role_map[key] = tgt

        cm = self.root.get("/ClassMap")
        if isinstance(cm, pikepdf.Dictionary):
            for k, v in cm.items():
                key = _name(k)
                if not key:
                    continue
                if isinstance(v, pikepdf.Array):
                    self.class_map[key] = [_to_py(x) for x in v if isinstance(x, pikepdf.Dictionary)]
                elif isinstance(v, pikepdf.Dictionary):
                    self.class_map[key] = [_to_py(v)]

        # PDF 2.0 namespaced role maps: each /Namespace dict may carry /RoleMapNS.
        for nsd in self._iter_namespaces():
            rmns = nsd.get("/RoleMapNS")
            if isinstance(rmns, pikepdf.Dictionary):
                table: Dict[str, Any] = {}
                for k, v in rmns.items():
                    key = _name(k)
                    if not key:
                        continue
                    # value is either a Name (same NS) or [Name, NamespaceDict]
                    if isinstance(v, pikepdf.Array) and len(v) >= 2:
                        table[key] = (_name(v[0]), self._ns_uri(v[1]))
                    else:
                        table[key] = (_name(v), None)
                self.role_map_ns[_obj_key(nsd)] = table

    def _iter_namespaces(self):
        if self.root is None:
            return
        arr = self.root.get("/Namespaces")
        if isinstance(arr, pikepdf.Array):
            for nsd in arr:
                if isinstance(nsd, pikepdf.Dictionary):
                    yield nsd

    @staticmethod
    def _ns_uri(nsd: Any) -> Optional[str]:
        if isinstance(nsd, pikepdf.Dictionary):
            return _text(nsd.get("/NS"))
        return None

    # — role resolution ————————————————————————————————————————————————————————
    def _resolve_role(self, tag: Optional[str], ns_dict: Any) -> Optional[str]:
        """Resolve *tag* to a standard type, honouring PDF 2.0 RoleMapNS first and
        the legacy document RoleMap second, with cycle detection. Custom tags with
        no mapping are returned verbatim."""
        if tag is None:
            return None
        cur = tag
        cur_ns_key = _obj_key(ns_dict) if isinstance(ns_dict, pikepdf.Dictionary) else None
        cur_ns_uri = self._ns_uri(ns_dict)
        seen: set = set()
        for _ in range(32):  # generous; cycle detection is the real guard
            if _is_standard(cur, cur_ns_uri):
                return cur
            step = (cur, cur_ns_key)
            if step in seen:
                return cur  # cycle — return where we are
            seen.add(step)

            nxt: Optional[str] = None
            nxt_ns_key = cur_ns_key
            nxt_ns_uri = cur_ns_uri
            # 1) namespaced role map of the current namespace (PDF 2.0)
            if cur_ns_key is not None and cur_ns_key in self.role_map_ns:
                mapped = self.role_map_ns[cur_ns_key].get(cur)
                if mapped is not None:
                    nxt, tgt_uri = mapped
                    if tgt_uri is not None:
                        nxt_ns_uri = tgt_uri
                        nxt_ns_key = self._ns_key_for_uri(tgt_uri)
            # 2) legacy document-level RoleMap
            if nxt is None:
                nxt = self.role_map.get(cur)
                nxt_ns_key = None  # legacy targets are PDF 1.7 namespace
                nxt_ns_uri = NS_PDF_1_7

            if nxt is None:
                return cur  # custom tag, no mapping — keep verbatim
            cur, cur_ns_key, cur_ns_uri = nxt, nxt_ns_key, nxt_ns_uri
        return cur

    def _ns_key_for_uri(self, uri: Optional[str]) -> Optional[Tuple[int, int]]:
        if uri is None:
            return None
        for nsd in self._iter_namespaces():
            if self._ns_uri(nsd) == uri:
                return _obj_key(nsd)
        return None

    # — attributes ————————————————————————————————————————————————————————————
    def _collect_attrs(self, elem: pikepdf.Dictionary) -> Dict[str, Any]:
        """Merge attributes for one element: /C class refs first, then /A overrides.
        Handles /A as a dict, a single attr dict, or an array interleaving attr
        dicts with revision-number integers (which are skipped)."""
        out: Dict[str, Any] = {}

        # /C : class name or array of class names -> ClassMap lookup
        c = elem.get("/C")
        if c is not None:
            names = [_name(c)] if not isinstance(c, pikepdf.Array) else [_name(x) for x in c]
            for cname in names:
                for d in self.class_map.get(cname, []):
                    if isinstance(d, dict):
                        out.update(d)

        # /A : attribute dict, or array of (dict | revision-int)
        a = elem.get("/A")
        if a is not None:
            if isinstance(a, pikepdf.Array):
                for item in a:
                    if isinstance(item, pikepdf.Dictionary):
                        out.update(_to_py(item))
                    # integers are revision numbers — ignore
            elif isinstance(a, pikepdf.Dictionary):
                out.update(_to_py(a))
        return out

    # — walk ——————————————————————————————————————————————————————————————————
    def _page_index(self, pg: Any) -> Optional[int]:
        if pg is None:
            return None
        try:
            return self._page_of.get(pg.objgen)
        except Exception:
            return None

    def _record(self, mcid: int, pg: Any, leaf: StructInfo, stm: Any) -> None:
        pi = self._page_index(pg)
        leaf.rank = self._rank_counter
        leaf.stm = _obj_key(stm) if stm is not None else None
        self._rank_counter += 1
        key = (pi, int(mcid))
        if key not in self.index:  # first stream wins on (page, mcid) collision
            self.index[key] = leaf

    def _walk(
        self,
        elem: pikepdf.Dictionary,
        anc_tags: List[str],
        anc_ids: List[int],
        anc_attrs: List[Dict[str, Any]],
        inherited_pg: Any,
        path: set,
    ) -> None:
        if not isinstance(elem, pikepdf.Dictionary):
            return
        ekey = _obj_key(elem)
        if ekey in path:
            return  # cycle guard
        path = path | {ekey}

        raw = _name(elem.get("/S"))
        ns_dict = elem.get("/NS")
        ns_uri = self._ns_uri(ns_dict)
        tag = self._resolve_role(raw, ns_dict)

        elem_id = self._elem_counter
        self._elem_counter += 1

        pg = elem.get("/Pg") or inherited_pg
        attrs = self._collect_attrs(elem)

        child_tags = anc_tags + [tag] if tag else anc_tags
        child_ids = anc_ids + [elem_id] if tag else anc_ids
        child_attrs = anc_attrs + [attrs] if tag else anc_attrs

        leaf_template = dict(
            tag=tag, raw_tag=raw, ns=ns_uri, elem_id=elem_id,
            ancestors=child_tags, ancestor_ids=child_ids, attrs=attrs,
            chain_attrs=child_attrs,
            actual_text=_text(elem.get("/ActualText")),
            alt=_text(elem.get("/Alt")),
            lang=_text(elem.get("/Lang")),
        )

        self._process_k(elem.get("/K"), leaf_template, child_tags, child_ids,
                        child_attrs, pg, path)

    def _process_k(
        self,
        k: Any,
        leaf_template: dict,
        child_tags: List[str],
        child_ids: List[int],
        child_attrs: List[Dict[str, Any]],
        pg: Any,
        path: set,
    ) -> None:
        if k is None:
            return
        if isinstance(k, pikepdf.Array):
            for item in k:
                self._process_k(item, leaf_template, child_tags, child_ids,
                                child_attrs, pg, path)
            return

        # bare MCID integer -> leaf content of the current element
        if not isinstance(k, (pikepdf.Dictionary, pikepdf.Array, pikepdf.Name,
                              pikepdf.String)):
            mcid = _as_int(k)
            if mcid is not None:
                self._record(mcid, pg, StructInfo(rank=0, **leaf_template), None)
            return

        if isinstance(k, pikepdf.Dictionary):
            t = _name(k.get("/Type"))
            if t == "MCR":  # marked-content reference
                mcid = _as_int(k.get("/MCID"))
                if mcid is not None:
                    self._record(mcid, k.get("/Pg") or pg,
                                 StructInfo(rank=0, **leaf_template), k.get("/Stm"))
            elif t == "OBJR":
                # Object reference to an annotation (form widget / link). It has
                # no text content of its own, but the element holding it is the
                # form field's owning struct element — record the edge so form
                # values can be joined to their label words structurally rather
                # than by spatial proximity.
                obj = k.get("/Obj")
                try:
                    og = obj.objgen if obj is not None else None
                except Exception:
                    og = None
                if og:
                    self.widget_links[og] = WidgetLink(
                        elem_id=leaf_template["elem_id"],
                        chain_ids=list(child_ids),
                        page_index=self._page_index(k.get("/Pg") or pg),
                        rank=self._rank_counter,
                    )
                    # Consume a rank slot so the widget occupies its true
                    # reading-order position between the surrounding MCID leaves.
                    self._rank_counter += 1
                return
            else:
                # Nested structure element. child_tags already includes the
                # current element, which is exactly the ancestor chain the child
                # inherits; _walk appends the child's own tag on top.
                self._walk(k, child_tags, child_ids, child_attrs, pg, path)

    def parse(self) -> Dict[Tuple[Optional[int], int], StructInfo]:
        if self.root is None:
            return {}
        top = self.root.get("/K")
        if top is None:
            return {}
        items = top if isinstance(top, pikepdf.Array) else [top]
        for item in items:
            if isinstance(item, pikepdf.Dictionary):
                self._walk(item, [], [], [], None, set())
        return self.index


# ── Public API ────────────────────────────────────────────────────────────────
def build_struct_index(
    source: Union[str, Path, pikepdf.Pdf],
) -> Dict[Tuple[Optional[int], int], StructInfo]:
    """Parse the structure tree of *source* (path or open ``pikepdf.Pdf``).

    Returns ``{(page_index, mcid): StructInfo}``. ``page_index`` is 0-based, or
    ``None`` when the owning element declares no resolvable ``/Pg``. Returns an
    empty dict for untagged PDFs (no ``/StructTreeRoot``)."""
    if isinstance(source, pikepdf.Pdf):
        return _StructTreeParser(source).parse()
    with pikepdf.open(str(Path(source).expanduser())) as pdf:
        return _StructTreeParser(pdf).parse()


def build_struct_index_with_links(
    source: Union[str, Path, pikepdf.Pdf],
) -> Tuple[
    Dict[Tuple[Optional[int], int], StructInfo],
    Dict[Tuple[int, int], WidgetLink],
]:
    """Like :func:`build_struct_index`, but also returns the widget-annotation
    links discovered during the same walk.

    Returns ``(struct_index, widget_links)`` where ``widget_links`` maps each
    form widget's ``objgen`` to the :class:`WidgetLink` describing its owning
    structure element. Both are empty for untagged PDFs."""
    def _run(pdf: pikepdf.Pdf):
        parser = _StructTreeParser(pdf)
        index = parser.parse()
        return index, parser.widget_links

    if isinstance(source, pikepdf.Pdf):
        return _run(source)
    with pikepdf.open(str(Path(source).expanduser())) as pdf:
        return _run(pdf)


def has_struct_tree(source: Union[str, Path, pikepdf.Pdf]) -> bool:
    """True if the PDF carries a ``/StructTreeRoot`` (tagged)."""
    if isinstance(source, pikepdf.Pdf):
        return source.Root.get("/StructTreeRoot") is not None
    with pikepdf.open(str(Path(source).expanduser())) as pdf:
        return pdf.Root.get("/StructTreeRoot") is not None


# ── CLI probe ─────────────────────────────────────────────────────────────────
def _main(argv: List[str]) -> int:
    import argparse
    from collections import Counter

    ap = argparse.ArgumentParser(description="Probe a PDF's structure tree (pikepdf).")
    ap.add_argument("pdf")
    ap.add_argument("--page", type=int, default=None, help="1-based page filter")
    args = ap.parse_args(argv)

    index = build_struct_index(args.pdf)
    if not index:
        print(f"{args.pdf}: no /StructTreeRoot recovered — likely untagged.")
        return 0

    rows = sorted(index.items(), key=lambda kv: (kv[1].rank))
    tags = Counter()
    spans = 0
    print(f"{args.pdf}: {len(index)} (page, mcid) leaves\n")
    for (pi, mcid), info in rows:
        if args.page is not None and pi != args.page - 1:
            continue
        tags[info.tag or "?"] += 1
        span = ""
        if info.col_span != 1 or info.row_span != 1:
            span = f"  span={info.col_span}x{info.row_span}"
            spans += 1
        anc = " > ".join(info.ancestors)
        raw = f" (raw={info.raw_tag})" if info.raw_tag != info.tag else ""
        print(f"  p{0 if pi is None else pi+1:<3} mcid={mcid:<4} {info.tag or '?':<10}{raw}{span}")
        print(f"        {anc}")
    print("\n  tag histogram:", dict(tags.most_common(25)))
    print(f"  cells with a non-trivial span: {spans}")
    return 0


if __name__ == "__main__":
    sys.exit(_main(sys.argv[1:]))
