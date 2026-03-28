"""
Document region assignment for parsed document DataFrames.

Each row in the input DataFrame represents one text line from a document.
This module assigns a ``document_region`` value to every row, classifying
it as one of: ``toc``, ``exhibits``, ``coverpage``, ``last_page``,
``body``, ``front_matter``, ``back_matter``, ``annex``, ``financials``,
or ``schedules``.

Assignment runs in four passes, each of which only fills rows whose
``document_region`` is still null:

1. **Block-role pass** — rows with a known ``block_role`` (``toc``,
   ``toc_header``, ``exhibit``, ``exhibit_header``) are assigned directly.

2. **Coverpage pass** — the leading pages of the document are examined
   using four detection strategies tried in order (first match wins):

   * Scenario 1 – pages with a blank ``page_label`` in a doc that
     otherwise has labels.
   * Scenario 2 – pages before an early TOC when labels are fully
     populated or absent.
   * Scenario 3 – visual/layout heuristics (center-aligned text, low
     density, coloured background) when no labels are present.
   * Scenario 4 – SEC filing text-based detection: at least
     ``SEC_MIN_HITS`` full-line matches against ``SEC_COVERPAGE_HEURISTICS``
     plus at least one checkbox character (☐ / ☒).

3. **Last-page pass** — the final page is tagged when it matches a
   blank-label or low-density/contact-info pattern.

4. **Page-label pass** — remaining pages are classified by their
   ``page_label_type`` (arabic → body, roman → front/back matter,
   alpha-prefixed → annex/financials/schedules, etc.).

Required column: ``page_number`` (int-coercible).
Optional columns improve accuracy: ``block_role``, ``page_label``,
``page_label_type``, ``page_label_value``, ``text``, ``text_align``,
``char_count``, ``background_non_stroking_color``.
"""
from __future__ import annotations

from typing import Literal

import pandas as pd
import re


DocRegionType = Literal[
    "toc",
    "exhibits",
    "coverpage",
    "last_page",
    "body",
    "annex",
    "financials",
    "schedules",
    "front_matter",
    "back_matter",
]


# ================================================================================
# STEP 1: Assign document_region from block_role
# ================================================================================

_BLOCK_ROLE_TO_DOC_REGION = {
    "toc": "toc",
    "toc_header": "toc",
    "exhibit": "exhibits",
    "exhibit_header": "exhibits",
}


def assign_doc_region_from_block_role(df: pd.DataFrame) -> pd.DataFrame:
    """
    Assign ``document_region`` for rows whose ``block_role`` maps to a known
    region.  Rows with an already-set ``document_region`` are left untouched.

    Optional columns: ``block_role``, ``document_region``.
    """
    if df.empty:
        out = df.copy()
        if "document_region" not in out.columns:
            out["document_region"] = pd.Series(dtype=object)
        return out

    if "block_role" not in df.columns:
        out = df.copy()
        if "document_region" not in out.columns:
            out["document_region"] = pd.Series(index=out.index, dtype=object)
        return out

    out = df.copy()

    if "document_region" not in out.columns:
        out["document_region"] = pd.Series(index=out.index, dtype=object)

    br = out["block_role"].astype(str).str.strip().str.lower()
    mapped = br.map(_BLOCK_ROLE_TO_DOC_REGION)
    out["document_region"] = out["document_region"].where(out["document_region"].notna(), mapped)

    return out


# ================================================================================
# STEP 2A: Assign coverpage
# ================================================================================

_MAX_PAGES_COVERPAGE = 5
_MAX_TOC_DISTANCE = 5

# Thresholds for the layout-scoring coverpage detector (scenario 3).
_COVER_CENTER_RATIO_MULT = 2.0   # page center_ratio must exceed  median * this
_COVER_DENSITY_RATIO_MAX = 0.5   # page density must be below     median * this
_COVER_BG_RATIO_MULT = 2.0       # page bg_ratio must exceed      median * this

# ---------------------------------------------------------------------------
# SEC filing coverpage heuristics (scenario 4)
# ---------------------------------------------------------------------------
# Each entry is one candidate "hit".  A full-line match (stripped,
# case-insensitive) against any entry counts as one unique hit.
# Extend or trim this list to adjust which filings are recognised.
SEC_COVERPAGE_HEURISTICS: list[str] = [
    "SECURITIES AND EXCHANGE COMMISSION",
    "Washington, D.C. 20549",
    "(I.R.S. Employer Identification No.)",
    "(Exact name of Registrant as specified in its charter)",
    "(Name of Registrant as Specified In Its Charter)",
    "(Registrant's telephone number, including area code)",
    "(CUSIP Number of Class of Securities)",
]

# Number of distinct heuristic lines that must match for scenario 4 to fire.
SEC_MIN_HITS: int = 3

# Checkbox characters that must appear at least once (scenario 4).
SEC_CHECKBOX_CHARS: frozenset[str] = frozenset({"☐", "☒"})


def _norm_str_series(s: pd.Series) -> pd.Series:
    return s.astype("string").str.strip().str.lower()


def _is_blank_str_series(s: pd.Series) -> pd.Series:
    ss = s.astype("string")
    return ss.isna() | (ss.str.strip() == "")


def _ensure_document_region(out: pd.DataFrame) -> None:
    if "document_region" not in out.columns:
        out["document_region"] = pd.Series(index=out.index, dtype=object)


def _pages_mask(out: pd.DataFrame, pages: set[int]) -> pd.Series:
    return out["page_number"].astype(int).isin(pages)


def _assign_coverpage_on_pages(out: pd.DataFrame, pages: set[int]) -> None:
    if not pages:
        return
    m = _pages_mask(out, pages)
    out.loc[m & out["document_region"].isna(), "document_region"] = "coverpage"


def _detect_coverpage_scenario_1(out: pd.DataFrame) -> set[int]:
    """
    Applies when the document has page labels but the leading pages lack one.

    Returns the contiguous run of blank-labeled pages starting at page 1,
    capped at ``_MAX_PAGES_COVERPAGE``.
    """
    if "page_label" not in out.columns:
        return set()

    lbl_blank = _is_blank_str_series(out["page_label"])
    if not bool((~lbl_blank).any()):
        return set()  # no labels anywhere in the document

    per_page_blank_all = lbl_blank.groupby(out["page_number"].astype(int)).all()

    cover_pages: set[int] = set()
    for p in range(1, _MAX_PAGES_COVERPAGE + 1):
        if bool(per_page_blank_all.get(p, False)):
            cover_pages.add(p)
        else:
            break
    return cover_pages


def _detect_coverpage_scenario_2(out: pd.DataFrame) -> set[int]:
    """
    Applies when a TOC is detected within the first ``_MAX_TOC_DISTANCE``
    pages and all page labels are filled (or absent).

    Returns all pages before the first TOC page, capped at
    ``_MAX_PAGES_COVERPAGE``.
    """
    if "document_region" not in out.columns:
        return set()

    pn = out["page_number"].astype(int)
    is_toc_line = _norm_str_series(out["document_region"]).eq("toc")
    toc_pages = pn[is_toc_line].unique()
    if toc_pages.size == 0:
        return set()

    first_toc_page = int(pd.Series(toc_pages).min())
    if first_toc_page < 1 or first_toc_page > _MAX_TOC_DISTANCE:
        return set()

    if "page_label" in out.columns:
        lbl_blank = _is_blank_str_series(out["page_label"])
        if bool((~lbl_blank).any()):
            # Labels exist; reject if any page has a blank label.
            per_page_blank_all = lbl_blank.groupby(pn).all()
            if bool(per_page_blank_all.any()):
                return set()

    if first_toc_page <= 1:
        return set()

    last_cover = min(first_toc_page - 1, _MAX_PAGES_COVERPAGE)
    return set(range(1, last_cover + 1))


def _detect_coverpage_scenario_3(out: pd.DataFrame) -> set[int]:
    """
    Layout-based fallback for documents with no page labels and no early TOC.

    Scores each of the first ``_MAX_PAGES_COVERPAGE`` pages against global
    page-level medians:

    * +1 if ``center_ratio`` is significantly above the median.
    * +1 if character density is significantly below the median.
    * +1 if background-color ratio is significantly above the median.

    Returns the contiguous prefix of pages starting at page 1 where the
    score is ≥ 2.
    """
    pn = out["page_number"].astype(int)

    if "document_region" in out.columns:
        is_toc_line = _norm_str_series(out["document_region"]).eq("toc")
        if bool(((pn <= _MAX_TOC_DISTANCE) & is_toc_line).any()):
            return set()

    if "page_label" in out.columns:
        lbl_blank = _is_blank_str_series(out["page_label"])
        if bool((~lbl_blank).any()):
            return set()

    have_align = "text_align" in out.columns
    have_chars = "char_count" in out.columns
    have_bg = "background_non_stroking_color" in out.columns

    per_page_total = pn.value_counts(sort=False)

    if have_align:
        is_center = _norm_str_series(out["text_align"]).eq("center")
        per_page_center = is_center.groupby(pn).sum()
        per_page_center_ratio = (per_page_center / per_page_total).fillna(0.0)
    else:
        per_page_center_ratio = pd.Series(0.0, index=per_page_total.index)

    if have_chars:
        chars = pd.to_numeric(out["char_count"], errors="coerce").fillna(0)
        per_page_density = chars.groupby(pn).sum()
    else:
        per_page_density = pd.Series(0.0, index=per_page_total.index)

    if have_bg:
        bg_blank = _is_blank_str_series(out["background_non_stroking_color"])
        per_page_bg = (~bg_blank).groupby(pn).sum()
        per_page_bg_ratio = (per_page_bg / per_page_total).fillna(0.0)
    else:
        per_page_bg_ratio = pd.Series(0.0, index=per_page_total.index)

    med_center = float(per_page_center_ratio.median()) if not per_page_center_ratio.empty else 0.0
    med_density = float(per_page_density.median()) if not per_page_density.empty else 0.0
    med_bg = float(per_page_bg_ratio.median()) if not per_page_bg_ratio.empty else 0.0

    cover_pages: set[int] = set()
    for p in range(1, _MAX_PAGES_COVERPAGE + 1):
        if p not in per_page_total.index:
            break

        score = 0

        r_center = float(per_page_center_ratio.get(p, 0.0))
        if med_center <= 0:
            if r_center >= 0.30:
                score += 1
        else:
            if r_center >= med_center * _COVER_CENTER_RATIO_MULT:
                score += 1

        d = float(per_page_density.get(p, 0.0))
        if med_density > 0 and d <= med_density * _COVER_DENSITY_RATIO_MAX:
            score += 1

        r_bg = float(per_page_bg_ratio.get(p, 0.0))
        if med_bg <= 0:
            if r_bg >= 0.10:
                score += 1
        else:
            if r_bg >= med_bg * _COVER_BG_RATIO_MULT:
                score += 1

        if score >= 2:
            cover_pages.add(p)
        else:
            break

    return cover_pages


def _detect_coverpage_scenario_4(
    out: pd.DataFrame,
    heuristics: list[str] | None = None,
    min_hits: int = SEC_MIN_HITS,
) -> set[int]:
    """
    Text-based fallback for SEC filings that lack page labels and have no
    early TOC (and therefore don't trigger scenarios 1–3).

    Fires when, across pages 1–``_MAX_PAGES_COVERPAGE``:

    1. At least ``min_hits`` *distinct* entries from ``heuristics`` appear
       as exact full-line matches (stripped, case-insensitive).
    2. At least one checkbox character (☐ / ☒) is present.

    Returns ``range(1, last_matched_page + 1)`` as the coverpage set.

    Parameters
    ----------
    heuristics:
        Override the module-level ``SEC_COVERPAGE_HEURISTICS`` list.
    min_hits:
        Override ``SEC_MIN_HITS``.
    """
    if "text" not in out.columns:
        return set()

    _heuristics = heuristics if heuristics is not None else SEC_COVERPAGE_HEURISTICS
    norm_heuristics = {h.strip().lower() for h in _heuristics}
    if not norm_heuristics:
        return set()

    pn = out["page_number"].astype(int)
    cand = out[pn <= _MAX_PAGES_COVERPAGE]
    if cand.empty:
        return set()

    text_stripped = cand["text"].astype("string").str.strip()
    text_lower = text_stripped.str.lower()

    hits_mask = text_lower.isin(norm_heuristics)
    if int(text_lower[hits_mask].nunique()) < min_hits:
        return set()

    checkbox_pattern = "|".join(re.escape(c) for c in SEC_CHECKBOX_CHARS)
    checkbox_mask = text_stripped.str.contains(checkbox_pattern, regex=True, na=False)
    if not bool(checkbox_mask.any()):
        return set()

    hit_pages = set(cand.loc[hits_mask, "page_number"].astype(int))
    checkbox_pages = set(cand.loc[checkbox_mask, "page_number"].astype(int))

    last_relevant = max(hit_pages | checkbox_pages)
    return set(range(1, last_relevant + 1))


def assign_coverpage(df: pd.DataFrame) -> pd.DataFrame:
    """
    Assign ``document_region = 'coverpage'`` to leading pages of the document.

    Tries four detection strategies in order; the first that identifies at
    least one page is used and the rest are skipped.  Existing non-null
    ``document_region`` values are never overwritten.

    Requires: ``page_number``.
    """
    if df.empty:
        out = df.copy()
        _ensure_document_region(out)
        return out

    if "page_number" not in df.columns:
        raise KeyError("assign_coverpage: missing required column: 'page_number'")

    out = df.copy()
    _ensure_document_region(out)

    for detect in (
        _detect_coverpage_scenario_1,
        _detect_coverpage_scenario_2,
        _detect_coverpage_scenario_3,
        _detect_coverpage_scenario_4,
    ):
        pages = detect(out)
        if pages:
            _assign_coverpage_on_pages(out, pages)
            return out

    return out


# ================================================================================
# STEP 2B: Assign last_page
# ================================================================================

_LAST_DENSITY_RATIO_MAX = 0.75  # last page density must be below  median * this
_LAST_BG_RATIO_MULT = 1.5       # last page bg_ratio must exceed    median * this

_EMAIL_RE = re.compile(r"\b[a-z0-9._%+\-]+@[a-z0-9.\-]+\.[a-z]{2,}\b", re.IGNORECASE)

# Matches international phone numbers with common separators; a digit-count
# gate (≥ 9 digits) is applied after matching to suppress false positives.
_PHONE_RE = re.compile(r"(?<!\w)(?:\+?\d[\d().\- ]{6,}\d)(?!\w)")


def _last_page_has_email_or_phone(out: pd.DataFrame, last_page: int) -> bool:
    if "text" not in out.columns:
        return False

    pn = out["page_number"].astype(int)
    text = out.loc[pn.eq(last_page), "text"].astype("string")

    if text.empty:
        return False

    blob = " ".join(text.dropna().astype(str).tolist())
    if not blob:
        return False

    if _EMAIL_RE.search(blob):
        return True

    m = _PHONE_RE.search(blob)
    if not m:
        return False

    digits = sum(ch.isdigit() for ch in m.group(0))
    return digits >= 9


def _detect_last_page_scenario_1(out: pd.DataFrame) -> int | None:
    """
    Applies when the document has page labels and exactly one trailing page
    lacks a label.  That page is returned as ``last_page``; two or more
    unlabeled trailing pages indicate a back-matter section, not a
    standalone last page, so ``None`` is returned.
    """
    if "page_label" not in out.columns:
        return None

    pn = out["page_number"].astype(int)

    lbl_blank = _is_blank_str_series(out["page_label"])
    if not bool((~lbl_blank).any()):
        return None

    per_page_blank_all = lbl_blank.groupby(pn).all()
    last_page = int(pn.max())

    trailing_blank = 0
    p = last_page
    while p >= 1 and bool(per_page_blank_all.get(p, False)):
        trailing_blank += 1
        p -= 1

    return last_page if trailing_blank == 1 else None


def _detect_last_page_scenario_2(out: pd.DataFrame) -> int | None:
    """
    Layout and contact-info based detector for documents whose labels are
    fully populated (blank pages only permitted on already-assigned coverpage
    pages) or entirely absent.

    Scores the final page:

    * +1 if character density is significantly below the document median.
    * +1 if background-color ratio is significantly above the median.
    * +1 if the page contains an email address or phone number.

    Returns the last page number when the score is ≥ 2, otherwise ``None``.
    """
    pn = out["page_number"].astype(int)
    last_page = int(pn.max())

    if "page_label" in out.columns:
        lbl_blank = _is_blank_str_series(out["page_label"])
        if bool((~lbl_blank).any()):
            per_page_blank_all = lbl_blank.groupby(pn).all()

            if bool(per_page_blank_all.any()):
                if "document_region" in out.columns:
                    is_cover = _norm_str_series(out["document_region"]).eq("coverpage")
                    cover_pages = set(pn[is_cover].unique().tolist())
                else:
                    cover_pages = set()

                blank_pages = set(per_page_blank_all[per_page_blank_all].index.astype(int).tolist())
                if blank_pages - cover_pages:
                    return None

    have_chars = "char_count" in out.columns
    have_bg = "background_non_stroking_color" in out.columns

    per_page_total = pn.value_counts(sort=False)

    if have_chars:
        chars = pd.to_numeric(out["char_count"], errors="coerce").fillna(0)
        per_page_density = chars.groupby(pn).sum()
    else:
        per_page_density = pd.Series(0.0, index=per_page_total.index)

    if have_bg:
        bg_blank = _is_blank_str_series(out["background_non_stroking_color"])
        per_page_bg = (~bg_blank).groupby(pn).sum()
        per_page_bg_ratio = (per_page_bg / per_page_total).fillna(0.0)
    else:
        per_page_bg_ratio = pd.Series(0.0, index=per_page_total.index)

    med_density = float(per_page_density.median()) if not per_page_density.empty else 0.0
    med_bg = float(per_page_bg_ratio.median()) if not per_page_bg_ratio.empty else 0.0

    score = 0

    d_last = float(per_page_density.get(last_page, 0.0))
    if med_density > 0 and d_last <= med_density * _LAST_DENSITY_RATIO_MAX:
        score += 1

    r_bg_last = float(per_page_bg_ratio.get(last_page, 0.0))
    if med_bg <= 0:
        if r_bg_last >= 0.10:
            score += 1
    else:
        if r_bg_last >= med_bg * _LAST_BG_RATIO_MULT:
            score += 1

    if _last_page_has_email_or_phone(out, last_page):
        score += 1

    return last_page if score >= 2 else None


def assign_last_page(df: pd.DataFrame) -> pd.DataFrame:
    """
    Assign ``document_region = 'last_page'`` to the final page of the
    document when it matches a known closing-page pattern.

    At most one page (the highest ``page_number``) can receive this region.
    Existing non-null ``document_region`` values are never overwritten.

    Requires: ``page_number``.
    """
    if df.empty:
        out = df.copy()
        _ensure_document_region(out)
        return out

    if "page_number" not in df.columns:
        raise KeyError("assign_last_page: missing required column: 'page_number'")

    out = df.copy()
    _ensure_document_region(out)

    lp = _detect_last_page_scenario_1(out) or _detect_last_page_scenario_2(out)

    if lp is None:
        return out

    pn = out["page_number"].astype(int)
    out.loc[pn.eq(lp) & out["document_region"].isna(), "document_region"] = "last_page"
    return out


# ================================================================================
# STEP 3: Assign standard regions from page_label_type
# ================================================================================

_STD_REGIONS = {
    "body",
    "annex",
    "financials",
    "schedules",
    "front_matter",
    "back_matter",
}

_ALPHA_TYPES = {"alpha_numeric", "alpha_roman"}
_ROMAN_TYPES = {"roman", "roman_numeric"}


def _first_nonblank_str(s: pd.Series) -> str | None:
    ss = s.astype("string")
    ss = ss[~_is_blank_str_series(ss)]
    if ss.empty:
        return None
    return str(ss.iloc[0])


def _build_page_index_df(out: pd.DataFrame) -> pd.DataFrame:
    """
    Reduce the line-level DataFrame to one row per page with:
    ``page_number``, ``label_blank_all``, ``page_label_type``,
    ``page_label_value``, ``has_coverpage``, ``has_toc``.
    """
    pn = out["page_number"].astype(int)

    page = pd.DataFrame(index=pd.Index(sorted(pn.unique()), name="page_number"))
    page["page_number"] = page.index.astype(int)

    if "page_label" in out.columns:
        lbl_blank = _is_blank_str_series(out["page_label"])
        page["label_blank_all"] = lbl_blank.groupby(pn).all().reindex(page.index, fill_value=True)
    else:
        page["label_blank_all"] = True

    if "page_label_type" in out.columns:
        page["page_label_type"] = (
            out.groupby(pn)["page_label_type"].apply(_first_nonblank_str).reindex(page.index)
        )
    else:
        page["page_label_type"] = None

    if "page_label_value" in out.columns:
        page["page_label_value"] = (
            out.groupby(pn)["page_label_value"].apply(_first_nonblank_str).reindex(page.index)
        )
    else:
        page["page_label_value"] = None

    dr = _norm_str_series(out["document_region"]) if "document_region" in out.columns else pd.Series([], dtype="string")
    page["has_coverpage"] = dr.eq("coverpage").groupby(pn).any().reindex(page.index, fill_value=False)
    page["has_toc"] = dr.eq("toc").groupby(pn).any().reindex(page.index, fill_value=False)

    return page


_ALPHA_PREFIX_RE = re.compile(r"^\s*([A-Z])\s*[-–]\s*([A-Z0-9IVXLCDM]+)\s*$", re.IGNORECASE)


def _alpha_prefix(val: str | None) -> str | None:
    if not val:
        return None
    m = _ALPHA_PREFIX_RE.match(str(val).strip())
    if not m:
        return None
    return m.group(1).upper()


def _contiguous_runs(pages: list[int]) -> list[tuple[int, int, int]]:
    """Return ``(start, end, length)`` tuples for each contiguous run in *pages*."""
    if not pages:
        return []
    runs = []
    s = e = pages[0]
    for p in pages[1:]:
        if p == e + 1:
            e = p
        else:
            runs.append((s, e, e - s + 1))
            s = e = p
    runs.append((s, e, e - s + 1))
    return runs


def _assign_per_page_regions(page: pd.DataFrame) -> dict[int, str]:
    """
    Derive a proposed ``document_region`` for each page based on label type.

    Uses ``page.at[p, col]`` for scalar lookups because ``page`` is indexed
    by ``page_number`` (an Index, not a Series).
    """
    pages = page["page_number"].astype(int).tolist()
    if not pages:
        return {}

    proposed: dict[int, str] = {}

    doc_pages_set = set(pages)
    min_page = min(pages)
    max_page = max(pages)

    label_blank = page["label_blank_all"].fillna(False).astype(bool)
    ptype = page["page_label_type"].fillna("").astype("string").str.strip().str.lower()
    has_cover = page["has_coverpage"].fillna(False).astype(bool)
    has_toc = page["has_toc"].fillna(False).astype(bool)

    # When no page labels exist at all, every non-special page is body.
    if not bool((~label_blank).any()):
        for p in pages:
            if not bool(has_cover.at[p]) and not bool(has_toc.at[p]):
                proposed[p] = "body"
        return proposed

    cover_pages = page.loc[has_cover, "page_number"].astype(int).tolist()
    cover_end = max(cover_pages) if cover_pages else None

    toc_pages = page.loc[has_toc, "page_number"].astype(int).tolist()
    first_toc = min(toc_pages) if toc_pages else None

    # A TOC page immediately following the coverpage is classified as
    # front_matter rather than body.
    toc_under_cover: int | None = None
    if cover_end is not None and toc_pages:
        toc_after_cover = [p for p in toc_pages if p > cover_end]
        if toc_after_cover:
            cand = min(toc_after_cover)
            if cand == cover_end + 1:
                toc_under_cover = cand
                proposed[toc_under_cover] = "front_matter"

    # (1) Alpha-prefixed labels → annex / financials / schedules
    is_alpha = ptype.isin(_ALPHA_TYPES) & (~label_blank)
    alpha_pages = page.loc[is_alpha, "page_number"].astype(int).tolist()
    if alpha_pages:
        prefixes = []
        for p in alpha_pages:
            v = page.at[p, "page_label_value"]
            pref = _alpha_prefix(None if pd.isna(v) else str(v))
            if pref:
                prefixes.append(pref)

        prefix_set = set(prefixes)
        alpha_region = "annex"
        if prefix_set == {"F"}:
            alpha_region = "financials"
        elif prefix_set == {"S"}:
            alpha_region = "schedules"

        for p in alpha_pages:
            proposed[p] = alpha_region

    # (2) Body = longest contiguous run of filled arabic pages (excluding
    #     coverpage and the TOC page directly under the coverpage).
    is_arabic_filled = ptype.eq("arabic") & (~label_blank)
    eligible_for_body = is_arabic_filled & (~has_cover)
    if toc_under_cover is not None:
        eligible_for_body = eligible_for_body & (page["page_number"].astype(int) != toc_under_cover)

    arabic_pages = page.loc[eligible_for_body, "page_number"].astype(int).sort_values().tolist()
    arabic_runs = _contiguous_runs(arabic_pages)

    body_run: tuple[int, int, int] | None = None
    if arabic_runs:
        body_run = sorted(arabic_runs, key=lambda t: (-t[2], t[0]))[0]
        bs, be, _ = body_run
        for p in range(bs, be + 1):
            proposed.setdefault(p, "body")

    # (3) Roman-numbered pages → front_matter (before body) or back_matter (after).
    is_roman_filled = ptype.isin(_ROMAN_TYPES) & (~label_blank)
    roman_pages = page.loc[is_roman_filled, "page_number"].astype(int).tolist()

    if body_run is not None:
        bs, be, _ = body_run
        for p in roman_pages:
            if p < bs:
                proposed.setdefault(p, "front_matter")
            elif p > be:
                proposed.setdefault(p, "back_matter")
    else:
        for p in roman_pages:
            proposed.setdefault(p, "front_matter")

    # (4) Blank-labeled arabic pages between coverpage and first TOC → front_matter.
    if cover_end is not None and first_toc is not None and first_toc > cover_end + 1:
        for p in range(cover_end + 1, first_toc):
            if p not in doc_pages_set:
                continue
            if (ptype.at[p] == "arabic") and bool(label_blank.at[p]):
                proposed.setdefault(p, "front_matter")

    # (5) Trailing blank-labeled pages → back_matter.
    trailing_blank = 0
    p = max_page
    while p >= min_page and p in doc_pages_set and bool(label_blank.at[p]):
        trailing_blank += 1
        p -= 1

    if trailing_blank > 0:
        for pp in range(max_page - trailing_blank + 1, max_page + 1):
            if pp in doc_pages_set:
                proposed.setdefault(pp, "back_matter")

    # (6) Secondary arabic runs (page-number resets) → front_matter or back_matter;
    #     immediately preceding blank pages are attached to the same region.
    if arabic_runs:
        for (rs, re_, _len) in arabic_runs:
            if body_run is not None and (rs, re_, _len) == body_run:
                continue

            if body_run is None:
                run_region = "back_matter"
            else:
                bs, be, _ = body_run
                run_region = "front_matter" if re_ < bs else "back_matter"

            for p in range(rs, re_ + 1):
                proposed.setdefault(p, run_region)

            q = rs - 1
            while q >= min_page and q in doc_pages_set:
                if not bool(label_blank.at[q]):
                    break
                if bool(has_cover.at[q]) or bool(has_toc.at[q]):
                    break
                proposed.setdefault(q, run_region)
                q -= 1

    return proposed


def assign_doc_region_from_page_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Assign standard regions (``body``, ``front_matter``, ``back_matter``,
    ``annex``, ``financials``, ``schedules``) based on ``page_label``,
    ``page_label_type``, and ``page_label_value``.

    Existing non-null ``document_region`` values are never overwritten.

    Requires: ``page_number``.
    """
    if df.empty:
        out = df.copy()
        _ensure_document_region(out)
        return out

    if "page_number" not in df.columns:
        raise KeyError("assign_doc_region_from_page_labels: missing required column: 'page_number'")

    out = df.copy()
    _ensure_document_region(out)

    page = _build_page_index_df(out)
    per_page_region = _assign_per_page_regions(page)

    if not per_page_region:
        return out

    pn = out["page_number"].astype(int)
    dr_isna = out["document_region"].isna()

    for region in _STD_REGIONS:
        pages_for_region = {p for p, r in per_page_region.items() if r == region}
        if not pages_for_region:
            continue
        m = pn.isin(pages_for_region)
        out.loc[m & dr_isna, "document_region"] = region

    return out


# ================================================================================
# Public API
# ================================================================================

def assign_doc_region(df: pd.DataFrame) -> pd.DataFrame:
    """
    Assign a ``document_region`` to every row in *df*.

    This is the main entry point.  It runs the four assignment passes in
    order and returns a copy of *df* with the ``document_region`` column
    populated.  Existing non-null values are preserved throughout.

    Passes
    ------
    1. ``assign_doc_region_from_block_role`` — maps ``block_role`` values
       ``toc``, ``toc_header``, ``exhibit``, and ``exhibit_header`` to
       ``toc`` / ``exhibits``.

    2. ``assign_coverpage`` — detects the leading cover page(s) using up to
       four strategies (blank labels → TOC anchor → layout scoring →
       SEC text heuristics).

    3. ``assign_last_page`` — detects a standalone closing page using blank
       labels or low-density / contact-info signals.

    4. ``assign_doc_region_from_page_labels`` — classifies the remaining
       pages as ``body``, ``front_matter``, ``back_matter``, ``annex``,
       ``financials``, or ``schedules`` based on page label type and value.

    Parameters
    ----------
    df:
        Line-level DataFrame.  Must contain a ``page_number`` column.
        All other columns are optional; missing columns reduce detection
        accuracy but never raise errors.

    Returns
    -------
    pd.DataFrame
        A copy of *df* with ``document_region`` filled where detectable.
        Rows that could not be classified are left as ``NaN``.
    """
    df = assign_doc_region_from_block_role(df)
    df = assign_coverpage(df)
    df = assign_last_page(df)
    df = assign_doc_region_from_page_labels(df)
    return df
