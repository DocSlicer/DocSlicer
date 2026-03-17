# step_03_doc_region_assigner.py
from __future__ import annotations

from typing import Dict, Literal

import pandas as pd
import re


DocRegionType = Literal[ # FYI only
    #special
    "coverpage",
    "last_page", #-- not acccessible if doc has no page labels
    #from page_label_type
    "body",
    "annex",
    "financials",
    "schedules",
    "front_matter",
    "back_matter",
]


# ================================================================================
# STEP 1: Trivially assigned document_region based on block_role
# ================================================================================

_BLOCK_ROLE_TO_DOC_REGION = {
    "toc": "toc",
    "toc_header": "toc",
    "exhibit": "exhibits",
    "exhibit_header": "exhibits",
    "signatures": "signatures",
}


def assign_doc_region_from_block_role(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fast, vectorized assignment of doc_region from block_role.

    Rules:
      - Only assigns document_region for known block_role values.
      - Does NOT overwrite existing non-null document_region values.
      - Safe/idempotent.

    Required:
      - page_number

    Optional:
      - block_role
      - document_region
    """
    if df.empty:
        out = df.copy()
        if "document_region" not in out.columns:
            out["document_region"] = pd.Series(dtype=object)
        return out

    if "page_number" not in df.columns:
        raise KeyError(
            "assign_doc_region_trivial_from_block_role: missing required column: 'page_number'"
        )

    # If there's no block_role, nothing to do (but still ensure document_region exists)
    if "block_role" not in df.columns:
        out = df.copy()
        if "document_region" not in out.columns:
            out["document_region"] = pd.Series(index=out.index, dtype=object)
        return out

    out = df.copy()

    if "document_region" not in out.columns:
        out["document_region"] = pd.Series(index=out.index, dtype=object)

    # Normalize block_role once (vectorized)
    br = out["block_role"].astype(str).str.strip().str.lower()

    # Map roles -> region (vectorized hash lookup)
    mapped = br.map(_BLOCK_ROLE_TO_DOC_REGION)

    # Fill only where document_region is null (fast path)
    # (If you also want to treat "" as empty, see note below.)
    out["document_region"] = out["document_region"].where(out["document_region"].notna(), mapped)

    return out


# ================================================================================
# STEP 2A: Assign Special Case: coverpage
# ================================================================================

_MAX_PAGES_COVERPAGE = 5
_MAX_TOC_DISTANCE = 5

# Heuristic multipliers for ">>" comparisons (tweak as needed)
_COVER_CENTER_RATIO_MULT = 2.0      # center_ratio > median * mult
_COVER_DENSITY_RATIO_MAX = 0.5     # page_density < median * ratio
_COVER_BG_RATIO_MULT = 2.0          # bg_ratio > median * mult


def _norm_str_series(s: pd.Series) -> pd.Series:
    # preserves NaN; normalized for comparisons
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
    # do NOT overwrite existing regions (e.g., toc/exhibits/signatures)
    out.loc[m & out["document_region"].isna(), "document_region"] = "coverpage"


def _detect_coverpage_scenario_1(out: pd.DataFrame) -> set[int]:
    """
    Scenario 1:
      - page_label exists (not fully blank in doc)
      - first X pages (up to _MAX_PAGES_COVERPAGE) have blank page_label
      -> assign those pages as coverpage (must start at page 1, contiguous)
    """
    if "page_label" not in out.columns:
        return set()

    # doc-level: do we have any non-blank labels anywhere?
    lbl_blank = _is_blank_str_series(out["page_label"])
    has_any_nonblank = bool((~lbl_blank).any())
    if not has_any_nonblank:
        return set()

    # page-level: blank if ALL lines on that page have blank label
    per_page_blank_all = lbl_blank.groupby(out["page_number"].astype(int)).all()

    cover_pages: set[int] = set()
    for p in range(1, _MAX_PAGES_COVERPAGE + 1):
        if bool(per_page_blank_all.get(p, False)):
            cover_pages.add(p)
        else:
            break  # contiguous prefix only
    return cover_pages


def _detect_coverpage_scenario_2(out: pd.DataFrame) -> set[int]:
    """
    Scenario 2:
      - either:
        A) there are page_labels and NONE of the pages have a blank page_label
        OR
        B) there are no page_labels at all
      - AND toc is detected within first _MAX_TOC_DISTANCE pages
      -> all pages before first toc page become coverpage
    """
    # detect toc within first _MAX_TOC_DISTANCE pages
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

    # label condition A/B
    if "page_label" in out.columns:
        lbl_blank = _is_blank_str_series(out["page_label"])
        has_any_nonblank = bool((~lbl_blank).any())
        if has_any_nonblank:
            # if labels exist, require: no page has a blank label (blank-all by page)
            per_page_blank_all = lbl_blank.groupby(pn).all()
            any_blank_page = bool(per_page_blank_all.any())
            if any_blank_page:
                return set()
            labels_condition_ok = True
        else:
            # no page labels (fully blank)
            labels_condition_ok = True
    else:
        # no page_label column -> treat as no page labels
        labels_condition_ok = True

    if not labels_condition_ok:
        return set()

    # pages before toc
    if first_toc_page <= 1:
        return set()

    # toc distance guarantees small; still cap by _MAX_PAGES_COVERPAGE for safety
    last_cover = min(first_toc_page - 1, _MAX_PAGES_COVERPAGE)
    return set(range(1, last_cover + 1))


def _detect_coverpage_scenario_3(out: pd.DataFrame) -> set[int]:
    """
    Scenario 3 (last resort):
      - no blank page_labels, no page_labels at all, and no toc within first 5 pages
      - score pages 1.._MAX_PAGES_COVERPAGE vs global medians across pages:
        +1 if center_ratio >> median
        +1 if page_density << median
        +1 if bg_ratio >> median
      - accept contiguous prefix starting at page 1 where score >= 2
    """
    # prerequisites:
    # - ensure no scenario-1 blank labels and no scenario-2 toc near front.
    pn = out["page_number"].astype(int)

    # toc near front must be absent
    if "document_region" in out.columns:
        is_toc_line = _norm_str_series(out["document_region"]).eq("toc")
        if bool(((pn <= _MAX_TOC_DISTANCE) & is_toc_line).any()):
            return set()

    # page_label conditions: "no blank page_labels" AND "no page_labels at all"
    # (i.e., page_label column missing or fully blank everywhere)
    if "page_label" in out.columns:
        lbl_blank = _is_blank_str_series(out["page_label"])
        if bool((~lbl_blank).any()):
            # there ARE page labels -> scenario 3 explicitly excludes this
            return set()
        # fully blank everywhere: ok
    # missing page_label: ok

    # scoring requires these columns; if missing, you simply won't score enough
    have_align = "text_align" in out.columns
    have_chars = "char_count" in out.columns
    have_bg = "background_non_stroking_color" in out.columns

    # Build per-page metrics over ALL pages for medians
    # total_lines per page
    per_page_total = pn.value_counts(sort=False)

    # center_ratio
    if have_align:
        is_center = _norm_str_series(out["text_align"]).eq("center")
        per_page_center = is_center.groupby(pn).sum()
        per_page_center_ratio = (per_page_center / per_page_total).fillna(0.0)
    else:
        per_page_center_ratio = pd.Series(0.0, index=per_page_total.index)

    # density (sum of char_count)
    if have_chars:
        chars = pd.to_numeric(out["char_count"], errors="coerce").fillna(0)
        per_page_density = chars.groupby(pn).sum()
    else:
        per_page_density = pd.Series(0.0, index=per_page_total.index)

    # bg_ratio
    if have_bg:
        bg_blank = _is_blank_str_series(out["background_non_stroking_color"])
        has_bg = (~bg_blank)
        per_page_bg = has_bg.groupby(pn).sum()
        per_page_bg_ratio = (per_page_bg / per_page_total).fillna(0.0)
    else:
        per_page_bg_ratio = pd.Series(0.0, index=per_page_total.index)

    # global medians across pages (page-level)
    med_center = float(per_page_center_ratio.median()) if not per_page_center_ratio.empty else 0.0
    med_density = float(per_page_density.median()) if not per_page_density.empty else 0.0
    med_bg = float(per_page_bg_ratio.median()) if not per_page_bg_ratio.empty else 0.0

    # Score pages 1.._MAX_PAGES_COVERPAGE
    cover_pages: set[int] = set()
    for p in range(1, _MAX_PAGES_COVERPAGE + 1):
        if p not in per_page_total.index:
            break

        score = 0

        # centered
        r_center = float(per_page_center_ratio.get(p, 0.0))
        if med_center <= 0:
            # if doc median is ~0, require a small absolute floor to avoid overfitting
            if r_center >= 0.30:
                score += 1
        else:
            if r_center >= med_center * _COVER_CENTER_RATIO_MULT:
                score += 1

        # low density
        d = float(per_page_density.get(p, 0.0))
        if med_density > 0 and d <= med_density * _COVER_DENSITY_RATIO_MAX:
            score += 1

        # background text ratio
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
            break  # contiguous prefix only

    return cover_pages


def assign_coverpage(df: pd.DataFrame) -> pd.DataFrame:
    """
    Assign document_region = 'coverpage' using the 3-scenario logic.
    Never overwrites existing non-null document_region.
    """
    if df.empty:
        out = df.copy()
        _ensure_document_region(out)
        return out

    if "page_number" not in df.columns:
        raise KeyError("assign_coverpage: missing required column: 'page_number'")

    out = df.copy()
    _ensure_document_region(out)

    # Scenario 1
    pages = _detect_coverpage_scenario_1(out)
    if pages:
        _assign_coverpage_on_pages(out, pages)
        return out

    # Scenario 2
    pages = _detect_coverpage_scenario_2(out)
    if pages:
        _assign_coverpage_on_pages(out, pages)
        return out

    # Scenario 3
    pages = _detect_coverpage_scenario_3(out)
    if pages:
        _assign_coverpage_on_pages(out, pages)
        return out

    return out


# ================================================================================
# STEP 2B: Assign Special Case: last_page
# ================================================================================

_LAST_DENSITY_RATIO_MAX = 0.75   # last_page_density < median * ratio
_LAST_BG_RATIO_MULT = 1.5        # last_bg_ratio > median * mult

# Email + phone detection (fast-ish, line-level)
_EMAIL_RE = re.compile(r"\b[a-z0-9._%+\-]+@[a-z0-9.\-]+\.[a-z]{2,}\b", re.IGNORECASE)

# A permissive international-ish phone regex:
# - optional +country
# - allows separators (space, dash, dot, parentheses)
# - requires a minimum digit count to reduce false positives
_PHONE_RE = re.compile(r"(?<!\w)(?:\+?\d[\d().\- ]{6,}\d)(?!\w)")


def _last_page_has_email_or_phone(out: pd.DataFrame, last_page: int) -> bool:
    if "text" not in out.columns:
        return False

    pn = out["page_number"].astype(int)
    text = out.loc[pn.eq(last_page), "text"].astype("string")

    # quick reject
    if text.empty:
        return False

    # Vectorized-ish check: join once, single regex pass each
    blob = " ".join(text.dropna().astype(str).tolist())
    if not blob:
        return False

    if _EMAIL_RE.search(blob):
        return True

    m = _PHONE_RE.search(blob)
    if not m:
        return False

    # Digit-count gate to reduce false positives (e.g., years, short ids)
    digits = sum(ch.isdigit() for ch in m.group(0))
    return digits >= 9


def _detect_last_page_scenario_1(out: pd.DataFrame) -> int | None:
    """
    Scenario 1 (revised):
      - doc has page_labels (not fully blank)
      - compute trailing run of blank-labeled pages at the end:
          * if exactly 1 trailing blank page -> last_page
          * if 2+ trailing blank pages -> indicates "no last_page"
    Returns page_number or None.
    """
    if "page_label" not in out.columns:
        return None

    pn = out["page_number"].astype(int)

    lbl_blank = _is_blank_str_series(out["page_label"])
    has_any_nonblank = bool((~lbl_blank).any())
    if not has_any_nonblank:
        return None  # no labels in doc

    # page-level blank: ALL lines on that page have blank page_label
    per_page_blank_all = lbl_blank.groupby(pn).all()

    last_page = int(pn.max())

    # trailing blank run length
    trailing_blank = 0
    p = last_page
    while p >= 1 and bool(per_page_blank_all.get(p, False)):
        trailing_blank += 1
        p -= 1

    if trailing_blank == 1:
        return last_page

    # if 0: last page has label -> scenario 1 doesn't apply
    # if >=2: multiple unlabeled pages at end => "no last_page"
    return None


def _detect_last_page_scenario_2(out: pd.DataFrame) -> int | None:
    pn = out["page_number"].astype(int)
    last_page = int(pn.max())

    # label condition: (no labels) OR (all labels filled EXCEPT coverpage pages)
    if "page_label" in out.columns:
        lbl_blank = _is_blank_str_series(out["page_label"])
        has_any_nonblank = bool((~lbl_blank).any())

        if has_any_nonblank:
            per_page_blank_all = lbl_blank.groupby(pn).all()

            # If there are blank-labeled pages, ONLY allow them if they are coverpage pages
            if bool(per_page_blank_all.any()):
                if "document_region" in out.columns:
                    is_cover = _norm_str_series(out["document_region"]).eq("coverpage")
                    cover_pages = set(pn[is_cover].unique().tolist())
                else:
                    cover_pages = set()

                blank_pages = set(per_page_blank_all[per_page_blank_all].index.astype(int).tolist())
                non_cover_blank_pages = blank_pages - cover_pages

                # If blanks exist outside coverpage -> scenario 2 should NOT run
                if non_cover_blank_pages:
                    return None
        # else: fully blank everywhere -> ok (no labels at all)
    # else: no page_label column -> ok

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
        has_bg = (~bg_blank)
        per_page_bg = has_bg.groupby(pn).sum()
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

    # +1 bonus if email OR phone is present on last page
    if _last_page_has_email_or_phone(out, last_page):
        score += 1

    if score >= 2:
        return last_page

    return None



def assign_last_page(df: pd.DataFrame) -> pd.DataFrame:
    """
    Assign document_region = 'last_page' using the 2-scenario logic.
    Constraints:
      - only 1 page can be last_page (the last page_number)
      - never overwrites existing non-null document_region
    """
    if df.empty:
        out = df.copy()
        _ensure_document_region(out)
        return out

    if "page_number" not in df.columns:
        raise KeyError("assign_last_page: missing required column: 'page_number'")

    out = df.copy()
    _ensure_document_region(out)

    pn = out["page_number"].astype(int)
    last_page = int(pn.max())

    lp = _detect_last_page_scenario_1(out)
    if lp is None:
        lp = _detect_last_page_scenario_2(out)

    if lp is None:
        return out

    m = pn.eq(lp)
    out.loc[m & out["document_region"].isna(), "document_region"] = "last_page"
    return out



# ================================================================================
# STEP 3: Assign Standard Regions based on page_label_type
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
    Returns one row per page with:
      - page_number
      - label_blank_all
      - page_label_type
      - page_label_value
      - has_coverpage (based on existing document_region)
      - has_toc (based on existing document_region)
    """
    pn = out["page_number"].astype(int)

    page = pd.DataFrame(index=pd.Index(sorted(pn.unique()), name="page_number"))
    page["page_number"] = page.index.astype(int)

    # label_blank_all
    if "page_label" in out.columns:
        lbl_blank = _is_blank_str_series(out["page_label"])
        page["label_blank_all"] = lbl_blank.groupby(pn).all().reindex(page.index, fill_value=True)
    else:
        page["label_blank_all"] = True

    # page_label_type/value
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

    # existing regions
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
    """
    Given sorted page numbers, return runs as (start, end, length) where pages are consecutive.
    """
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


# Fix: page.index is an Index, not a Series -> no .eq().
# Use direct scalar access via .at[p, col] (fast) since page is indexed by page_number.

def _assign_per_page_regions(page: pd.DataFrame) -> dict[int, str]:
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
    
    # If there are NO page labels at all (all pages have blank labels), default everything to "body"
    # This prevents the "trailing blank" logic from assigning everything to "back_matter"
    has_any_labels = bool((~label_blank).any())
    if not has_any_labels:
        # No page labels detected - assign all non-special pages to body
        for p in pages:
            # Don't override coverpage, toc, or other special regions
            if not bool(has_cover.at[p]) and not bool(has_toc.at[p]):
                proposed[p] = "body"
        return proposed

    # ---- cover_end / first_toc (page-level) ----
    cover_pages = page.loc[has_cover, "page_number"].astype(int).tolist()
    cover_end = max(cover_pages) if cover_pages else None

    toc_pages = page.loc[has_toc, "page_number"].astype(int).tolist()
    first_toc = min(toc_pages) if toc_pages else None

    # ------------------------------------------------------------
    # NEW: only force TOC page to front_matter when it's directly under coverpage
    # ------------------------------------------------------------
    toc_under_cover: int | None = None
    if cover_end is not None and toc_pages:
        toc_after_cover = [p for p in toc_pages if p > cover_end]
        if toc_after_cover:
            cand = min(toc_after_cover)
            if cand == cover_end + 1:
                toc_under_cover = cand
                # remaining null lines on that TOC page become front_matter
                proposed[toc_under_cover] = "front_matter"

    # ------------------------------------------------------------
    # (1) alpha_numeric / alpha_roman => annex/financials/schedules
    # (unchanged)
    # ------------------------------------------------------------
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

    # ------------------------------------------------------------
    # (2) body = longest contiguous run of filled arabic
    # TOC pages are allowed EXCEPT toc_under_cover
    # ------------------------------------------------------------
    is_arabic_filled = ptype.eq("arabic") & (~label_blank)

    eligible_for_body = is_arabic_filled & (~has_cover)
    if toc_under_cover is not None:
        # exclude ONLY that specific toc page
        eligible_for_body = eligible_for_body & (page["page_number"].astype(int) != toc_under_cover)

    arabic_pages = page.loc[eligible_for_body, "page_number"].astype(int).sort_values().tolist()
    arabic_runs = _contiguous_runs(arabic_pages)

    body_run: tuple[int, int, int] | None = None
    if arabic_runs:
        arabic_runs_sorted = sorted(arabic_runs, key=lambda t: (-t[2], t[0]))
        body_run = arabic_runs_sorted[0]
        bs, be, _ = body_run
        for p in range(bs, be + 1):
            proposed.setdefault(p, "body")

    # ------------------------------------------------------------
    # (3) roman logic (unchanged)
    # ------------------------------------------------------------
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

    # ------------------------------------------------------------
    # (4) unfilled arabic pages between coverpage and first toc => front_matter
    # (unchanged)
    # ------------------------------------------------------------
    if cover_end is not None and first_toc is not None and first_toc > cover_end + 1:
        for p in range(cover_end + 1, first_toc):
            if p not in doc_pages_set:
                continue
            if (ptype.at[p] == "arabic") and bool(label_blank.at[p]):
                proposed.setdefault(p, "front_matter")

    # trailing blank-labeled pages => back_matter
    trailing_blank = 0
    p = max_page
    while p >= min_page and p in doc_pages_set and bool(label_blank.at[p]):
        trailing_blank += 1
        p -= 1

    if trailing_blank > 0:
        for pp in range(max_page - trailing_blank + 1, max_page + 1):
            if pp in doc_pages_set:
                proposed.setdefault(pp, "back_matter")

    # ------------------------------------------------------------
    # (5) arabic repeats + attach blank pages to NEXT region
    # ------------------------------------------------------------
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

            # attach immediate preceding blank-labeled pages to this run's region
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
    Assign standard regions using rules (1)-(5), based on page_label, page_label_type, page_label_value.
    Does NOT overwrite existing document_region values.
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

    # Apply without overwriting
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
    Main entry point.
    """
    df = assign_doc_region_from_block_role(df)
    df = assign_coverpage(df)
    df = assign_last_page(df)

    # TODO: Now back matter can span the latest arabic sequence + pages with blank labels at the end. See if thats desired.
    df = assign_doc_region_from_page_labels(df)
    return df