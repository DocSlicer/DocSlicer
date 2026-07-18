"""
Document section identification for parsed document DataFrames.

classify_sections() runs in multiple passes:

Pass A — build_temp_sections()
    Reduce rows to a per-page index, detect section boundaries from
    page-label format changes, value restarts, and docx section_id changes,
    then assemble a compact section index.  The temp_section_id is joined
    back to the row-level DataFrame as a new column.

Pass B — assign_coverpage_and_last_page()
    Detect and mark coverpage and last_page sections, splitting a temp
    section when only part of it qualifies.

Pass C — assign_section_labels()
    Assign human-readable section labels (financials, schedules, annex,
    body, front_matter, back_matter) to the remaining unlabeled sections.

The final 'section' column is then smeared back to rows in classify_sections().
"""
from __future__ import annotations

import re

import pandas as pd


# ================================================================================
# Constants
# ================================================================================

_KNOWN_LABEL_FORMATS: frozenset[str] = frozenset({
    "arabic", "arabic_sub", "roman", "roman_numeric",
    "alpha_numeric", "alpha_roman",
})

_ALPHA_FORMATS: frozenset[str] = frozenset({"alpha_numeric", "alpha_roman"})
_ARABIC_FORMATS: frozenset[str] = frozenset({"arabic", "arabic_sub"})
_ROMAN_FORMATS: frozenset[str] = frozenset({"roman", "roman_numeric"})

_TOC_BLOCK_TYPES: frozenset[str] = frozenset({"toc", "toc_heading"})

# Max blank pages pulled into a following section that starts at value > 1
_MAX_BLANK_PREFIX_PULL = 3

# Extracts the leading letter from alpha-prefixed labels like "A-1", "F-12", "S-3"
_ALPHA_PREFIX_RE = re.compile(r"^\s*([A-Za-z])\s*[-–.]")

# Roman numeral parser (covers values up to 3999)
_ROMAN_MAP = [
    (1000, "m"), (900, "cm"), (500, "d"), (400, "cd"),
    (100,  "c"), (90,  "xc"), (50,  "l"), (40,  "xl"),
    (10,   "x"), (9,   "ix"), (5,   "v"), (4,   "iv"), (1, "i"),
]


def _roman_to_int(s: str) -> int | None:
    """Parse a roman numeral string to an integer, or return None on failure."""
    s = s.strip().lower()
    if not s:
        return None
    result = 0
    i = 0
    for value, numeral in _ROMAN_MAP:
        while s[i: i + len(numeral)] == numeral:
            result += value
            i += len(numeral)
    return result if i == len(s) else None


def _parse_label_value(label_text: str | None, label_type: str | None) -> int | None:
    """
    Extract an integer from a raw page label string.

    Used as a fallback when the ``page_label_value`` column is absent
    (e.g. the HTML pipeline does not pre-parse this).

    Handles:
    - arabic / arabic_sub: strip to digits and parse
    - roman / roman_numeric: roman numeral conversion
    - alpha_numeric / alpha_roman: extract the trailing integer portion
    """
    if not label_text or not label_type:
        return None
    txt = label_text.strip()
    if not txt:
        return None

    ltype = label_type.lower()

    if ltype in ("arabic", "arabic_sub"):
        m = re.search(r"\d+", txt)
        return int(m.group()) if m else None

    if ltype in ("roman", "roman_numeric"):
        # roman_numeric may have a trailing number: "I-1" → use the roman part
        m = re.match(r"^([ivxlcdmIVXLCDM]+)", txt)
        if m:
            return _roman_to_int(m.group(1))
        return None

    if ltype in ("alpha_numeric", "alpha_roman"):
        # e.g. "A-1", "B-12" → trailing integer
        m = re.search(r"(\d+)\s*$", txt)
        return int(m.group(1)) if m else None

    return None


# ================================================================================
# Pass A — step 1: reduce line-level df to one row per page
# ================================================================================

def _reduce_to_page_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse the line-level DataFrame to one row per page.

    Fully vectorized: every per-page signal is a single groupby over the
    page_number key instead of a boolean slice of *df* per page.

    Output columns:
        page_number, label_type, label_value, label_blank,
        docx_section_id (int or None), has_toc_block (bool)
    """
    pn = df["page_number"].astype(int)
    pages = pd.Index(sorted(pn.unique()), name="page_number")
    n_pages = len(pages)

    # Docx: prefer body rows for label extraction; header/footer rows carry
    # their own repeated labels that shouldn't drive section boundaries.
    # label_mask: body rows on pages that have any, all rows elsewhere.
    if "header_footer_type" in df.columns:
        is_body = df["header_footer_type"].astype(str).str.lower().eq("body")
        label_mask = is_body | ~is_body.groupby(pn).transform("any")
    else:
        label_mask = pd.Series(True, index=df.index)

    def _first_per_page(values: pd.Series, mask: pd.Series) -> pd.Series:
        """First value of *values* per page among rows where *mask* holds."""
        sub = values[mask]
        if sub.empty:
            return pd.Series(index=pages, dtype="object")
        return sub.groupby(pn[mask]).first().reindex(pages)

    # Dominant label type — prefer any known format over unknown/None
    if "page_label_type" in df.columns:
        lt = df["page_label_type"].astype("string")
        lt_nonblank = lt.notna() & lt.str.strip().ne("")
        known_first = _first_per_page(lt, label_mask & lt.isin(_KNOWN_LABEL_FORMATS))
        any_first = _first_per_page(lt, label_mask & lt_nonblank)
        type_per_page = known_first.where(known_first.notna(), any_first)
        label_type = [str(v) if pd.notna(v) else None for v in type_per_page]
    else:
        label_type = [None] * n_pages

    # First non-blank raw page label per page (drives the value fallback,
    # blank detection, and the alpha prefix)
    if "page_label" in df.columns:
        pl = df["page_label"].astype("string")
        pl_nonblank = pl.notna() & pl.str.strip().ne("")
        first_label = _first_per_page(pl, label_mask & pl_nonblank)
    else:
        first_label = pd.Series(index=pages, dtype="object")

    # First integer label value on this page.
    # Prefer the pre-parsed page_label_value column (PDF pipeline); fall
    # back to parsing the raw page_label text (HTML pipeline omits the column).
    if "page_label_value" in df.columns:
        plv = pd.to_numeric(df["page_label_value"], errors="coerce")
        value_first = _first_per_page(plv, label_mask & plv.notna())
    else:
        value_first = pd.Series(index=pages, dtype="float64")
    label_value: list[int | None] = [
        int(v) if pd.notna(v)
        else (_parse_label_value(str(lbl), t) if pd.notna(lbl) else None)
        for v, lbl, t in zip(value_first, first_label, label_type)
    ]

    # Is the page label blank?
    if "page_label" in df.columns:
        label_blank = first_label.isna().tolist()
    else:
        label_blank = [t is None for t in label_type]

    # Page-label series id: first value on this page.  PDF pipeline
    # writes page_label_series_id, HTML writes page_label_group_id;
    # docx has neither.  Whichever is present is used.  Scans ALL rows
    # (not label_mask rows): the id sits on one chosen row per page,
    # which may be a header/footer row.
    label_series_id = pd.Series(index=pages, dtype="float64")
    for series_col in ("page_label_series_id", "page_label_group_id"):
        if series_col in df.columns:
            sv = pd.to_numeric(df[series_col], errors="coerce")
            firsts = _first_per_page(sv, sv.notna()).astype("float64")
            label_series_id = label_series_id.where(label_series_id.notna(), firsts)

    # Docx section_id: most common value among all rows on this page
    # (mode tie-break: smallest id, matching Series.mode().iloc[0])
    docx_section_id = pd.Series(index=pages, dtype="float64")
    if "section_id" in df.columns:
        sid = pd.to_numeric(df["section_id"], errors="coerce")
        sid_frame = pd.DataFrame({"p": pn[sid.notna()], "sid": sid.dropna()})
        if not sid_frame.empty:
            counts = sid_frame.groupby(["p", "sid"]).size().reset_index(name="n")
            counts = counts.sort_values(["p", "n", "sid"], ascending=[True, False, True])
            docx_section_id = counts.drop_duplicates("p").set_index("p")["sid"].reindex(pages)

    # Any TOC block on this page?
    if "block_type" in df.columns:
        is_toc = df["block_type"].astype("string").str.lower().isin(_TOC_BLOCK_TYPES)
        has_toc = is_toc.groupby(pn).any().reindex(pages, fill_value=False).tolist()
    else:
        has_toc = [False] * n_pages

    # Leading letter of alpha-prefixed labels ("A-1" → "A", "F-3" → "F")
    label_prefix: list[str | None] = []
    for t, lbl in zip(label_type, first_label):
        prefix = None
        if t in _ALPHA_FORMATS and pd.notna(lbl):
            m = _ALPHA_PREFIX_RE.match(str(lbl).strip())
            if m:
                prefix = m.group(1).upper()
        label_prefix.append(prefix)

    pi = pd.DataFrame({
        "page_number": pages.to_numpy(),
        "label_type": label_type,
        "label_value": label_value,
        "label_blank": label_blank,
        "label_prefix": label_prefix,
        "label_series_id": label_series_id.to_numpy(),
        "docx_section_id": docx_section_id.to_numpy(),
        "has_toc_block": has_toc,
    })

    # Infer series membership for gap pages: every page between the first
    # and last occurrence of a series id belongs to that series, even when
    # the page itself carries no id (e.g. blank chapter-title slides).
    sid_series = pd.to_numeric(pi["label_series_id"], errors="coerce")
    for sid in sid_series.dropna().unique():
        pos = sid_series[sid_series == sid].index
        gap = pi.index[(pi.index >= pos.min()) & (pi.index <= pos.max()) & sid_series.isna()]
        pi.loc[gap, "label_series_id"] = int(sid)

    return pi


# ================================================================================
# Pass A — step 2: detect section boundaries
# ================================================================================

def _detect_boundaries(page_index: pd.DataFrame) -> list[int]:
    """
    Walk the page index and return page numbers where a new temp section begins.

    Boundary triggers (checked in order; first match wins per page transition):

    1. docx_section_id changes between consecutive pages.
    2. page_label_type format changes between two known formats.
    3. Value restart within the same format:
       a. Value drops to 1 from a higher value.
       b. Current page is blank and the next page has value 2
          (the blank page is implicitly "1", next confirms the restart).

    A boundary is suppressed when it would split a page-label series:
    if the nearest labeled page before it and the nearest labeled page
    at/after it share the same label_series_id, the transition (e.g. a
    blank chapter-title slide inside a slide deck) stays in one section.

    The first page is always included as the start of section 1.
    """
    if page_index.empty:
        return []

    rows = page_index.reset_index(drop=True)
    n = len(rows)
    boundaries: list[int] = [int(rows.at[0, "page_number"])]

    has_docx = (
        "docx_section_id" in rows.columns
        and rows["docx_section_id"].notna().any()
    )

    # For each position: last known series id at/before it, next at/after it
    sids = (
        pd.to_numeric(rows["label_series_id"], errors="coerce")
        if "label_series_id" in rows.columns
        else pd.Series([None] * n, dtype="object")
    )
    prev_sid: list[int | None] = [None] * n
    last: int | None = None
    for i in range(n):
        v = sids.iloc[i]
        if pd.notna(v):
            last = int(v)
        prev_sid[i] = last
    next_sid: list[int | None] = [None] * n
    upcoming: int | None = None
    for i in range(n - 1, -1, -1):
        v = sids.iloc[i]
        if pd.notna(v):
            upcoming = int(v)
        next_sid[i] = upcoming

    for i in range(1, n):
        prev = rows.iloc[i - 1]
        curr = rows.iloc[i]
        nxt = rows.iloc[i + 1] if i + 1 < n else None
        page_num = int(curr["page_number"])
        trigger: str | None = None

        # 1. Docx section_id change
        if has_docx:
            p_sid = prev["docx_section_id"]
            c_sid = curr["docx_section_id"]
            if pd.notna(p_sid) and pd.notna(c_sid) and int(p_sid) != int(c_sid):
                trigger = "docx_section_id"

        # 2. Effective label format change.
        #    Blank pages use the sentinel "_blank" so that blank↔non-blank
        #    and format-to-format transitions all fire a boundary.
        #    Consecutive blank pages share the same sentinel → no boundary.
        if trigger is None:
            p_eff = "_blank" if bool(prev["label_blank"]) else (prev["label_type"] or "_unknown")
            c_eff = "_blank" if bool(curr["label_blank"]) else (curr["label_type"] or "_unknown")
            if p_eff != c_eff:
                trigger = "label_format_change"

        # 3a. Value restart: same format, value drops to 1 from > 1
        if trigger is None:
            c_blank = bool(curr["label_blank"])
            p_blank = bool(prev["label_blank"])
            if not c_blank and not p_blank:
                p_type = prev["label_type"]
                c_type = curr["label_type"]
                p_val = prev["label_value"]
                c_val = curr["label_value"]
                if (
                    isinstance(c_type, str) and c_type in _KNOWN_LABEL_FORMATS
                    and p_type == c_type
                    and pd.notna(c_val) and int(c_val) == 1
                    and pd.notna(p_val) and int(p_val) > 1
                ):
                    trigger = "value_restart"

        # 3b. Blank-then-2: curr is blank, next page has value 2,
        #     and the previous page had value > 2 in the same format.
        if trigger is None and bool(curr["label_blank"]) and nxt is not None:
            p_type = prev["label_type"]
            n_type = nxt["label_type"]
            p_val = prev["label_value"]
            n_val = nxt["label_value"]
            if (
                isinstance(p_type, str) and p_type in _KNOWN_LABEL_FORMATS
                and pd.notna(n_type) and str(n_type) == p_type
                and pd.notna(n_val) and int(n_val) == 2
                and pd.notna(p_val) and int(p_val) > 2
            ):
                trigger = "value_restart_blank"

        # Never split a page-label series: suppress the boundary when the
        # labeled pages on both sides belong to the same series.
        if (
            trigger is not None
            and prev_sid[i - 1] is not None
            and next_sid[i] == prev_sid[i - 1]
        ):
            trigger = None

        if trigger is not None:
            boundaries.append(page_num)

    return boundaries


def _pull_blank_prefix_pages(
    page_index: pd.DataFrame,
    boundaries: list[int],
) -> list[int]:
    """
    Shift boundaries backwards over implicitly-numbered blank pages.

    When a section's first page carries label value v > 1, the v-1 pages
    before it are implicitly labeled 1..v-1.  If those immediately
    preceding pages are blank, they belong to this section, not the prior
    one — so the boundary moves back by up to v-1 blank pages, capped at
    _MAX_BLANK_PREFIX_PULL.  A prior boundary whose section is fully
    consumed is dropped (sections merge).
    """
    if len(boundaries) < 2:
        return boundaries

    rows = page_index.reset_index(drop=True)
    pos_of_page = {int(rows.at[i, "page_number"]): i for i in range(len(rows))}

    adjusted = [boundaries[0]]
    for b in boundaries[1:]:
        i = pos_of_page[b]
        row = rows.iloc[i]
        v = row["label_value"]
        if (
            bool(row["label_blank"])
            or row["label_type"] not in _KNOWN_LABEL_FORMATS
            or pd.isna(v) or int(v) <= 1
        ):
            adjusted.append(b)
            continue

        prev_b = adjusted[-1]
        k = i
        pulled = 0
        max_pull = min(int(v) - 1, _MAX_BLANK_PREFIX_PULL)
        while pulled < max_pull and k - 1 >= 0:
            cand = rows.iloc[k - 1]
            if int(cand["page_number"]) < prev_b or not bool(cand["label_blank"]):
                break
            k -= 1
            pulled += 1

        new_b = int(rows.at[k, "page_number"])
        if new_b == prev_b:
            adjusted.pop()  # prior section fully consumed → merge
        adjusted.append(new_b)

    return adjusted


# ================================================================================
# Pass A — step 3: assemble the section index
# ================================================================================

def _assemble_section_index(
    page_index: pd.DataFrame,
    boundaries: list[int],
) -> tuple[pd.DataFrame, dict[int, int]]:
    """
    Produce one row per temp section from the page index and boundary list.

    Output columns:
        temp_section_id, start_page, end_page, n_pages,
        label_format, label_value_start, label_value_end,
        all_blank, has_toc_block, docx_section_ids

    Also returns ``page_to_sid``: a dict mapping every actual page_number
    in the index to its temp_section_id.
    """
    if page_index.empty or not boundaries:
        return pd.DataFrame(), {}

    pi = page_index.set_index("page_number")
    all_pages = sorted(page_index["page_number"].tolist())
    boundary_set = set(boundaries)

    # Map every page to its temp_section_id
    current_sid = 0
    page_to_sid: dict[int, int] = {}
    for p in all_pages:
        if p in boundary_set:
            current_sid += 1
        page_to_sid[p] = current_sid

    has_docx = "docx_section_id" in pi.columns

    records = []
    for sid in range(1, current_sid + 1):
        sec_pages = sorted(p for p, s in page_to_sid.items() if s == sid)
        if not sec_pages:
            continue

        sec_rows = pi.loc[pi.index.isin(sec_pages)]

        # Dominant label format among known types
        known = sec_rows["label_type"].astype("string")
        known = known[known.isin(_KNOWN_LABEL_FORMATS)]
        label_format: str | None = str(known.mode().iloc[0]) if not known.empty else None

        # First and last integer label values
        vals = pd.to_numeric(sec_rows["label_value"], errors="coerce").dropna()
        val_start: int | None = int(vals.iloc[0]) if not vals.empty else None
        val_end: int | None = int(vals.iloc[-1]) if not vals.empty else None

        blank_mask = sec_rows["label_blank"].fillna(True).astype(bool)
        all_blank = bool(blank_mask.all())
        blank_pages = sorted(int(p) for p in sec_rows.index[blank_mask])
        has_toc = bool(sec_rows["has_toc_block"].any())

        # Alpha prefix — dominant letter among pages that have one
        label_prefix: str | None = None
        if label_format in _ALPHA_FORMATS and "label_prefix" in sec_rows.columns:
            prefixes = sec_rows["label_prefix"].dropna()
            if not prefixes.empty:
                label_prefix = str(prefixes.mode().iloc[0])

        docx_ids: str | None = None
        if has_docx:
            uids = sec_rows["docx_section_id"].dropna().unique()
            if len(uids):
                docx_ids = ",".join(str(x) for x in sorted(int(x) for x in uids))

        records.append({
            "temp_section_id": sid,
            "start_page": sec_pages[0],
            "end_page": sec_pages[-1],
            "n_pages": len(sec_pages),
            "label_format": label_format,
            "label_value_start": val_start,
            "label_value_end": val_end,
            "all_blank": all_blank,
            "n_blank": len(blank_pages),
            "blank_pages": blank_pages,
            "has_toc_block": has_toc,
            "label_prefix": label_prefix,
            "docx_section_ids": docx_ids,
        })

    return pd.DataFrame(records), page_to_sid


def _build_temp_sections(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[int, int]]:
    """
    Run Pass A in full.

    Returns:
        (page_index, section_index, page_to_sid)

    ``page_to_sid`` maps every page_number that appears in *df* to its
    temp_section_id.  Using the actual pages (not a range) means non-
    consecutive or gap-bearing page numbering is handled correctly.
    """
    page_index = _reduce_to_page_index(df)
    boundaries = _detect_boundaries(page_index)
    boundaries = _pull_blank_prefix_pages(page_index, boundaries)
    section_index, page_to_sid = _assemble_section_index(page_index, boundaries)
    return page_index, section_index, page_to_sid


# ================================================================================
# Pass B: assign coverpage and last_page labels
# ================================================================================

_MAX_PAGES_COVERPAGE = 5
_MAX_TOC_DISTANCE = 5

# Layout thresholds (scenario 3)
_COVER_CENTER_RATIO_MULT = 2.0
_COVER_DENSITY_RATIO_MAX = 0.5
_COVER_BG_RATIO_MULT = 2.0

_LAST_DENSITY_RATIO_MAX = 0.75
_LAST_BG_RATIO_MULT = 1.5

# SEC text heuristics (scenario 4)
SEC_COVERPAGE_HEURISTICS: list[str] = [
    "SECURITIES AND EXCHANGE COMMISSION",
    "Washington, D.C. 20549",
    "(I.R.S. Employer Identification No.)",
    "(Exact name of Registrant as specified in its charter)",
    "(Name of Registrant as Specified In Its Charter)",
    "(Registrant's telephone number, including area code)",
    "(CUSIP Number of Class of Securities)",
]
SEC_MIN_HITS: int = 3
SEC_CHECKBOX_CHARS: frozenset[str] = frozenset({"☐", "☒"})

_EMAIL_RE = re.compile(r"\b[a-z0-9._%+\-]+@[a-z0-9.\-]+\.[a-z]{2,}\b", re.IGNORECASE)
_PHONE_RE = re.compile(r"(?<!\w)(?:\+?\d[\d().\- ]{6,}\d)(?!\w)")


# ---------------------------------------------------------------------------
# Section-index row builder (used when splitting a section in Pass B)
# ---------------------------------------------------------------------------

def _build_section_row(
    sid: int,
    pages: list[int],
    pi_rows: pd.DataFrame,
    *,
    section: object = pd.NA,
) -> dict:
    """Build one section-index row dict from a list of pages and their page-index rows."""
    known = pi_rows["label_type"].astype("string")
    known = known[known.isin(_KNOWN_LABEL_FORMATS)]
    label_format: str | None = str(known.mode().iloc[0]) if not known.empty else None

    vals = pd.to_numeric(pi_rows["label_value"], errors="coerce").dropna()
    blank_mask = pi_rows["label_blank"].fillna(True).astype(bool)
    blank_pages_list = sorted(int(p) for p in pi_rows.index[blank_mask])

    has_docx = "docx_section_id" in pi_rows.columns
    docx_ids: str | None = None
    if has_docx:
        uids = pi_rows["docx_section_id"].dropna().unique()
        if len(uids):
            docx_ids = ",".join(str(x) for x in sorted(int(x) for x in uids))

    label_prefix: str | None = None
    if label_format in _ALPHA_FORMATS and "label_prefix" in pi_rows.columns:
        prefixes = pi_rows["label_prefix"].dropna()
        if not prefixes.empty:
            label_prefix = str(prefixes.mode().iloc[0])

    return {
        "temp_section_id": sid,
        "start_page": pages[0],
        "end_page": pages[-1],
        "n_pages": len(pages),
        "label_format": label_format,
        "label_value_start": int(vals.iloc[0]) if not vals.empty else None,
        "label_value_end": int(vals.iloc[-1]) if not vals.empty else None,
        "all_blank": bool(blank_mask.all()),
        "n_blank": len(blank_pages_list),
        "blank_pages": blank_pages_list,
        "has_toc_block": bool(pi_rows["has_toc_block"].any()),
        "label_prefix": label_prefix,
        "docx_section_ids": docx_ids,
        "section": section,
    }


# ---------------------------------------------------------------------------
# Splitting helper
# ---------------------------------------------------------------------------

def _apply_page_cover_set(
    section_index: pd.DataFrame,
    page_index: pd.DataFrame,
    page_to_sid: dict[int, int],
    cover_pages: set[int],
) -> tuple[pd.DataFrame, dict[int, int]]:
    """
    Given a set of page numbers identified as coverpage (from scenarios 3/4),
    mark the corresponding sections and split any section where only a prefix
    of its pages belongs to the cover set.

    Returns the updated (section_index, page_to_sid).
    """
    if not cover_pages:
        return section_index, page_to_sid

    pi = page_index.set_index("page_number")
    affected_sids = {page_to_sid[p] for p in cover_pages if p in page_to_sid}

    new_rows: list[dict] = []
    new_page_to_sid: dict[int, int] = {}
    new_sid = 0

    for _, row in section_index.sort_values("start_page").iterrows():
        old_sid = int(row["temp_section_id"])

        if old_sid not in affected_sids:
            new_sid += 1
            d = row.to_dict()
            d["temp_section_id"] = new_sid
            new_rows.append(d)
            for p, s in page_to_sid.items():
                if s == old_sid:
                    new_page_to_sid[p] = new_sid
            continue

        sec_pages = sorted(p for p, s in page_to_sid.items() if s == old_sid)
        in_cover = [p for p in sec_pages if p in cover_pages]
        not_cover = [p for p in sec_pages if p not in cover_pages]

        if not not_cover:
            # Whole section is coverpage
            new_sid += 1
            d = row.to_dict()
            d["temp_section_id"] = new_sid
            d["section"] = "coverpage"
            new_rows.append(d)
            for p in sec_pages:
                new_page_to_sid[p] = new_sid
        else:
            # Split: cover prefix → coverpage, remainder → unlabeled
            new_sid += 1
            cover_sid = new_sid
            new_rows.append(_build_section_row(
                cover_sid, in_cover, pi.loc[pi.index.isin(in_cover)], section="coverpage"
            ))
            for p in in_cover:
                new_page_to_sid[p] = cover_sid

            new_sid += 1
            rest_sid = new_sid
            new_rows.append(_build_section_row(
                rest_sid, not_cover, pi.loc[pi.index.isin(not_cover)]
            ))
            for p in not_cover:
                new_page_to_sid[p] = rest_sid

    return pd.DataFrame(new_rows), new_page_to_sid


# ---------------------------------------------------------------------------
# Coverpage scenarios (operate on section index; scenarios 3/4 also use df)
# ---------------------------------------------------------------------------

def _coverpage_scenario_1(section_index: pd.DataFrame) -> set[int]:
    """
    Leading all-blank sections within the first _MAX_PAGES_COVERPAGE pages.
    Requires that at least one section is NOT all-blank.
    """
    if bool(section_index["all_blank"].all()):
        return set()  # entire document is unlabeled → can't decide here

    result: set[int] = set()
    cumulative = 0
    for _, row in section_index.sort_values("start_page").iterrows():
        if not bool(row["all_blank"]):
            break
        new_total = cumulative + int(row["n_pages"])
        if new_total > _MAX_PAGES_COVERPAGE:
            break
        cumulative = new_total
        result.add(int(row["temp_section_id"]))
    return result


def _coverpage_scenario_2(section_index: pd.DataFrame) -> set[int]:
    """
    Sections that precede the first early TOC section
    (has_toc_block=True and start_page ≤ _MAX_TOC_DISTANCE),
    capped at _MAX_PAGES_COVERPAGE total pages.
    """
    sorted_si = section_index.sort_values("start_page")

    # Find first section with an early TOC
    toc_sid: int | None = None
    for _, row in sorted_si.iterrows():
        if bool(row["has_toc_block"]) and int(row["start_page"]) <= _MAX_TOC_DISTANCE:
            toc_sid = int(row["temp_section_id"])
            break

    if toc_sid is None:
        return set()

    result: set[int] = set()
    cumulative = 0
    for _, row in sorted_si.iterrows():
        if int(row["temp_section_id"]) == toc_sid:
            break
        new_total = cumulative + int(row["n_pages"])
        if new_total > _MAX_PAGES_COVERPAGE:
            break
        cumulative = new_total
        result.add(int(row["temp_section_id"]))
    return result


def _coverpage_scenario_3_pages(
    section_index: pd.DataFrame,
    df: pd.DataFrame,
) -> set[int]:
    """
    Layout-based fallback for fully unlabeled documents.

    Only fires when every section is all-blank (no label anywhere).
    Scores pages 1–_MAX_PAGES_COVERPAGE by center-align ratio, density,
    and background color against document medians.  Returns the contiguous
    prefix where score ≥ 2.
    """
    if not bool(section_index["all_blank"].all()):
        return set()  # labels exist somewhere → not applicable

    if "section" in df.columns:
        # Bail if an early TOC was already marked
        pn = df["page_number"].astype(int)
        has_early_toc = (
            df["block_type"].astype("string").str.lower().isin(_TOC_BLOCK_TYPES)
            & pn.le(_MAX_TOC_DISTANCE)
        ).any() if "block_type" in df.columns else False
        if has_early_toc:
            return set()

    pn = df["page_number"].astype(int)
    have_align = "text_align" in df.columns
    have_chars = "char_count" in df.columns
    have_bg = "background_non_stroking_color" in df.columns

    per_page_total = pn.value_counts(sort=False)

    if have_align:
        is_center = df["text_align"].astype("string").str.strip().str.lower().eq("center")
        center_ratio = (is_center.groupby(pn).sum() / per_page_total).fillna(0.0)
    else:
        center_ratio = pd.Series(0.0, index=per_page_total.index)

    if have_chars:
        chars = pd.to_numeric(df["char_count"], errors="coerce").fillna(0)
        density = chars.groupby(pn).sum()
    else:
        density = pd.Series(0.0, index=per_page_total.index)

    if have_bg:
        bg_blank = df["background_non_stroking_color"].astype("string").str.strip().eq("")
        bg_ratio = ((~bg_blank).groupby(pn).sum() / per_page_total).fillna(0.0)
    else:
        bg_ratio = pd.Series(0.0, index=per_page_total.index)

    med_center = float(center_ratio.median()) if not center_ratio.empty else 0.0
    med_density = float(density.median()) if not density.empty else 0.0
    med_bg = float(bg_ratio.median()) if not bg_ratio.empty else 0.0

    cover_pages: set[int] = set()
    for p in range(int(pn.min()), int(pn.min()) + _MAX_PAGES_COVERPAGE):
        if p not in per_page_total.index:
            break
        score = 0
        r_c = float(center_ratio.get(p, 0.0))
        score += 1 if (med_center <= 0 and r_c >= 0.30) or (med_center > 0 and r_c >= med_center * _COVER_CENTER_RATIO_MULT) else 0
        d = float(density.get(p, 0.0))
        score += 1 if med_density > 0 and d <= med_density * _COVER_DENSITY_RATIO_MAX else 0
        r_bg = float(bg_ratio.get(p, 0.0))
        score += 1 if (med_bg <= 0 and r_bg >= 0.10) or (med_bg > 0 and r_bg >= med_bg * _COVER_BG_RATIO_MULT) else 0
        if score >= 2:
            cover_pages.add(p)
        else:
            break

    return cover_pages


def _coverpage_scenario_4_pages(
    section_index: pd.DataFrame,
    df: pd.DataFrame,
) -> set[int]:
    """
    SEC text-heuristic fallback for fully unlabeled documents.

    Only fires when every section is all-blank.  Requires at least
    SEC_MIN_HITS distinct heuristic lines and one checkbox character
    across the first _MAX_PAGES_COVERPAGE pages.
    """
    if not bool(section_index["all_blank"].all()):
        return set()

    if "text" not in df.columns:
        return set()

    norm_heuristics = {h.strip().lower() for h in SEC_COVERPAGE_HEURISTICS}
    pn = df["page_number"].astype(int)
    cand = df[pn <= _MAX_PAGES_COVERPAGE]
    if cand.empty:
        return set()

    text_stripped = cand["text"].astype("string").str.strip()
    text_lower = text_stripped.str.lower()

    hits_mask = text_lower.isin(norm_heuristics)
    if int(text_lower[hits_mask].nunique()) < SEC_MIN_HITS:
        return set()

    checkbox_pat = "|".join(re.escape(c) for c in SEC_CHECKBOX_CHARS)
    checkbox_mask = text_stripped.str.contains(checkbox_pat, regex=True, na=False)
    if not bool(checkbox_mask.any()):
        return set()

    hit_pages = set(cand.loc[hits_mask, "page_number"].astype(int))
    checkbox_pages = set(cand.loc[checkbox_mask, "page_number"].astype(int))
    last_relevant = max(hit_pages | checkbox_pages)
    return set(range(int(pn.min()), last_relevant + 1))


# ---------------------------------------------------------------------------
# Last-page scenarios
# ---------------------------------------------------------------------------

def _last_page_scenario_1(section_index: pd.DataFrame) -> int | None:
    """
    The trailing section is all-blank and exactly one page long.
    Returns its section_id, or None.
    """
    if bool(section_index["all_blank"].all()):
        return None  # whole document blank

    last = section_index.sort_values("start_page").iloc[-1]
    if bool(last["all_blank"]) and int(last["n_pages"]) == 1:
        return int(last["temp_section_id"])
    return None


def _last_page_has_contact(df: pd.DataFrame, last_page: int) -> bool:
    """Return True if the last page contains an email address or phone number."""
    if "text" not in df.columns:
        return False
    text_rows = df.loc[df["page_number"].astype(int).eq(last_page), "text"]
    blob = " ".join(text_rows.astype(str).dropna())
    if _EMAIL_RE.search(blob):
        return True
    m = _PHONE_RE.search(blob)
    return bool(m) and sum(ch.isdigit() for ch in m.group(0)) >= 9


def _last_page_scenario_2(
    section_index: pd.DataFrame,
    df: pd.DataFrame,
    cover_sids: set[int],
) -> int | None:
    """
    Layout and contact-info scoring for the final page of the document.

    Scores:
    +1  character density is well below the document median
    +1  background-colour ratio is well above the median
    +1  page contains an email address or phone number

    Returns the last section's id when score ≥ 2 and that section is
    exactly one page long, else None.
    """
    pn = df["page_number"].astype(int)

    # If blank pages exist outside coverpage, don't apply this scenario —
    # the trailing blank section would have been caught by scenario 1.
    if "page_label" in df.columns:
        lbl_blank = df["page_label"].astype("string").str.strip().eq("")
        if bool((~lbl_blank).any()):
            blank_pages_series = lbl_blank.groupby(pn).all()
            cover_pages_set = {
                p for _, row in section_index[section_index["temp_section_id"].isin(cover_sids)].iterrows()
                for p in range(int(row["start_page"]), int(row["end_page"]) + 1)
            }
            extra_blanks = {int(p) for p, v in blank_pages_series.items() if v} - cover_pages_set
            if extra_blanks:
                return None

    last_sid_row = section_index.sort_values("start_page").iloc[-1]
    if int(last_sid_row["n_pages"]) != 1:
        return None  # only applies to a single-page tail

    last_page = int(last_sid_row["end_page"])

    have_chars = "char_count" in df.columns
    have_bg = "background_non_stroking_color" in df.columns

    per_page_total = pn.value_counts(sort=False)
    density = (
        pd.to_numeric(df["char_count"], errors="coerce").fillna(0).groupby(pn).sum()
        if have_chars else pd.Series(0.0, index=per_page_total.index)
    )
    bg_blank = df["background_non_stroking_color"].astype("string").str.strip().eq("") if have_bg else None
    bg_ratio = (
        ((~bg_blank).groupby(pn).sum() / per_page_total).fillna(0.0)
        if have_bg else pd.Series(0.0, index=per_page_total.index)
    )

    med_density = float(density.median()) if not density.empty else 0.0
    med_bg = float(bg_ratio.median()) if not bg_ratio.empty else 0.0

    score = 0
    d_last = float(density.get(last_page, 0.0))
    if med_density > 0 and d_last <= med_density * _LAST_DENSITY_RATIO_MAX:
        score += 1
    r_bg = float(bg_ratio.get(last_page, 0.0))
    if (med_bg <= 0 and r_bg >= 0.10) or (med_bg > 0 and r_bg >= med_bg * _LAST_BG_RATIO_MULT):
        score += 1
    if _last_page_has_contact(df, last_page):
        score += 1

    return int(last_sid_row["temp_section_id"]) if score >= 2 else None


# ---------------------------------------------------------------------------
# Main Pass B orchestrator
# ---------------------------------------------------------------------------

def _assign_coverpage_and_last_page(
    section_index: pd.DataFrame,
    page_index: pd.DataFrame,
    df: pd.DataFrame,
    page_to_sid: dict[int, int],
) -> tuple[pd.DataFrame, dict[int, int]]:
    """
    Pass B: detect and mark coverpage and last_page sections.

    Coverpage is tried in four scenarios (first match wins):
      1. Leading all-blank sections
      2. Sections before an early TOC section
      3. Layout scoring (fully unlabeled documents)
      4. SEC text heuristics (fully unlabeled documents)

    Last-page is tried in two scenarios (first match wins):
      1. Single trailing all-blank section
      2. Single trailing section whose last page scores on layout/contact signals
    """
    si = section_index.copy()
    if "section" not in si.columns:
        si["section"] = pd.NA

    # --- Coverpage ---
    cover_sids = _coverpage_scenario_1(si)
    if not cover_sids:
        cover_sids = _coverpage_scenario_2(si)

    if cover_sids:
        si.loc[si["temp_section_id"].isin(cover_sids), "section"] = "coverpage"
    else:
        # Scenarios 3 and 4 return page sets and may need to split a section
        cover_pages = _coverpage_scenario_3_pages(si, df)
        if not cover_pages:
            cover_pages = _coverpage_scenario_4_pages(si, df)
        if cover_pages:
            si, page_to_sid = _apply_page_cover_set(si, page_index, page_to_sid, cover_pages)
            cover_sids = {
                int(row["temp_section_id"])
                for _, row in si.iterrows()
                if str(row.get("section", "")) == "coverpage"
            }

    # --- Last page ---
    last_sid = _last_page_scenario_1(si)
    if last_sid is None:
        last_sid = _last_page_scenario_2(si, df, cover_sids)
    if last_sid is not None:
        si.loc[si["temp_section_id"] == last_sid, "section"] = "last_page"

    return si, page_to_sid


# ================================================================================
# Pass C: assign human-readable section labels
# ================================================================================

def _assign_section_labels(section_index: pd.DataFrame) -> pd.DataFrame:
    """
    Pass C: assign human-readable section labels to unlabeled sections.

    Sections already labeled (coverpage, last_page) are left untouched.
    Assignment order:
      1. Alpha-prefixed  → financials (F), schedules (S), annex (all others)
      2. Longest arabic  → body  (tie-break: earliest start_page)
      3. Remaining arabic before body → front_matter; after → back_matter
      4. Roman before body → front_matter; after → back_matter
      5. Remaining (blank / unknown) before body → front_matter; after → back_matter
      6. Still unassigned → body  (safety net)
    """
    si = section_index.copy()
    if "section" not in si.columns:
        si["section"] = pd.NA

    def _unlabeled(row) -> bool:
        v = row.get("section")
        return pd.isna(v) or str(v).strip() == ""

    # 1. Alpha-prefixed sections
    # Consecutive alpha_numeric sections form a run. Within each run the
    # collective set of distinct prefixes decides the label: sole "F" →
    # financials, sole "S" → schedules, mixed / unknown → annex.
    # alpha_roman sections are always annex and do NOT break a run.
    sorted_idx = si.sort_values("start_page").index.tolist()
    runs: list[list[int]] = []
    current_run: list[int] = []
    for idx in sorted_idx:
        row = si.loc[idx]
        if _unlabeled(row) and row.get("label_format") == "alpha_numeric":
            current_run.append(idx)
        else:
            if current_run:
                runs.append(current_run)
                current_run = []
    if current_run:
        runs.append(current_run)

    for run in runs:
        prefixes = {str(si.at[idx, "label_prefix"] or "").upper() for idx in run}
        prefixes.discard("")
        if prefixes == {"F"}:
            label = "financials"
        elif prefixes == {"S"}:
            label = "schedules"
        else:
            label = "annex"
        for idx in run:
            si.at[idx, "section"] = label

    for idx, row in si.iterrows():
        if not _unlabeled(row):
            continue
        if row.get("label_format") == "alpha_roman":
            si.at[idx, "section"] = "annex"

    # 2. Body = longest unlabeled arabic section; tie-break: earliest start_page
    arabic_candidates: list[tuple[int, int, int]] = []  # (start_page, sid, n_pages)
    for idx, row in si.iterrows():
        if not _unlabeled(row):
            continue
        if row.get("label_format") not in _ARABIC_FORMATS:
            continue
        arabic_candidates.append((int(row["start_page"]), int(row["temp_section_id"]), int(row["n_pages"])))

    # Fallback: no arabic section → longest unlabeled blank/unknown section
    if not arabic_candidates:
        for idx, row in si.iterrows():
            if not _unlabeled(row):
                continue
            fmt = row.get("label_format")
            if pd.notna(fmt) and fmt in _KNOWN_LABEL_FORMATS:
                continue
            arabic_candidates.append((int(row["start_page"]), int(row["temp_section_id"]), int(row["n_pages"])))

    body_sid: int | None = None
    if arabic_candidates:
        best = sorted(arabic_candidates, key=lambda t: (-t[2], t[0]))[0]
        body_sid = best[1]
        si.loc[si["temp_section_id"] == body_sid, "section"] = "body"

    # Reference point: start page of the body section (or document start if no body yet)
    if body_sid is not None:
        body_start = int(si.loc[si["temp_section_id"] == body_sid, "start_page"].iloc[0])
    else:
        unlabeled_starts = si[si.apply(_unlabeled, axis=1)]["start_page"]
        body_start = int(unlabeled_starts.min()) if not unlabeled_starts.empty else 0

    # 3. Remaining arabic sections
    for idx, row in si.iterrows():
        if not _unlabeled(row):
            continue
        if row.get("label_format") not in _ARABIC_FORMATS:
            continue
        si.at[idx, "section"] = "front_matter" if int(row["start_page"]) < body_start else "back_matter"

    # 4. Roman sections
    for idx, row in si.iterrows():
        if not _unlabeled(row):
            continue
        if row.get("label_format") not in _ROMAN_FORMATS:
            continue
        si.at[idx, "section"] = "front_matter" if int(row["start_page"]) < body_start else "back_matter"

    # 5. Remaining blank / unknown sections
    for idx, row in si.iterrows():
        if not _unlabeled(row):
            continue
        if body_sid is None:
            si.at[idx, "section"] = "body"
        else:
            si.at[idx, "section"] = "front_matter" if int(row["start_page"]) < body_start else "back_matter"

    # 6. Safety net
    for idx, row in si.iterrows():
        if _unlabeled(row):
            si.at[idx, "section"] = "body"

    return si


# ================================================================================
# Debug printer
# ================================================================================

_BLANK_PAGES_INLINE_MAX = 8  # show individual page numbers up to this count


def _print_section_index(section_index: pd.DataFrame) -> None:
    """Print the temp section index to stdout as a compact table."""
    if section_index.empty:
        print("(no temp sections detected)")
        return

    n = len(section_index)
    sep = "=" * 90
    print(f"\n{sep}")
    print(f"TEMP SECTION INDEX  ({n} section{'s' if n != 1 else ''})")
    print(sep)
    print(
        f"{'ID':>3}  {'pages':<14}  {'n':>4}  "
        f"{'format':<14}  {'val_start':>9}  {'val_end':>7}  "
        f"{'blank':>5}  {'n_blank':>7}  {'toc':>3}  {'section':<12}  docx_ids"
    )
    print("-" * 90)

    for _, row in section_index.iterrows():
        page_range = f"p{int(row['start_page'])}-{int(row['end_page'])}"
        vs = str(int(row["label_value_start"])) if pd.notna(row["label_value_start"]) else ""
        ve = str(int(row["label_value_end"])) if pd.notna(row["label_value_end"]) else ""
        n_blank = int(row.get("n_blank", 0) or 0)
        blank_flag = "all" if row["all_blank"] else (str(n_blank) if n_blank else "no")
        sec_label = str(row["section"]) if "section" in row.index and pd.notna(row["section"]) else ""
        print(
            f"{int(row['temp_section_id']):>3}  "
            f"{page_range:<14}  "
            f"{int(row['n_pages']):>4}  "
            f"{str(row['label_format'] if pd.notna(row['label_format']) else ''):14}  "
            f"{vs:>9}  "
            f"{ve:>7}  "
            f"{blank_flag:>5}  "
            f"{n_blank:>7}  "
            f"{'yes' if row['has_toc_block'] else 'no':>3}  "
            f"{sec_label:<12}  "
            f"{str(row['docx_section_ids'] or '')}"
        )
        # Annotate individual blank page numbers when there are only a few
        blank_pages = row.get("blank_pages") or []
        if 0 < n_blank <= _BLANK_PAGES_INLINE_MAX:
            pages_str = ", ".join(f"p{p}" for p in blank_pages)
            print(f"     └─ blank: {pages_str}")

    print(f"{sep}\n")


# ================================================================================
# Public API
# ================================================================================

def classify_sections(df: pd.DataFrame, *, debug: bool = False) -> pd.DataFrame:
    """
    Assign a ``section`` to every row in *df*.

    Pass A — build temp_section_id index from page-label boundaries.
    Pass B — detect and mark coverpage and last_page sections.

    Parameters
    ----------
    df:
        Line-level DataFrame.  Must contain ``page_number``.
        Optional columns that improve accuracy: ``page_label``,
        ``page_label_type``, ``page_label_value``, ``section_id``
        (docx only), ``block_type``, ``header_footer_type`` (docx only),
        ``text_align``, ``char_count``, ``background_non_stroking_color``,
        ``text``.
    debug:
        When True, print the section index table (post Pass B) to stdout.

    Returns
    -------
    pd.DataFrame
        Copy of *df* with ``temp_section_id`` (int) and ``section`` columns.
        Rows not covered by any detected section receive ``pd.NA``.
    """
    if df.empty or "page_number" not in df.columns:
        out = df.copy()
        out["temp_section_id"] = pd.NA
        out["section"] = pd.NA
        return out

    out = df.copy()

    # Pass A
    page_index, section_index, page_to_sid = _build_temp_sections(out)

    # Pass B
    section_index, page_to_sid = _assign_coverpage_and_last_page(
        section_index, page_index, out, page_to_sid
    )

    # Pass C
    section_index = _assign_section_labels(section_index)

    if debug:
        _print_section_index(section_index)

    if page_to_sid:
        pn = out["page_number"].astype(int)
        out["temp_section_id"] = pn.map(page_to_sid)

        # Build page → section mapping from the updated index
        sid_to_section: dict[int, str] = {
            int(row["temp_section_id"]): str(row["section"])
            for _, row in section_index.iterrows()
            if pd.notna(row.get("section"))
        }
        page_to_section = {
            p: sid_to_section[s] for p, s in page_to_sid.items() if s in sid_to_section
        }
        out["section"] = pn.map(page_to_section)
    else:
        out["temp_section_id"] = pd.NA
        out["section"] = pd.NA

    return out
