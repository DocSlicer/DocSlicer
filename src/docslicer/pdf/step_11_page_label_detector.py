# step_03a_pdf_page_label_candidates.py
"""
Step 03A — Mark page label candidates (cells-based)

Goal (Phase 1 only):
(1) Mark candidates (no series scoring yet)

Rules:
- For each page_number:
  - Take cells from top _MAX_LINES_TOP line_id’s
  - Take cells from bottom _MAX_LINES_BOTTOM line_id’s
  - Exclude line_id’s with more cells than _MAX_CELLS_PER_LINE (likely a table)

Candidate extraction:
- Normalize tokens
- Try extracting candidate tokens from a cell by splitting on:
  dash, pipe, whitespace, brackets/parens wrappers, etc.

If a candidate matches a page label type:
→ add columns:
  - page_label_raw            (full cell text)
  - page_label_candidate      (normalized token)
  - page_label_type           (pattern name)
  - page_label_value          (parsed int best-effort)
  - page_label_cell_sharing   (bool; True if the label shares a cell with other text)
  - page_label_wrapper        ("plain" | "parens" | "dashes" | "brackets")
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np


# ================================================================================
# STEP 1: Mark page label candidates (cells-based)
# ================================================================================

# =====================
# CONFIG
# =====================

_MAX_LINES_TOP = 5
_MAX_LINES_BOTTOM = 5
_MAX_CELLS_PER_LINE = 5

# Marked-content tag that tagged PDFs use for page furniture (running
# headers/footers, pagination). When present, page numbers almost always
# carry it, so we can find them without positional heuristics.
_ARTIFACT_BDC_TAG = "artifact"


# =====================
# PAGE LABEL PATTERNS (EXPECTED SHAPE)
# =====================

@dataclass(frozen=True)
class PageLabelPattern:
    name: str
    compiled: re.Pattern


@dataclass(frozen=True)
class PageLabelConfig:
    max_length: int
    patterns: tuple[PageLabelPattern, ...]


# =====================
# NORMALIZATION + PARSING
# =====================

_ROMAN_MAP = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}


def _roman_to_int(s: str) -> Optional[int]:
    if not s:
        return None
    t = re.sub(r"[^IVXLCDM]", "", str(s).upper())
    if not t:
        return None

    total = 0
    prev = 0
    for ch in reversed(t):
        v = _ROMAN_MAP.get(ch, 0)
        if v < prev:
            total -= v
        else:
            total += v
            prev = v

    if total <= 0:
        return None
    return total


def _normalize_token(s: str) -> str:
    """
    Normalization used for comparisons + matching.
    - Strip
    - Normalize unicode dashes to "-"
    - Collapse whitespace
    - Strip trailing punctuation
    - Collapse whitespace around hyphen
    """
    t = str(s or "").strip()
    if not t:
        return ""

    t = t.replace("–", "-").replace("—", "-")
    t = re.sub(r"[.:;,]+$", "", t).strip()
    t = re.sub(r"\s+", " ", t)
    t = re.sub(r"\s*-\s*", "-", t)

    return t


def _match_page_label_type(token: str, cfg: PageLabelConfig) -> str:
    for p in cfg.patterns:
        if p.compiled.fullmatch(token):
            return p.name
    return "unknown"


def _parse_value_int(token: str, label_type: str) -> Optional[int]:
    """
    Best-effort int for ordering/scoring in later steps.
    - arabic -> int
    - roman -> int(roman)
    - alpha_numeric -> numeric part
    - alpha_roman -> roman part
    - roman_numeric -> composite roman_prefix*10000 + numeric (keeps order within prefix)
    - arabic_sub -> major*1000 + minor (e.g. 347/1 → 347001)
    """
    if not token:
        return None

    u = token.upper()

    if label_type == "arabic":
        try:
            return int(u)
        except Exception:
            return None

    if label_type == "roman":
        return _roman_to_int(u)

    if label_type == "alpha_numeric":
        m = re.match(r"^([A-Z]+)-?(\d+)$", u)
        if not m:
            return None
        return int(m.group(2))

    if label_type == "alpha_roman":
        m = re.match(r"^([A-Z]+)-([IVXLCDM]+)$", u)
        if not m:
            return None
        return _roman_to_int(m.group(2))

    if label_type == "roman_numeric":
        m = re.match(r"^([IVXLCDM]+)-(\d+)$", u)
        if not m:
            return None
        rp = _roman_to_int(m.group(1))
        if rp is None:
            return None
        return rp * 10000 + int(m.group(2))

    if label_type == "arabic_sub":
        m = re.match(r"^(\d+)/(\d+)$", u)
        if not m:
            return None
        return int(m.group(1)) * 1000 + int(m.group(2))

    return None


# =====================
# WRAPPER + EXTRACTION
# =====================

_WRAPPER_PARENS_RE = re.compile(r"^\((.+)\)$")
_WRAPPER_BRACKETS_RE = re.compile(r"^\[(.+)\]$|^\{(.+)\}$")
_WRAPPER_DASHES_RE = re.compile(r"^-\s*(.+?)\s*-$")


def _detect_wrapper(raw: str, extracted_segment: str) -> str:
    """
    Determine the wrapper style based on the *segment* we extracted from the raw cell.
    """
    seg = str(extracted_segment or "").strip()
    if not seg:
        return "plain"

    # Normalize unicode dashes to ASCII hyphen for wrapper tests
    seg = seg.replace("–", "-").replace("—", "-")

    if _WRAPPER_PARENS_RE.match(seg):
        return "parens"
    if _WRAPPER_BRACKETS_RE.match(seg):
        return "brackets"
    if _WRAPPER_DASHES_RE.match(seg):
        return "dashes"

    # Also treat cases like "— 2 —" or "-2-" after normalization
    if re.match(r"^-\s*\d+\s*-$", seg):
        return "dashes"

    return "plain"


def _unwrap_wrapper(segment: str) -> str:
    """
    Remove simple wrappers if present: (x), [x], {x}, - x -
    """
    s = str(segment or "").strip()
    if not s:
        return ""

    s = s.replace("–", "-").replace("—", "-")

    m = _WRAPPER_PARENS_RE.match(s)
    if m:
        return str(m.group(1)).strip()

    m = _WRAPPER_BRACKETS_RE.match(s)
    if m:
        inner = m.group(1) if m.group(1) is not None else m.group(2)
        return str(inner).strip() if inner is not None else ""

    m = _WRAPPER_DASHES_RE.match(s)
    if m:
        return str(m.group(1)).strip()

    return s


# Split candidates from a cell:
# - strong separators: pipe / long dash / hyphen blocks
# - then whitespace tokens
_SPLIT_PRIMARY_RE = re.compile(r"[|]+|(?:\s+-\s+)|(?:\s+—\s+)|(?:\s+–\s+)")

# Reject classic heading patterns like: "4. Title", "2.3 Revenue"
_HEADING_LIKE_RE = re.compile(r"^\s*\d+(?:\.\d+)*\.\s+\S+")

# Reject alpha/roman list-item markers like: "c. Revenue", "ii. Summary", "I. Introduction"
_LIST_ITEM_ALPHA_RE = re.compile(r"^\s*[a-zA-Z]{1,5}\.\s+\S+")

# Reject "I" used as the sentence-starting pronoun, not roman numeral one.
# Examples: "I have reviewed...", "I, James Dimon, certify that:"
_PROSE_START_I_RE = re.compile(r"^\s*I\s*,|^\s*I\s+[a-z]")

# Whole cells longer than this are never page labels; guards against long prose
# cells that happen to end in a digit or short token.
_MAX_CELL_LEN = 40

# "Page 1 of 19", "2 of 20" — end-anchored, 1-4 digits only.
# End-anchored to avoid "3 of 10 items"; validated post-match: 0 < N ≤ M.
_N_OF_M_RE = re.compile(r"\b(\d{1,4})\s+of\s+(\d{1,4})\s*$", re.IGNORECASE)

# Candidate type preference: arabic/arabic_sub beat roman so "L 347/1" → 347/1, not L.
_TYPE_PRIO: dict[str, int] = {"arabic": 0, "arabic_sub": 0, "alpha_numeric": 1, "roman_numeric": 2, "alpha_roman": 3, "roman": 4}


def _try_extract_embedded_label(text: str) -> Optional[str]:
    """
    Try to extract a page number token from compound text that would otherwise
    yield the wrong candidate via generic splitting.

    Supported patterns (checked in order of specificity):

    1. **N-of-M suffix** — ``"Page 2 of 20"``, ``"2 of 20"``
       Extracts N.  Requires ``0 < N ≤ M`` to limit false positives.

    2. **Pipe-terminated label** — ``"Apple Inc. | Q1 2026 | 2"``
       Takes the last segment after the final ``"|"``.
       Type-matching by the caller rejects non-label segments.

    Returns:
        Extracted candidate string (un-normalized), or ``None``.
    """
    if not text:
        return None

    stripped = text.strip()

    # Pattern 1: N of M — most specific, check first.
    m = _N_OF_M_RE.search(stripped)
    if m:
        n, total = int(m.group(1)), int(m.group(2))
        if 0 < n <= total:
            return m.group(1)

    # Pattern 2: pipe-terminated — take the last segment.
    if "|" in stripped:
        last = stripped.rsplit("|", 1)[-1].strip()
        if last:
            return last

    return None


# lowercase word, dot or comma, space, Uppercase word
# Examples:
#   "dividends. This"
#   "option, However"
#   "value. The"
_SENTENCE_LOWER_DOT_UPPER_RE = re.compile(r"[a-z]+\s*[.,]\s+[A-Z][a-z]+")


def _extract_candidate_tokens(raw_text: str, max_len: int, has_link: bool) -> list[Tuple[str, bool, str]]:
    """
    Return list of (candidate_token_normalized, cell_sharing_bool, wrapper_type).
    Ordered by preference.

    cell_sharing_bool: True if the candidate likely shares the cell with other text.
    wrapper_type: plain/parens/dashes/brackets based on the extracted segment.
    """
    raw = str(raw_text or "")
    if not raw.strip():
        return []

    # Hard exclude: looks like a numbered heading
    # Examples: "4. Revenue", "2.3. Financial summary"
    if _HEADING_LIKE_RE.match(raw):
        return []

    # Hard exclude: looks like an alpha/roman list-item marker
    # Examples: "c. Revenue", "ii. Summary", "I. Introduction"
    if _LIST_ITEM_ALPHA_RE.match(raw):
        return []

    # Hard exclude: "I" as the sentence-starting pronoun, not roman numeral one
    # Example: "I have reviewed the attached document."
    if _PROSE_START_I_RE.match(raw):
        return []

    # Hard exclude: prose-like cells (must contain . or ,)
    if _SENTENCE_LOWER_DOT_UPPER_RE.search(raw):
        return []

    # Hard exclude: cells that contain links
    # Page labels are never hyperlinks; links are almost always body content or TOC
    if bool(has_link):
        return []

    # Hard exclude: whole cell is too long to plausibly be a page label
    if len(raw.strip()) > _MAX_CELL_LEN:
        return []


    s = raw.strip()

    # Prefer: full-string segment, then last segment after separators, then first segment,
    # then whitespace last token, then whitespace first token.
    segs: list[str] = []

    # 0) Embedded patterns — highest priority so they beat generic ws/pipe splits.
    #    Handles "Page 1 of 19" (extracts "1", not "19") and
    #    "Apple Inc. | Q1 2026 | 2" (extracts "2").
    embedded = _try_extract_embedded_label(s)
    if embedded:
        segs.append(embedded)

    # 1) Whole cell
    segs.append(s)

    # 2) Primary splits (pipe, spaced dashes)
    primary_parts = [p.strip() for p in _SPLIT_PRIMARY_RE.split(s) if p and p.strip()]
    if primary_parts:
        # prioritize end-of-cell (common: "... | 51")
        segs.extend([primary_parts[-1], primary_parts[0]])

    # 3) Hyphen splits without spaces (II-3, F-12) should not be broken;
    # but strings like "Page-12" might appear. We handle whitespace tokenization next.
    ws_parts = [p.strip() for p in re.split(r"\s+", s) if p and p.strip()]
    if ws_parts:
        segs.extend([ws_parts[-1], ws_parts[0]])

    # Deduplicate while preserving order
    seen = set()
    ordered_segs = []
    for seg in segs:
        if seg not in seen:
            ordered_segs.append(seg)
            seen.add(seg)

    out: list[Tuple[str, bool, str]] = []
    for seg in ordered_segs:
        wrapper = _detect_wrapper(raw, seg)
        unwrapped = _unwrap_wrapper(seg)
        norm = _normalize_token(unwrapped)
        if not norm:
            continue
        if len(norm) > max_len:
            continue

        # cell sharing heuristic:
        # - if the extracted segment != whole-cell trimmed, or if primary_parts indicates multiple parts
        cell_sharing = (seg.strip() != s.strip()) or (len(primary_parts) >= 2)

        out.append((norm, bool(cell_sharing), wrapper))

    return out


# =====================
# Main Candidate Orchestrator
# =====================

def _resolve_candidate(
    raw_text: str,
    has_link: bool,
    max_len: int,
    cfg: PageLabelConfig,
) -> Optional[Tuple[str, str, Optional[int], bool, str]]:
    """
    Resolve a single cell's best page-label candidate.

    Returns (token_norm, label_type, value_int, cell_sharing, wrapper) or None.
    Depends only on (raw_text, has_link, max_len, cfg), so callers memoize it on
    the text — running headers/footers repeat verbatim across many pages, and the
    regex work here is the dominant cost of candidate marking.
    """
    candidates = _extract_candidate_tokens(raw_text, max_len=max_len, has_link=has_link)
    if not candidates:
        return None

    # Pick the best candidate by type preference, then extraction order.
    # arabic/arabic_sub > alpha_numeric > roman_numeric > alpha_roman > roman
    # This prevents a single-letter roman (e.g. "L" from "L 347/1") from
    # winning over a later arabic_sub token ("347/1").
    chosen = None
    chosen_prio = 99
    for token_norm, cell_sharing, wrapper in candidates:
        label_type = _match_page_label_type(token_norm, cfg)
        if label_type == "unknown":
            continue

        value_int = _parse_value_int(token_norm, label_type)
        prio = _TYPE_PRIO.get(label_type, 99)
        # Accept if: no candidate yet, OR this type is strictly better
        # (only upgrade; don't downgrade to a worse type for a parseable value)
        if chosen is None or (prio < chosen_prio and value_int is not None):
            chosen = (token_norm, label_type, value_int, bool(cell_sharing), wrapper)
            chosen_prio = prio
            if label_type in ("arabic", "arabic_sub") and value_int is not None:
                break

    return chosen


def mark_pdf_page_label_candidates(
    df_cells: pd.DataFrame,
    page_label_config: PageLabelConfig,
    max_lines_top: int = _MAX_LINES_TOP,
    max_lines_bottom: int = _MAX_LINES_BOTTOM,
    max_cells_per_line: int = _MAX_CELLS_PER_LINE,
    artifact_only: bool = False,
) -> pd.DataFrame:
    """
    Phase (1): Mark candidates only.

    Required columns:
      - page_number
      - line_id
      - cell_id
      - text

    artifact_only:
      When True, restrict candidate extraction to cells whose ``bdc_tag`` is
      ``"Artifact"`` and bypass the positional config (top/bottom lines,
      max cells per line). Tagged PDFs mark page furniture as Artifacts, so
      any page number living there can be found without those heuristics.
      If the frame has no ``bdc_tag`` column, no candidates are marked.

    Adds/overwrites columns:
      - page_label_raw
      - page_label_candidate
      - page_label_type
      - page_label_value
      - page_label_cell_sharing
      - page_label_wrapper

    Notes:
      - Multiple candidate sources can exist per page; we mark every cell that yields
        at least one valid candidate (best match per cell by extraction order).
      - This does NOT decide final series / best path — just marks candidates.
    """
    out = df_cells.copy()

    required = {"page_number", "line_id", "cell_id", "text"}
    missing = required - set(out.columns)
    if missing:
        raise ValueError(f"mark_pdf_page_label_candidates: missing columns: {sorted(missing)}")

    # Ensure output columns exist
    out["page_label_raw"] = None
    out["page_label_candidate"] = None
    out["page_label_type"] = None
    out["page_label_value"] = None
    out["page_label_cell_sharing"] = False
    out["page_label_wrapper"] = None

    max_len = int(page_label_config.max_length)

    # Artifact pre-pass: restrict to page-furniture cells and skip positional
    # heuristics. If the frame lacks bdc_tag we can't run it — return with no
    # candidates so the caller falls back to the positional pass.
    if artifact_only:
        if "bdc_tag" not in out.columns:
            return out
        artifact_mask = out["bdc_tag"].astype(str).str.strip().str.lower() == _ARTIFACT_BDC_TAG
        if not artifact_mask.any():
            return out

    # ---- Select candidate cells (focus set), fully vectorized ----
    if artifact_only:
        focus_mask = artifact_mask.to_numpy()
    else:
        # Cells per (page, line): a table-ish line has many cells.
        cells_in_line = (
            out.groupby(["page_number", "line_id"], sort=False)["cell_id"].transform("size")
        )

        # Top/bottom line selection without a per-page Python loop. Dense rank
        # over line_id within each page reproduces "first N / last N distinct
        # line_ids": duplicate lines share a rank, and NaN line_ids rank NaN
        # (never selected) — matching the previous dropna()+sorted() logic.
        asc_rank = out.groupby("page_number")["line_id"].rank(method="dense", ascending=True)
        desc_rank = out.groupby("page_number")["line_id"].rank(method="dense", ascending=False)
        in_boundary = (asc_rank <= max_lines_top) | (desc_rank <= max_lines_bottom)

        focus_mask = (
            in_boundary.to_numpy()
            & (cells_in_line.fillna(0).astype(int).to_numpy() <= int(max_cells_per_line))
        )

    focus_idx = out.index[focus_mask]
    if len(focus_idx) == 0:
        return out

    # Extract candidates once per unique (text, has_link). The regex-heavy
    # resolution is memoized because running headers/footers repeat verbatim
    # across pages, so most cells hit the cache instead of re-running the regexes.
    texts = out["text"].to_numpy(dtype=object)[focus_mask]
    if "has_link" in out.columns:
        links = out["has_link"].fillna(False).astype(bool).to_numpy()[focus_mask]
    else:
        links = np.zeros(len(focus_idx), dtype=bool)

    _MISS = object()
    cache: Dict[Tuple[str, bool], Optional[Tuple]] = {}
    m_idx, m_raw, m_tok, m_typ, m_val, m_share, m_wrap = [], [], [], [], [], [], []

    for pos, ridx in enumerate(focus_idx):
        raw = texts[pos]
        if raw is None:
            continue
        raw_s = str(raw)
        if not raw_s.strip():
            continue

        key = (raw_s, bool(links[pos]))
        res = cache.get(key, _MISS)
        if res is _MISS:
            res = _resolve_candidate(raw_s, key[1], max_len, page_label_config)
            cache[key] = res
        if res is None:
            continue

        token_norm, label_type, value_int, cell_sharing, wrapper = res
        m_idx.append(ridx)
        m_raw.append(raw_s)
        m_tok.append(token_norm)
        m_typ.append(label_type)
        m_val.append(value_int)
        m_share.append(cell_sharing)
        m_wrap.append(wrapper)

    # Bulk-assign (one write per column) instead of scalar .at in a loop.
    if m_idx:
        out.loc[m_idx, "page_label_raw"] = m_raw
        out.loc[m_idx, "page_label_candidate"] = m_tok
        out.loc[m_idx, "page_label_type"] = m_typ
        out.loc[m_idx, "page_label_value"] = m_val
        out.loc[m_idx, "page_label_cell_sharing"] = m_share
        out.loc[m_idx, "page_label_wrapper"] = m_wrap

    return out


# ================================================================================
# STEP 2: Assign page label scores
# ================================================================================

import numpy as np
import pandas as pd


def add_page_label_score(
    df: pd.DataFrame,
    *,
    window_pages: int = 5,          # +/- N pages
    local_fp_bonus: float = 0.5,
    local_char_bonus: float = 0.5,
) -> pd.DataFrame:
    """
    Adds/overwrites page_label_score for ALL candidate rows (page_label_candidate != null).

    Global scoring (doc-wide):
      - type: arabic +0.2, roman -0.2
      - wrapper: plain +0.2, parens -0.2
      - page_label_value == page_number: +1
      - doc fingerprint mode (type, cell_sharing, wrapper): +1 if matches
      - global median char_count:
          per page, candidate(s) closest to global median get +1

    Local scoring (rolling window around each page):
      - local fingerprint mode within [p-window_pages, p+window_pages]: +0.5 if matches
      - local median char_count within that window:
          per page, candidate(s) closest to local median get +0.5

    Boundary bonus (+0.5):
      - closest candidate to top/bottom boundary per page
      - uses y_top/y_bottom if available; else line_id fallback
    """
    out = df.copy()

    required = {
        "page_number",
        "page_label_candidate",
        "page_label_type",
        "page_label_value",
        "page_label_cell_sharing",
        "page_label_wrapper",
        "char_count",
    }
    missing = required - set(out.columns)
    if missing:
        raise ValueError(f"add_page_label_score: missing columns: {sorted(missing)}")

    cand_mask = out["page_label_candidate"].notna() & (out["page_label_candidate"].astype(str).str.len() > 0)
    out["page_label_score"] = np.nan
    if not cand_mask.any():
        return out

    c = out.loc[cand_mask].copy()

    # Normalize types/wrappers for consistent fingerprinting
    c["_t"] = c["page_label_type"].astype(str).str.lower()
    c["_w"] = c["page_label_wrapper"].astype(str).str.lower()
    c["_share"] = c["page_label_cell_sharing"].astype(bool)
    c["_fp"] = c["_t"] + "|" + c["_share"].astype(int).astype(str) + "|" + c["_w"]

    # Numeric helpers
    c["_page_int"] = pd.to_numeric(c["page_number"], errors="coerce")
    c["_val_int"] = pd.to_numeric(c["page_label_value"], errors="coerce")
    c["_char"] = pd.to_numeric(c["char_count"], errors="coerce")

    score = pd.Series(0.0, index=c.index)

    # ---------------------------
    # Global (doc-wide) scoring
    # ---------------------------

    # arabic/arabic_sub +0.2, roman -0.2
    score += np.where(c["_t"].isin(["arabic", "arabic_sub"]), 0.2, 0.0)
    score += np.where(c["_t"] == "roman", -0.2, 0.0)

    # plain +0.2, parens -0.2
    score += np.where(c["_w"] == "plain", 0.2, 0.0)
    score += np.where(c["_w"] == "parens", -0.2, 0.0)

    # value == page_number +1
    score += np.where(
        c["_page_int"].notna() & c["_val_int"].notna() & (c["_page_int"] == c["_val_int"]),
        1.0,
        0.0,
    )

    # doc fingerprint mode +1
    fp_counts = c["_fp"].value_counts(dropna=True)
    if not fp_counts.empty:
        top_ct = fp_counts.iloc[0]
        top_fps = fp_counts[fp_counts == top_ct].index.tolist()
        top_fps.sort()
        fp_mode = top_fps[0]
        score += np.where(c["_fp"] == fp_mode, 1.0, 0.0)

    # global median char_count: per page, closest candidate(s) get +1
    global_median_char = float(c["_char"].dropna().median()) if c["_char"].notna().any() else np.nan
    if np.isfinite(global_median_char):
        dist_global = (c["_char"] - global_median_char).abs()
        min_dist_global_on_page = dist_global.groupby(c["page_number"]).transform("min")
        score += np.where(dist_global.notna() & (dist_global == min_dist_global_on_page), 1.0, 0.0)

    # ---------------------------
    # Local (rolling window) scoring
    # ---------------------------

    # We compute, for each page p:
    # - local fp mode (within ±window_pages pages)
    # - local median char (within ±window_pages pages)
    #
    # Then:
    # - if candidate fp == local mode: +local_fp_bonus
    # - per page: candidate(s) closest to local median char: +local_char_bonus

    # Build per-page aggregates (candidates only)
    # We want a stable page axis even if some pages have no candidates.
    pages = c["_page_int"].dropna().astype(int)
    if pages.empty:
        # Can't compute local window without numeric pages
        pass
    else:
        page_min = int(pages.min())
        page_max = int(pages.max())

        # Vectorized window aggregation over a contiguous page axis.
        # The window [p-w, p+w] in page-number space equals a ±w window in axis
        # index space (axis is contiguous), so both the local fp-mode and the
        # local median char are windowed sums over per-page histograms computed
        # once via cumsum — no per-page Python/pandas work inside the loop.
        g = c[c["_page_int"].notna()].copy()
        g["_page_int"] = g["_page_int"].astype(int)

        P = page_max - page_min + 1
        w = int(window_pages)
        axis_pos = g["_page_int"].to_numpy() - page_min  # axis row per candidate

        # Per-axis-row window bounds (clipped to the axis; pages outside it have
        # no candidates, so clipping is equivalent to the old index-membership test).
        idx = np.arange(P)
        lo_i = np.maximum(idx - w, 0)
        hi_i = np.minimum(idx + w, P - 1)

        def _window_sum(cumsum_2d: np.ndarray) -> np.ndarray:
            """Rows lo_i..hi_i (inclusive) summed, from a cumsum along axis 0."""
            upper = cumsum_2d[hi_i]
            lower = np.where((lo_i > 0)[:, None], cumsum_2d[lo_i - 1], 0)
            return upper - lower

        # --- Local fp mode (tie -> lexicographically smallest) ---
        # Columns ordered by sorted fp so argmax's first-max rule yields the
        # lexicographically smallest fingerprint on ties.
        uniq_fps = sorted(g["_fp"].astype(str).unique().tolist())
        fp_code = {fp: i for i, fp in enumerate(uniq_fps)}
        codes = g["_fp"].astype(str).map(fp_code).to_numpy()

        fp_hist = np.zeros((P, len(uniq_fps)), dtype=np.int64)
        np.add.at(fp_hist, (axis_pos, codes), 1)
        fp_win = _window_sum(np.cumsum(fp_hist, axis=0))
        fp_totals = fp_win.sum(axis=1)
        fp_arg = fp_win.argmax(axis=1)

        local_mode_fp = {
            int(page_min + i): (uniq_fps[fp_arg[i]] if fp_totals[i] > 0 else None)
            for i in range(P)
        }

        # --- Local median char ---
        char_vals = g["_char"].to_numpy()
        finite = np.isfinite(char_vals)
        if not finite.any():
            local_median_char = {int(page_min + i): np.nan for i in range(P)}
        else:
            cv = char_vals[finite].astype(np.int64)
            cpos = axis_pos[finite]
            cmin = int(cv.min())
            char_hist = np.zeros((P, int(cv.max()) - cmin + 1), dtype=np.int64)
            np.add.at(char_hist, (cpos, cv - cmin), 1)
            char_win = _window_sum(np.cumsum(char_hist, axis=0))
            counts = char_win.sum(axis=1)
            cum = np.cumsum(char_win, axis=1)

            local_median_char = {}
            for i in range(P):
                n = int(counts[i])
                if n == 0:
                    local_median_char[int(page_min + i)] = np.nan
                    continue
                # Lower/upper middle order-statistics (0-indexed). pandas median
                # averages the two middle values for even n; equal for odd n.
                v1 = int(np.searchsorted(cum[i], (n - 1) // 2, side="right"))
                v2 = int(np.searchsorted(cum[i], n // 2, side="right"))
                local_median_char[int(page_min + i)] = (v1 + v2 + 2 * cmin) / 2.0

        # Apply local fp bonus
        g_page_int = c["_page_int"]
        lp = g_page_int.map(local_mode_fp)
        score += np.where(lp.notna() & (c["_fp"] == lp), float(local_fp_bonus), 0.0)

        # Apply local char bonus: per page, closest to local median char
        lm = g_page_int.map(local_median_char)
        lm = pd.to_numeric(lm, errors="coerce")
        dist_local = (c["_char"] - lm).abs()

        # For each page, find min distance among candidates on that page (only when local median exists)
        min_dist_local_on_page = dist_local.groupby(c["page_number"]).transform("min")
        score += np.where(
            lm.notna() & dist_local.notna() & (dist_local == min_dist_local_on_page),
            float(local_char_bonus),
            0.0,
        )

    # ---------------------------
    # Boundary bonus (+0.5)
    # ---------------------------

    boundary_bonus = 0.5

    has_geom = ("y_top" in out.columns) and ("y_bottom" in out.columns)
    if has_geom:
        y_top = pd.to_numeric(c["y_top"], errors="coerce")
        y_bottom = pd.to_numeric(c["y_bottom"], errors="coerce")

        page_min_top = y_top.groupby(c["page_number"]).transform("min")
        page_max_bottom = y_bottom.groupby(c["page_number"]).transform("max")

        dist_to_top = (y_top - page_min_top).abs()
        dist_to_bottom = (page_max_bottom - y_bottom).abs()
        dist_boundary = pd.concat([dist_to_top, dist_to_bottom], axis=1).min(axis=1)

        min_dist_boundary_on_page = dist_boundary.groupby(c["page_number"]).transform("min")
        score += np.where(dist_boundary.notna() & (dist_boundary == min_dist_boundary_on_page), boundary_bonus, 0.0)

    else:
        if "line_id" in out.columns:
            line_id = pd.to_numeric(c["line_id"], errors="coerce")
            page_min_line = line_id.groupby(c["page_number"]).transform("min")
            page_max_line = line_id.groupby(c["page_number"]).transform("max")

            dist_to_top = (line_id - page_min_line).abs()
            dist_to_bottom = (page_max_line - line_id).abs()
            dist_boundary = pd.concat([dist_to_top, dist_to_bottom], axis=1).min(axis=1)

            min_dist_boundary_on_page = dist_boundary.groupby(c["page_number"]).transform("min")
            score += np.where(dist_boundary.notna() & (dist_boundary == min_dist_boundary_on_page), boundary_bonus, 0.0)

    # Write back
    out.loc[c.index, "page_label_score"] = score.astype(float)

    # Cleanup helper cols if they existed in original df (they didn't), but keep out clean anyway
    out = out.drop(columns=[col for col in ["_t", "_w", "_share", "_fp", "_page_int", "_val_int", "_char"] if col in out.columns],
                   errors="ignore")

    return out





# ==============================================================================
# Public API
# ==============================================================================

def pick_pdf_page_label_winners_and_validate(
    df: pd.DataFrame,
    *,
    page_col: str = "page_number",
    score_col: str = "page_label_score",
    token_col: str = "page_label_candidate",
    type_col: str = "page_label_type",
    value_col: str = "page_label_value",
    fingerprint_cols: tuple[str, str, str] = ("page_label_type", "page_label_cell_sharing", "page_label_wrapper"),
    allow_blank_pages: bool = True,
    blank_penalty: float = 0.20,        # mild penalty so blanks are possible but not preferred
    same_fp_bonus: float = 0.50,
    same_type_bonus: float = 0.20,
    inc_by_one_bonus: float = 2.00,     # strong preference for +1 steps
    equal_bonus: float = 0.20,          # sometimes repeats happen
    forward_gap_penalty: float = 0.40,  # penalty per missing increment (curr-prev-1)
    backward_forbid: float = 1e9,       # effectively -inf
    restart_bonus: float = 1.00,
    restart_prev_min: int = 10,         # if previous label is "high"
    restart_curr_max: int = 5,          # and next label is "small"
    restart_support_lookahead: int = 2, # require support in next N pages if fp does not change
    out_final_col: str = "page_label",
    out_block_type_col: str = "block_type",
    out_role_value: str = "page_label",
    out_series_id_col: str = "page_label_series_id",
) -> pd.DataFrame:
    """
    Picks a winner candidate per page using DP (Viterbi-style) so it can:
    - choose among multiple candidates on a violating page
    - leave a page blank if nothing fits
    - handle restarts / format changes (e.g., 1–58 main, blank, restart 2–15 appendix)

    Requirements (candidate rows already marked + scored):
      - df has candidate rows where token_col not null
      - df has score_col numeric (can be NaN for non-candidates)
      - df has page_col, token_col, type_col, value_col
      - fingerprint columns exist (default uses type/sharing/wrapper)

    Effects:
      - Writes out_final_col only on the chosen candidate row for each page
      - Sets out_block_type_col = out_role_value for chosen candidate rows
      - Writes out_series_id_col on chosen rows (segment id for restarts)
      - Leaves other rows untouched (except clearing existing out_final_col on non-chosen candidates)

    Notes:
      - This uses integer page ordering. If page_col isn’t numeric, we coerce.
      - If restart_requires_blank_page=True, a restart is only allowed if the previous page chosen state was blank.
    """
    out = df.copy()

    # --- Validate columns ---
    required = {page_col, token_col, type_col, value_col, score_col, *fingerprint_cols}
    missing = required - set(out.columns)
    if missing:
        raise ValueError(f"pick_pdf_page_label_winners_and_validate: missing columns: {sorted(missing)}")

    if out_final_col not in out.columns:
        out[out_final_col] = None
    if out_block_type_col not in out.columns:
        out[out_block_type_col] = None
    if out_series_id_col not in out.columns:
        out[out_series_id_col] = None

    # Clear any previously-set finals (we will re-assign cleanly)
    # Only clear where there is a candidate token to avoid nuking other uses of these cols.
    cand_any = out[token_col].notna() & (out[token_col].astype(str).str.len() > 0)
    out.loc[cand_any, out_final_col] = None
    # Don't clear block_type globally; only set it on winners.

    # --- Build per-page candidate lists ---
    pages_raw = pd.to_numeric(out[page_col], errors="coerce")
    if pages_raw.isna().all():
        raise ValueError(f"{page_col} could not be coerced to numeric pages")
    out["_page_int__"] = pages_raw.astype("Int64")

    # Candidate rows with scores
    cand_mask = out["_page_int__"].notna() & cand_any
    if not cand_mask.any():
        out = out.drop(columns=["_page_int__"], errors="ignore")
        return out

    # Page set: include pages that exist in df (not only candidate pages)
    all_pages = (
        out.loc[out["_page_int__"].notna(), "_page_int__"]
        .astype(int)
        .unique()
        .tolist()
    )
    all_pages.sort()
    page_min, page_max = all_pages[0], all_pages[-1]

    # We run DP across every page in [min..max] to allow blanks to bridge gaps
    page_axis = list(range(page_min, page_max + 1))

    # Helper: make fingerprint string
    def _fp_for_row(r: pd.Series) -> str:
        t = str(r[fingerprint_cols[0]]).lower()
        s = bool(r[fingerprint_cols[1]])
        w = str(r[fingerprint_cols[2]]).lower()
        return f"{t}|{int(s)}|{w}"

    # Store candidates per page as state objects
    @dataclass(frozen=True)
    class CandState:
        row_idx: Optional[int]  # None => blank state
        score: float
        token: Optional[str]
        typ: Optional[str]
        val: Optional[float]
        fp: Optional[str]

    candidates_by_page: Dict[int, List[CandState]] = {}

    # Precompute candidate states
    cand_df = out.loc[cand_mask].copy()
    cand_df["_score__"] = pd.to_numeric(cand_df[score_col], errors="coerce")
    cand_df["_val__"] = pd.to_numeric(cand_df[value_col], errors="coerce")
    cand_df["_typ__"] = cand_df[type_col].astype(str).str.lower()
    cand_df["_fp__"] = cand_df.apply(_fp_for_row, axis=1)

    # Group candidate states by page
    for p, g in cand_df.groupby("_page_int__", sort=False):
        p = int(p)
        states: List[CandState] = []
        # Sort candidates within a page by score descending (helps keep state lists small if needed)
        g2 = g.sort_values("_score__", ascending=False)
        for idx, row in g2.iterrows():
            sc = float(row["_score__"]) if pd.notna(row["_score__"]) else float("-inf")
            states.append(
                CandState(
                    row_idx=int(idx),
                    score=sc,
                    token=str(row[token_col]) if pd.notna(row[token_col]) else None,
                    typ=str(row["_typ__"]) if pd.notna(row["_typ__"]) else None,
                    val=float(row["_val__"]) if pd.notna(row["_val__"]) else None,
                    fp=str(row["_fp__"]) if pd.notna(row["_fp__"]) else None,
                )
            )

        # Add blank state
        if allow_blank_pages:
            states.append(
                CandState(
                    row_idx=None,
                    score=-float(blank_penalty),  # mild penalty
                    token=None,
                    typ=None,
                    val=None,
                    fp=None,
                )
            )

        candidates_by_page[p] = states

    # For pages with no candidates, only blank state (if allowed) else empty (unlabeled)
    for p in page_axis:
        if p not in candidates_by_page:
            candidates_by_page[p] = (
                [CandState(None, -float(blank_penalty), None, None, None, None)]
                if allow_blank_pages
                else []
            )

    # --- Restart support lookahead (does the new low value continue as a +1 series?) ---
    def _restart_has_support(curr: CandState, curr_page: int) -> bool:
        """
        If fingerprint did not change, we only accept restart when it "starts a series":
        i.e., within the next N pages, there exist candidates with the same fp and
        incrementing values (+1, +2, ...).
        """
        if restart_support_lookahead <= 0:
            return True
        if curr.row_idx is None or curr.val is None or curr.fp is None:
            return False

        base = curr.val
        fp = curr.fp

        for step in range(1, int(restart_support_lookahead) + 1):
            p = curr_page + step
            if p not in candidates_by_page:
                return False

            want = base + step
            ok = False
            for st in candidates_by_page[p]:
                if st.row_idx is None:
                    continue
                if st.fp == fp and st.val is not None and st.val == want:
                    ok = True
                    break

            if not ok:
                return False

        return True

    # --- Transition scoring ---
    def _is_restart(prev: CandState, curr: CandState, curr_page: int) -> bool:
        # restarts only considered for non-blank candidates with numeric values
        if prev.row_idx is None or curr.row_idx is None:
            return False
        if prev.val is None or curr.val is None:
            return False

        # must be a decrease
        if curr.val >= prev.val:
            return False

        # Valid restart if either:
        #  (A) fingerprint changes (format change, e.g. roman -> arabic, wrapper/share
        #      change) — strong evidence on its own, so no magnitude gate. A short
        #      roman preamble (i–iii) must be able to restart into arabic 1.
        #  (B) fingerprint stays the same, the drop looks like a restart
        #      (high -> small), and it is supported by a +1 continuation
        fp_changed = (prev.fp is not None and curr.fp is not None and prev.fp != curr.fp)
        if fp_changed:
            return True

        # heuristic: looks like a restart (high -> small)
        if not (curr.val <= restart_curr_max and prev.val >= restart_prev_min):
            return False

        return _restart_has_support(curr, curr_page)

    def _transition(prev: CandState, curr: CandState, curr_page: int) -> float:
        # blank transitions
        if prev.row_idx is None and curr.row_idx is None:
            return 0.0
        if prev.row_idx is None and curr.row_idx is not None:
            return 0.0
        if prev.row_idx is not None and curr.row_idx is None:
            return 0.0

        # both non-blank
        bonus = 0.0

        if prev.fp is not None and curr.fp is not None and prev.fp == curr.fp:
            bonus += same_fp_bonus

        if prev.typ is not None and curr.typ is not None and prev.typ == curr.typ:
            bonus += same_type_bonus

        # value monotonic / restart logic
        if prev.val is not None and curr.val is not None:
            if curr.val == prev.val + 1:
                bonus += inc_by_one_bonus
            elif curr.val == prev.val:
                bonus += equal_bonus
            elif curr.val > prev.val:
                gap = curr.val - prev.val - 1
                if gap > 0:
                    bonus -= forward_gap_penalty * gap
            else:
                # curr < prev => only allowed if restart
                if _is_restart(prev, curr, curr_page):
                    bonus += restart_bonus
                else:
                    return -backward_forbid  # effectively impossible

        return bonus

    # --- DP (Viterbi) ---
    # dp[p][i] = best score up to page p ending in candidate i (index into candidates_by_page[p])
    # back[p][i] = (prev_i)
    dp: Dict[int, List[float]] = {}
    back: Dict[int, List[int]] = {}

    first = page_axis[0]
    states0 = candidates_by_page[first]
    dp[first] = [s.score for s in states0]
    back[first] = [-1] * len(states0)

    for p in page_axis[1:]:
        prev_p = p - 1
        prev_states = candidates_by_page[prev_p]
        curr_states = candidates_by_page[p]

        prev_dp = dp[prev_p]
        curr_dp = [float("-inf")] * len(curr_states)
        curr_back = [-1] * len(curr_states)

        for j, curr_s in enumerate(curr_states):
            best_val = float("-inf")
            best_i = -1

            for i, prev_s in enumerate(prev_states):
                if prev_dp[i] == float("-inf") or prev_s.score == float("-inf") or curr_s.score == float("-inf"):
                    continue

                tr = _transition(prev_s, curr_s, p)
                if tr <= -backward_forbid:
                    continue

                cand_val = prev_dp[i] + tr + curr_s.score
                if cand_val > best_val:
                    best_val = cand_val
                    best_i = i

            curr_dp[j] = best_val
            curr_back[j] = best_i

        dp[p] = curr_dp
        back[p] = curr_back

    # --- Decode best path ---
    last = page_axis[-1]
    last_dp = dp[last]
    if not last_dp or max(last_dp) == float("-inf"):
        out = out.drop(columns=["_page_int__"], errors="ignore")
        return out

    end_j = int(np.nanargmax(np.array(last_dp, dtype=float)))
    chosen_state_idx_by_page: Dict[int, int] = {}
    chosen_state_idx_by_page[last] = end_j

    for p in range(last, first, -1):
        j = chosen_state_idx_by_page[p]
        i = back[p][j]
        if i < 0:
            # no predecessor; break chain
            break
        chosen_state_idx_by_page[p - 1] = i

    # --- Apply winners + build series ids ---
    # First, gather chosen states in order
    chosen_states: List[Tuple[int, CandState]] = []
    for p in page_axis:
        j = chosen_state_idx_by_page.get(p, None)
        if j is None:
            continue
        st = candidates_by_page[p][j]
        chosen_states.append((p, st))

    # series id increments on restart (curr.val < prev.val) among non-blank chosen.
    # Blank pages do NOT reset the comparison value: a decrease across a blank
    # bridge (e.g. ...62, blank, A-2) is still a new series. Resetting here would
    # merge the post-blank run into the previous series and QC would then clear
    # it as a monotonicity violation.
    series_id = 1
    prev_nonblank_val = None

    for p, st in chosen_states:
        if st.row_idx is None:
            continue

        # detect restart (value decrease relative to previous nonblank)
        if (
            prev_nonblank_val is not None
            and st.val is not None
            and st.val < prev_nonblank_val
        ):
            series_id += 1

        prev_nonblank_val = st.val

        # mark final label on the chosen row
        out.at[st.row_idx, out_final_col] = st.token
        out.at[st.row_idx, out_block_type_col] = out_role_value
        out.at[st.row_idx, out_series_id_col] = series_id

    out = out.drop(columns=["_page_int__"], errors="ignore")
    return out

# ================================================================================
# STEP 2: Post-DP series QC
# ================================================================================

def qc_pdf_page_label_series(
    df: pd.DataFrame,
    *,
    page_col: str = "page_number",
    label_col: str = "page_label",
    value_col: str = "page_label_value",
    type_col: str = "page_label_type",
    block_type_col: str = "block_type",
    series_id_col: str = "page_label_series_id",
    enforce_unit_step: bool = False,
    step_anomaly_tolerance_pct: float = 0.02,
    series_hole_tolerance_pct: float = 0.02,
) -> pd.DataFrame:
    """
    Post-DP quality control that exploits the fact PDF page numbers are always known.

    Scans winner labels in page order, grouped by series_id. Within each series two
    step rules are applied. The series is judged holistically: a series is allowed
    up to max(1, step_anomaly_tolerance_pct * series length) violating steps (a
    62-page series with one printed-label quirk, e.g. the document skips label 36,
    stays intact). Only when a series exceeds that tolerance is it distrusted, and
    then the first violating page — plus every page after it in that series — is
    cleared.

    Rule 1 — strict monotonicity (always on):
        Consecutive labeled pages must have strictly increasing values.
        Catches (a) the same label repeating on consecutive pages (e.g. "L" forever),
        and (b) a backward jump that slipped through a blank-page bridge in the DP.

    Rule 2 — unit-step (enforce_unit_step=True, off by default):
        For consecutive labeled pages at PDF pages P and P+k, the value difference
        must be exactly k.  This catches forward value gaps (e.g. value jumps from
        347 to 352 while only one page passed).  Disabled by default because some
        documents use non-unit page label increments (article numbers, etc.).

    Rule 3 — minimum consecutive run (always on):
        At least one series must contain a run of consecutive labeled PDF pages of
        length >= min(5, num_pages_in_doc).  A 3-page document only needs 3; a
        200-page document needs 5.  If this gate fails all labels are cleared.

    Rule 4 — label coverage (always on):
        At least 50% of the document's distinct page numbers must have a winner
        label.  If fewer pages are labeled the detection is too sparse to trust
        and all labels are cleared.

    Rule 5 — per-series plausibility (always on):
        Each series is judged on its own and cleared wholesale if implausible:
        - a 1-label series is never trusted (rejects a stray "19" or "c");
        - a 2-3 label series must be hole-free and read exactly N, N+1(, N+2)
          with start N <= run length ("1,2", "2,3", "1,2,3", "3,4,5" pass;
          "1,7" or "19,20" do not);
        - a longer series must start at a value <= its run length, unless it
          labels 100% of the document's pages (academic offprints: a 4-page
          paper labeled 952..955 is fine), and may contain at most
          series_hole_tolerance_pct unlabeled holes within its page span
          (rejects accidental sparse runs like "Exhibit 1, blank, Exhibit 2,
          blank, blank, Exhibit 3" where half the span has no label).

    Pages cleared by this function are also stripped of block_type, series_id,
    value, and type so that the subsequent spread step treats them as unlabeled.
    """
    out = df.copy()

    if label_col not in out.columns or series_id_col not in out.columns:
        return out

    # Work on winner rows only (at most one per page at this stage)
    winner_mask = out[label_col].notna() & out[series_id_col].notna()
    if not winner_mask.any():
        return out

    winners = out.loc[winner_mask, [page_col, label_col, value_col, series_id_col]].copy()
    winners["_p"] = pd.to_numeric(winners[page_col], errors="coerce")
    winners["_v"] = pd.to_numeric(winners[value_col], errors="coerce")
    winners["_sid"] = pd.to_numeric(winners[series_id_col], errors="coerce")
    winners = winners.dropna(subset=["_p", "_v", "_sid"]).sort_values(["_sid", "_p"])

    if winners.empty:
        return out

    # Rule 3: minimum consecutive run gate.
    # Count distinct pages in the full document (not just winner rows).
    num_pages = int(pd.to_numeric(out[page_col], errors="coerce").nunique())
    min_run = min(5, max(1, num_pages))

    # Find the longest run of consecutive PDF pages that all have a winner label.
    labeled_pages = sorted(winners["_p"].astype(int).unique().tolist())
    max_run = 1
    cur_run = 1
    for k in range(1, len(labeled_pages)):
        if labeled_pages[k] == labeled_pages[k - 1] + 1:
            cur_run += 1
            max_run = max(max_run, cur_run)
        else:
            cur_run = 1

    def _clear_all_labels(frame: pd.DataFrame) -> pd.DataFrame:
        for col in (label_col, value_col, type_col, series_id_col):
            if col in frame.columns:
                frame[col] = None
        if block_type_col in frame.columns:
            frame.loc[frame[block_type_col] == "page_label", block_type_col] = None
        return frame

    if max_run < min_run:
        return _clear_all_labels(out)

    # Rule 4: label coverage gate.
    # At least 50% of the document's pages must have a winner label.
    num_labeled_pages = len(labeled_pages)
    if num_pages > 0 and (num_labeled_pages / num_pages) < 0.50:
        return _clear_all_labels(out)

    pages_to_clear: set = set()

    for _, grp in winners.groupby("_sid"):
        pages = grp["_p"].astype(int).tolist()
        values = grp["_v"].tolist()
        n = len(pages)

        # Rule 5: per-series plausibility.
        if n == 1:
            pages_to_clear.update(pages)
            continue

        span = pages[-1] - pages[0] + 1
        holes = span - n
        v0 = values[0]

        if n <= 3:
            # Tiny series: hole-free, exactly consecutive, start no deeper than
            # the run length (1,2 / 2,3 / 1,2,3 / ... / 3,4,5).
            if holes > 0 or not (1 <= v0 <= n) or any(values[i] != v0 + i for i in range(n)):
                pages_to_clear.update(pages)
                continue
        else:
            allowed_holes = max(1, int(round(series_hole_tolerance_pct * span)))
            if holes > allowed_holes:
                pages_to_clear.update(pages)
                continue
            # Start value must be plausible for a fresh series, unless the
            # series labels the entire document (academic offprint case).
            covers_full_doc = n == num_pages
            if not covers_full_doc and not (1 <= v0 <= n):
                pages_to_clear.update(pages)
                continue

        violations: List[int] = []
        for i in range(1, len(pages)):
            p1, v1 = pages[i - 1], values[i - 1]
            p2, v2 = pages[i], values[i]

            # Rule 1: value must strictly increase
            if v2 <= v1:
                violations.append(i)
            # Rule 2 (optional): value increment must equal page increment
            elif enforce_unit_step and (v2 - v1) != (p2 - p1):
                violations.append(i)

        if not violations:
            continue

        # Holistic tolerance: a long, otherwise-consistent series may contain a
        # few printed-label anomalies; only distrust it beyond the tolerance.
        allowed = max(1, int(round(step_anomaly_tolerance_pct * len(pages))))
        if len(violations) <= allowed:
            continue

        pages_to_clear.update(pages[violations[0]:])

    if not pages_to_clear:
        return out

    page_int = pd.to_numeric(out[page_col], errors="coerce").fillna(-1).astype(int)
    bad_mask = page_int.isin(pages_to_clear)

    out.loc[bad_mask, label_col] = None
    for col in (value_col, type_col, series_id_col):
        if col in out.columns:
            out.loc[bad_mask, col] = None
    if block_type_col in out.columns:
        out.loc[bad_mask & (out[block_type_col] == "page_label"), block_type_col] = None

    return out


# ================================================================================
# STEP 3: Drop unreliable prefix labels
# ================================================================================

def drop_unreliable_prefix_labels(
    df: pd.DataFrame,
    *,
    page_col: str = "page_number",
    label_col: str = "page_label",
    value_col: str = "page_label_value",
    min_run_len: int = 3,             # require at least 4 consecutive +1 steps
    max_offset_abs: int = 20,         # optional: disallow absurd offsets like page 2 -> label 11 (offset +9)
    clear_block_type: bool = True,
    block_type_col: str = "block_type",
    block_type_value: str = "page_label",
) -> pd.DataFrame:
    """
    Finds the earliest contiguous +1 run of (page -> label_value) of length >= min_run_len,
    and clears any labels on pages before that run starts.

    Assumes df already has page_label set on at least one row per labeled page.
    Best used after you've spread page_label across the page OR on winner rows only.
    """
    out = df.copy()

    # page-level view: one label_value per page — read from winner rows only.
    # Non-winner candidate rows also carry page_label_value (set during candidate
    # marking) and would produce wrong first-non-null values per page if included.
    winner_rows = out[out[label_col].notna() & (out[label_col].astype(str).str.len() > 0)]
    page_vals = (
        winner_rows[[page_col, value_col]]
        .assign(_p=pd.to_numeric(winner_rows[page_col], errors="coerce"),
                _v=pd.to_numeric(winner_rows[value_col], errors="coerce"))
        .dropna(subset=["_p"])
        .groupby("_p")["_v"]
        .apply(lambda s: s.dropna().iloc[0] if not s.dropna().empty else np.nan)
        .reset_index()
        .rename(columns={"_p": "page", "_v": "val"})
    )

    page_vals = page_vals.dropna(subset=["val"]).sort_values("page")
    if page_vals.empty:
        return out

    pages = page_vals["page"].astype(int).to_numpy()
    vals = page_vals["val"].astype(int).to_numpy()

    # helper: compute per-page offset (val - page)
    offsets = vals - pages

    # Find earliest run where:
    # - pages are consecutive
    # - vals increase by +1
    # - offset is not insane (optional constraint)
    best_start_page = None

    i = 0
    n = len(pages)
    while i < n:
        # start a run at i
        run_len = 1
        j = i
        while j + 1 < n:
            if pages[j + 1] != pages[j] + 1:
                break
            if vals[j + 1] != vals[j] + 1:
                break
            run_len += 1
            j += 1

        if run_len >= min_run_len:
            # optional offset sanity: use median offset within run
            run_off = offsets[i : i + run_len]
            med_off = int(np.median(run_off))
            if abs(med_off) <= int(max_offset_abs):
                best_start_page = int(pages[i])
                break

        i = max(i + 1, j + 1)

    if best_start_page is None:
        return out  # no reliable run found; keep as-is

    # Clear labels before best_start_page
    mask_prefix_pages = pd.to_numeric(out[page_col], errors="coerce").fillna(-1).astype(int) < best_start_page
    out.loc[mask_prefix_pages, label_col] = None

    # Also clear role markers if you want
    if clear_block_type and block_type_col in out.columns:
        out.loc[mask_prefix_pages & (out[block_type_col] == block_type_value), block_type_col] = None

    return out

# ================================================================================
# STEP 3: Spread page label across page
# ================================================================================

def spread_winner_page_label_across_page(
    df: pd.DataFrame,
    *,
    page_col: str = "page_number",
    label_col: str = "page_label",
    type_col: str = "page_label_type",
    value_col: str = "page_label_value",
) -> pd.DataFrame:
    """
    Assumes the winner selection step set `page_label` on at most one row per page.
    Spreads that winner's:
      - page_label
      - page_label_type
      - page_label_value
    to all rows on the same page.

    Pages with no winner (no non-null page_label) remain blank for all three columns.
    """
    out = df.copy()

    # Build page -> winner mappings using only winner rows (where page_label is set)
    winners = out.loc[out[label_col].notna(), [page_col, label_col, type_col, value_col]]

    # If something ever violates the "one winner per page" assumption, fail loudly
    dup_pages = winners[page_col][winners.duplicated(subset=[page_col])]
    if not dup_pages.empty:
        raise ValueError(
            f"More than one winner row found for pages: {sorted(dup_pages.unique().tolist())}"
        )

    page_to_label = winners.set_index(page_col)[label_col]
    page_to_type = winners.set_index(page_col)[type_col]
    page_to_value = winners.set_index(page_col)[value_col]

    # Broadcast across all rows
    out[label_col] = out[page_col].map(page_to_label)
    out[type_col] = out[page_col].map(page_to_type)
    out[value_col] = out[page_col].map(page_to_value)

    return out


# ================================================================================
# public API
# ================================================================================

def _run_page_label_pipeline(
    df: pd.DataFrame,
    page_label_config: PageLabelConfig,
    *,
    artifact_only: bool,
    max_lines_top: int = _MAX_LINES_TOP,
    max_lines_bottom: int = _MAX_LINES_BOTTOM,
    max_cells_per_line: int = _MAX_CELLS_PER_LINE,
) -> pd.DataFrame:
    """Full candidate → score → winner → QC → spread pipeline for one pass."""
    out = mark_pdf_page_label_candidates(
        df, page_label_config, max_lines_top, max_lines_bottom, max_cells_per_line,
        artifact_only=artifact_only,
    )
    out = add_page_label_score(out)
    out = pick_pdf_page_label_winners_and_validate(out)
    out = qc_pdf_page_label_series(out, enforce_unit_step=True)
    out = drop_unreliable_prefix_labels(out)
    out = spread_winner_page_label_across_page(out)
    return out


def detect_pdf_page_labels(
    df: pd.DataFrame,
    page_label_config: PageLabelConfig,
    max_lines_top: int = _MAX_LINES_TOP,
    max_lines_bottom: int = _MAX_LINES_BOTTOM,
    max_cells_per_line: int = _MAX_CELLS_PER_LINE,
) -> pd.DataFrame:
    # Pre-pass: look for a valid series among Artifact-tagged cells only,
    # bypassing the positional config. Tagged PDFs mark page furniture as
    # Artifacts, so when a page-number series lives there it survives the same
    # QC gates without the top/bottom-line + max-cells heuristics.
    artifact_out = _run_page_label_pipeline(df, page_label_config, artifact_only=True)
    if artifact_out["page_label"].notna().any():
        return artifact_out

    # Fallback: positional heuristic over all cells.
    return _run_page_label_pipeline(
        df,
        page_label_config,
        artifact_only=False,
        max_lines_top=max_lines_top,
        max_lines_bottom=max_lines_bottom,
        max_cells_per_line=max_cells_per_line,
    )
