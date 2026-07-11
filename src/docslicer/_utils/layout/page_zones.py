"""
page_zones.py

Step 0 of the v3 gutter detector: visual y-lines and header/footer bands.

Runs BEFORE obstacle collection / whitespace-rect enumeration, on words only:

  1. y_line_id — group words into *visual* lines (pure y-proximity, spanning
     the full page width across columns) by running line_merger.assign_line_id
     on a (page, y_top, x_left)-sorted view.  Its anchor semantics — dy is
     measured against the line's FIRST word, not the previous one — prevent
     chained drift on 2-column layouts whose baselines overlap slightly.
     Deliberately NOT the pipeline's reading-order line_id: one visual line
     across two columns later becomes two reading lines, so the column is
     named y_line_id and never leaves this stage's consumers.

  2. header / footer bands — a running header/footer is separated from the
     body by a full-page-width horizontal whitespace band.  Because a band is
     a gap in the y-projection of page text, detecting it is a 1-D scan over
     the visual-line records, not a 2-D rectangle search.  The scan is
     gap-first: walk the lines down from the page edge and stop at the FIRST
     whitespace gap of at least min_band_gap; the zone is whatever sits above
     it — never a fixed edge-fraction slice, which would pull early body
     lines into the zone on pages whose body starts high.  See _detect_band
     for the disqualifier list (line cap, zone-depth cap, min_body_lines),
     all disqualifiers rather than errors.  Words only — images are ignored
     (their PDF bboxes are often wildly off-page, and zone tagging is about
     text; a figure near a band neither blocks nor fakes it).

    Zone-tagged words are meant to be excluded from gutter detection's body
    bound and from reading-order zone sorting (header first, footer last) —
    otherwise a left-aligned page label joins column 1 and a right-aligned
    running title sinks below it in reading order.

Public API:
    df_words                 = assign_y_line_ids(df_words)
    df_words, df_bands       = assign_page_zones(df_words, config)

assign_y_line_ids adds:
    y_line_id   Int64; NA for vertical (TTB/BTT) text, which never joins a
                visual line (a rotated margin caption would fuse lines across
                half the page height).
assign_page_zones adds to df_words:
    page_zone   'header' / 'body' / 'footer' (vertical text is always 'body')
df_bands columns (one row per detected band = the whitespace gap itself):
    page_number, band_role ('header'/'footer'),
    band_y_top, band_y_bottom, band_height, n_zone_lines

Coordinate convention (matches the rest of the pipeline):
    y increases downward; y_top < y_bottom.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .line_merger import assign_line_id

# =======================================================================================================================
# CONFIG
# =======================================================================================================================


@dataclass(frozen=True)
class PageZoneConfig:
    header_zone_frac: float = 0.08  # top page fraction a header line must lie within
    footer_zone_frac: float = 0.08  # bottom page fraction a footer line must lie within
    min_band_gap: float = 10.0      # pt - min whitespace between zone and body
    max_header_lines: int = 2       # more candidate lines than this = no header
    max_footer_lines: int = 1       # more candidate lines than this = no footer
    min_body_lines: int = 3         # a band needs a real body on its other side


_BBOX_COLS = ["x_left", "y_top", "x_right", "y_bottom"]
_BAND_COLS = [
    "page_number", "band_role",
    "band_y_top", "band_y_bottom", "band_height", "n_zone_lines",
]


# =======================================================================================================================
# Visual y-lines
# =======================================================================================================================

def assign_y_line_ids(df_words: pd.DataFrame) -> pd.DataFrame:
    """
    Add y_line_id: visual-line ids from pure y-proximity, full page width.

    Runs line_merger.assign_line_id on a (page_number, y_top, x_left)-sorted
    view so words at the same height share a line regardless of which column
    they belong to, then maps the ids back onto the caller's row order under
    the name y_line_id (see module docstring for why it must not be called
    line_id).  Vertical (TTB/BTT) words get NA.
    """
    out = df_words.copy() if df_words is not None else pd.DataFrame()
    out["y_line_id"] = pd.Series(pd.NA, index=out.index, dtype="Int64")
    if out.empty:
        return out

    missing = {"page_number", *_BBOX_COLS} - set(out.columns)
    if missing:
        raise ValueError(f"df_words missing required columns: {sorted(missing)}")

    horiz = out
    if "text_orientation" in out.columns:
        orient = out["text_orientation"].astype(str).str.upper().str.strip()
        horiz = out[~orient.isin(["TTB", "BTT"])]
    if horiz.empty:
        return out

    s = horiz.sort_values(["page_number", "y_top", "x_left"], kind="mergesort")
    # table_row_id / block_type must not leak into visual-line decisions:
    # assign_line_id honors them, but step 0 runs before either is assigned
    # and a caller-supplied block_type ("image") would fragment real lines.
    s = assign_line_id(s[["page_number", "x_left", "y_top", "y_bottom"]])
    out.loc[s.index, "y_line_id"] = s["line_id"].to_numpy()
    return out


# =======================================================================================================================
# Header / footer bands
# =======================================================================================================================

def _page_line_records(page_words: pd.DataFrame) -> pd.DataFrame:
    """One record per visual line on the page, sorted top-down."""
    return (
        page_words.groupby("y_line_id")
        .agg(y_top=("y_top", "min"), y_bottom=("y_bottom", "max"))
        .reset_index()
        .sort_values("y_top")
        .reset_index(drop=True)
    )


def _detect_band(
    lines: pd.DataFrame,
    page_h: float,
    role: str,
    config: PageZoneConfig,
) -> tuple[list, dict | None]:
    """
    Try to find one header or footer band on one page's line records.

    Gap-first top-down scan (a footer is the same scan in flipped
    coordinates): walk the lines downward from the page edge, growing the
    zone's y-envelope, and stop at the FIRST whitespace gap of at least
    min_band_gap.  The zone is whatever sits above that gap — never defined
    by a fixed edge fraction, because a page whose body starts high would
    pull its first body lines into an edge-fraction zone and the "gap" would
    then be measured between two body lines.  The edge fraction only caps
    how deep the zone may reach.

    Disqualifiers (checked at the gap; any one kills the band):
      - more than max_header/footer_lines lines above the gap (a title
        block is body, not a header);
      - zone envelope deeper than header/footer_zone_frac of the page;
      - fewer than min_body_lines lines below the gap.

    Returns (zone_line_ids, band_record | None).
    """
    yt = lines["y_top"].to_numpy(dtype=np.float64)
    yb = lines["y_bottom"].to_numpy(dtype=np.float64)

    if role == "footer":  # flip so the scan always walks top-down
        yt, yb = page_h - yb, page_h - yt
        zone_limit = config.footer_zone_frac * page_h
        max_lines = config.max_footer_lines
    else:
        zone_limit = config.header_zone_frac * page_h
        max_lines = config.max_header_lines

    order = np.argsort(yt, kind="stable")
    n = len(order)

    env_bottom = -np.inf
    for k in range(min(max_lines, n - 1)):
        env_bottom = max(env_bottom, float(yb[order[k]]))
        if env_bottom > zone_limit:
            return [], None
        gap = float(yt[order[k + 1]]) - env_bottom
        if gap < config.min_band_gap:
            continue
        # First qualifying gap decides — a deeper gap would swallow body.
        if n - (k + 1) < config.min_body_lines:
            return [], None
        ids = lines.loc[lines.index[order[: k + 1]], "y_line_id"].tolist()
        gap_top, gap_bottom = env_bottom, float(yt[order[k + 1]])
        if role == "footer":  # flip the band back
            gap_top, gap_bottom = page_h - gap_bottom, page_h - gap_top
        band = {
            "band_role": role,
            "band_y_top": gap_top,
            "band_y_bottom": gap_bottom,
            "band_height": gap_bottom - gap_top,
            "n_zone_lines": k + 1,
        }
        return ids, band
    return [], None


def assign_page_zones(
    df_words: pd.DataFrame,
    config: PageZoneConfig = PageZoneConfig(),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Tag every word with page_zone ('header' / 'body' / 'footer') and return
    the detected whitespace bands.

    Assigns y_line_id first when absent.  Page height comes from a
    page_height column when present, else falls back to the page's content
    extent (which understates the true height, making edge-zone tests
    conservative — a band can only be missed, never invented, that way).

    Returns (df_words, df_bands); see module docstring for columns.
    """
    out = df_words if (df_words is not None and "y_line_id" in df_words.columns) else (
        assign_y_line_ids(df_words)
    )
    out = out.copy()
    out["page_zone"] = "body"
    if out.empty:
        return out, pd.DataFrame(columns=_BAND_COLS)

    horiz = out[out["y_line_id"].notna()]
    has_page_h = "page_height" in out.columns

    band_records: list = []
    for page_number, page_words in horiz.groupby("page_number", sort=True):
        if has_page_h:
            page_h = float(page_words["page_height"].iloc[0])
        else:
            page_h = float(page_words["y_bottom"].max())
        if page_h <= 0:
            continue

        lines = _page_line_records(page_words)

        for role in ("header", "footer"):
            ids, band = _detect_band(lines, page_h, role, config)
            if band is None:
                continue
            band["page_number"] = page_number
            band_records.append(band)
            mask = (out["page_number"] == page_number) & out["y_line_id"].isin(ids)
            out.loc[mask, "page_zone"] = role

    df_bands = pd.DataFrame(band_records, columns=_BAND_COLS)
    return out, df_bands
