"""
shape_marry.py

Marry shapes with words / cells. Runs *after* group_multiline_cells, taking the
words, cells and shapes tables and returning all three annotated.

This is a growing module. Steps:
  1. underlines / strikethroughs  (implemented)
  2. grid_cell_id                 (implemented)
  3. shape_id_hr_above / _below   (implemented)

STEP 2 — grid-cell containment
    Each word is tagged with the reconstructed table grid cell (from
    df_grid_cells) whose bbox fully contains it: grid_cell_id + table_grid_id on
    the words table. Grid cells tile without overlap, so a word matches at most
    one; words outside every cell (body text, a caption above the grid) get NA.

STEP 3 — cell horizontal rules
    Each cell is tagged with the horizontal rule bounding it above (nearest to
    y_top) and below (nearest to y_bottom): shape_id_hr_above / shape_id_hr_below
    on the cells table. Candidates are all horizontal line shapes (grids
    included), minus the underline/strikethrough shapes flagged in step 1. A rule
    must overlap the cell horizontally and sit within hr_max_dist outside the
    edge, and may cross hr_cross_tol into the cell (tight tables). NA when none.

------------------------------------------------------------------------------
STEP 1 — underlines & strikethroughs
------------------------------------------------------------------------------
A *true* underline is the ``<u>`` of docx/html: a short horizontal rule that
hugs the bottom of a run of text. It is NOT a table rule that happens to pass
below the text (those are shape_role == "table_grid" and are masked out here) and
NOT a cell/row border.

Why per-word, not per-cell
    A single logical (possibly multi-visual-line) table cell can have some lines
    underlined and others not. Deciding at the cell level would smear one shape
    across several lines and make the mapping ambiguous. So underline detection
    is done at the *word* level and then aggregated up to a char-weighted
    ``underlined_ratio`` on the cell.

Geometry of a match (thresholds live in UnderlineConfig)
    * candidate shapes: horizontal shape_type == "line", shape_role !=
      "table_grid" (table grid rules can't be underlines).
    * horizontal:  the line must cover at least ``min_h_overlap_ratio`` of the
      word's width (an underline spans the glyphs it decorates).
    * length:      the line is no longer than ``max_len_ratio`` (default 1.2x)
      the width of the word's *cell* — a longer rule is a table/section rule.
    * vertical:    the line's y sits within ``max_below`` pt below the word's
      y_bottom and up to ``max_above`` pt above it. Real underlines usually sit
      *slightly* above the glyph box (box bottom 168 -> line at 167), so a small
      negative offset is allowed; a large negative offset lands near the text
      center and is a strikethrough instead.
    * strikethrough: a line within ``strike_center_tol`` of the word's y_center
      (and closer to the center than to the bottom) is a strikethrough, not an
      underline.

Outputs
    words:  is_underline, shape_id_underline, is_strikethrough,
            shape_id_strikethrough
    cells:  underlined_ratio, is_underline, shape_id_underline (list — a
            multi-line cell can carry several),  strikethrough_ratio,
            is_strikethrough, shape_id_strikethrough (list),
            shape_id_hr_above, shape_id_hr_below
    shapes: is_underline, is_strikethrough  (True on a shape matched to >=1 word)

Everything is vectorised: candidate shapes are filtered down first (usually a
handful per page), then a single page-keyed many-to-many merge produces the
word x line pairs that boolean masks classify — no Python-level loops over words.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass

import numpy as np
import pandas as pd

from docslicer.pdf._utils.shape_relationships import add_grid_cell_relationships


# ================================================================================
# CONFIG
# ================================================================================

@dataclass(frozen=True)
class UnderlineConfig:
    # A line longer than this multiple of the word's cell width is a table/section
    # rule, not an underline.
    max_len_ratio: float = 1.05

    # The line must cover at least this fraction of the word's width.
    min_h_overlap_ratio: float = 0.5

    # How far *below* the word's y_bottom the line may sit (pt). Larger y = lower.
    max_below: float = 5.0
    # How far *above* the word's y_bottom the line may sit (pt). The word box
    # bottom includes descender space, so a real underline sits a few pt *above*
    # it (e.g. box bottom 435.8 -> rule at 433.6). Kept well clear of y_center so
    # a mid-glyph rule still reads as strikethrough, not underline.
    max_above: float = 4.0
    # A line within this of the word's y_center (and nearer the center than the
    # bottom) is a strikethrough, not an underline.
    strike_center_tol: float = 2.0

    # --- cell aggregation ---
    # A cell is flagged is_underline when this fraction (char-weighted) of its
    # text is underlined. Same threshold reused for strikethrough.
    cell_flag_min_ratio: float = 0.75

    # --- grid-cell containment (step 2) ---
    # Slack (pt) when testing "word fully inside a grid cell": the word may spill
    # this far past the cell edges and still count as contained (absorbs glyph
    # box / grid-line rounding).
    grid_contain_tol: float = 1.0

    # --- cell horizontal rules above/below (step 3) ---
    # How far a rule may cross *into* the cell and still count as its top/bottom
    # rule (a tight table draws the rule a hair inside the text box). There is no
    # outward cap: a cell casts an unbounded vertical ray up and down within its
    # x-band and catches the nearest rule each way, however much text lies between
    # (so every cell in a rule-bounded block resolves to the same above/below).
    hr_cross_tol: float = 2.0


CONFIG = UnderlineConfig()


# ================================================================================
# HELPERS
# ================================================================================

def _parse_id_list(v) -> list:
    """word_ids as a plain list (handles an actual list or its CSV repr string)."""
    if isinstance(v, (list, tuple, np.ndarray, pd.Series)):
        return list(v)
    if isinstance(v, str):
        s = v.strip()
        if not s or s == "[]":
            return []
        try:
            parsed = ast.literal_eval(s)
            return list(parsed) if hasattr(parsed, "__iter__") else [parsed]
        except (ValueError, SyntaxError):
            return []
    return []


def _candidate_lines(df_shapes: pd.DataFrame) -> pd.DataFrame:
    """
    Horizontal line shapes (shape_type == "line"), excluding table grid rules,
    eligible to be an underline. Returns a slim frame keyed for the merge:
    page_number, shape_id, line_x_left, line_x_right, line_len, line_y.
    """
    df = df_shapes
    mask = np.ones(len(df), dtype=bool)
    if "shape_role" in df.columns:
        mask &= (df["shape_role"].astype("string") != "table_grid").to_numpy()
    if "shape_orientation" in df.columns:
        mask &= (df["shape_orientation"].astype("string") == "horizontal").to_numpy()
    if "shape_type" in df.columns:
        mask &= (df["shape_type"].astype("string") == "line").to_numpy()

    y_top = df["y_top"].to_numpy(float)
    y_bot = df["y_bottom"].to_numpy(float)

    sub = df.loc[mask, ["page_number", "shape_id", "x_left", "x_right"]].copy()
    sub["line_x_left"] = sub["x_left"].to_numpy(float)
    sub["line_x_right"] = sub["x_right"].to_numpy(float)
    sub["line_len"] = sub["line_x_right"] - sub["line_x_left"]
    sub["line_y"] = (y_top[mask] + y_bot[mask]) / 2.0
    return sub[["page_number", "shape_id", "line_x_left", "line_x_right",
                "line_len", "line_y"]]


# ================================================================================
# WORD-LEVEL DETECTION
# ================================================================================

def _detect_word_decorations(
    df_words: pd.DataFrame,
    df_shapes: pd.DataFrame,
    config: UnderlineConfig,
) -> pd.DataFrame:
    """
    Classify each word as underlined / struck-through by matching it against the
    candidate horizontal lines. Returns a frame indexed like df_words with columns
    is_underline, shape_id_underline, is_strikethrough, shape_id_strikethrough.
    """
    n = len(df_words)
    empty = pd.DataFrame(
        {
            "is_underline": np.zeros(n, dtype=bool),
            "shape_id_underline": pd.array([pd.NA] * n, dtype="Int64"),
            "is_strikethrough": np.zeros(n, dtype=bool),
            "shape_id_strikethrough": pd.array([pd.NA] * n, dtype="Int64"),
        },
        index=df_words.index,
    )

    lines = _candidate_lines(df_shapes)
    if lines.empty:
        return empty

    # Word geometry + its cell's width (from words grouped by cell_id — robust to
    # the cell_id reindex that group_multiline_cells applies only to df_cells).
    w = pd.DataFrame({
        "_row": np.arange(n),
        "page_number": df_words["page_number"].to_numpy(),
        "word_id": df_words["word_id"].to_numpy(),
        "w_x_left": df_words["x_left"].to_numpy(float),
        "w_x_right": df_words["x_right"].to_numpy(float),
        "w_y_bottom": df_words["y_bottom"].to_numpy(float),
    })
    w["w_y_center"] = (df_words["y_top"].to_numpy(float) + w["w_y_bottom"].to_numpy()) / 2.0
    w["w_width"] = w["w_x_right"] - w["w_x_left"]

    cell_key = df_words["cell_id"].to_numpy()
    grp = pd.Series(w["w_x_right"].to_numpy(), index=cell_key).groupby(level=0)
    cell_left = pd.Series(w["w_x_left"].to_numpy(), index=cell_key).groupby(level=0).transform("min")
    cell_right = grp.transform("max")
    w["cell_width"] = cell_right.to_numpy() - cell_left.to_numpy()

    # One page-keyed many-to-many merge (candidate lines are few per page).
    pairs = w.merge(lines, on="page_number", how="inner")
    if pairs.empty:
        return empty

    # --- geometry masks (vectorised) ---
    overlap = (np.minimum(pairs["w_x_right"], pairs["line_x_right"])
               - np.maximum(pairs["w_x_left"], pairs["line_x_left"]))
    h_ok = overlap >= config.min_h_overlap_ratio * pairs["w_width"].clip(lower=1e-6)

    len_ok = pairs["line_len"] <= config.max_len_ratio * pairs["cell_width"].clip(lower=1e-6)

    dy_bottom = pairs["line_y"] - pairs["w_y_bottom"]          # +below, -above
    d_center = (pairs["line_y"] - pairs["w_y_center"]).abs()

    near_bottom = (dy_bottom <= config.max_below) & (dy_bottom >= -config.max_above)
    near_center = d_center <= config.strike_center_tol

    base = h_ok & len_ok
    # Strikethrough wins when the line is nearer the center than the bottom.
    is_strike = base & near_center & (d_center <= dy_bottom.abs())
    is_under = base & near_bottom & ~is_strike

    pairs["_dy_bottom_abs"] = dy_bottom.abs()
    pairs["_d_center"] = d_center

    out = empty.copy()

    # For each word keep the nearest matching line (min distance to the relevant
    # edge). drop_duplicates on a sorted frame is a vectorised argmin-per-group.
    under = pairs[is_under].sort_values("_dy_bottom_abs", kind="mergesort")
    under = under.drop_duplicates("_row", keep="first")
    if not under.empty:
        rows = under["_row"].to_numpy()
        out.iloc[rows, out.columns.get_loc("is_underline")] = True
        out.iloc[rows, out.columns.get_loc("shape_id_underline")] = \
            under["shape_id"].to_numpy()

    strike = pairs[is_strike].sort_values("_d_center", kind="mergesort")
    strike = strike.drop_duplicates("_row", keep="first")
    if not strike.empty:
        rows = strike["_row"].to_numpy()
        out.iloc[rows, out.columns.get_loc("is_strikethrough")] = True
        out.iloc[rows, out.columns.get_loc("shape_id_strikethrough")] = \
            strike["shape_id"].to_numpy()

    return out


# ================================================================================
# CELL-LEVEL AGGREGATION
# ================================================================================

def _aggregate_to_cells(
    df_cells: pd.DataFrame,
    df_words: pd.DataFrame,
    config: UnderlineConfig,
) -> pd.DataFrame:
    """
    Roll the per-word decoration flags up onto cells via each cell's word_ids
    (robust to merges: a merged cell's word_ids already lists all its words).

    underlined_ratio / strikethrough_ratio are char-weighted over the cell's
    words; the shape_id_* columns collect the distinct matched shapes as a list.
    """
    n = len(df_cells)
    ratio_u = np.zeros(n)
    ratio_s = np.zeros(n)
    shapes_u: list[list] = [[] for _ in range(n)]
    shapes_s: list[list] = [[] for _ in range(n)]

    if "word_ids" in df_cells.columns:
        char = (pd.to_numeric(df_words.get("char_count"), errors="coerce")
                .fillna(0).to_numpy(float)) if "char_count" in df_words.columns \
            else np.ones(len(df_words))
        wid = df_words["word_id"].to_numpy()
        w_lookup = pd.DataFrame({
            "char": char,
            "u": df_words["is_underline"].to_numpy(bool),
            "s": df_words["is_strikethrough"].to_numpy(bool),
            "su": df_words["shape_id_underline"].to_numpy(object),
            "ss": df_words["shape_id_strikethrough"].to_numpy(object),
        }, index=wid)

        # Explode (cell_row, word_id) once, then a single groupby does all cells.
        cell_word = [
            (i, w) for i, ids in enumerate(df_cells["word_ids"].map(_parse_id_list))
            for w in ids
        ]
        if cell_word:
            ex = pd.DataFrame(cell_word, columns=["_cell", "word_id"])
            ex = ex.join(w_lookup, on="word_id")
            ex["char"] = ex["char"].fillna(0.0)
            ex["u"] = ex["u"].fillna(False)
            ex["s"] = ex["s"].fillna(False)

            g = ex.groupby("_cell", sort=False)
            tot = g["char"].sum()
            uc = ex.assign(_uc=ex["char"] * ex["u"]).groupby("_cell")["_uc"].sum()
            sc = ex.assign(_sc=ex["char"] * ex["s"]).groupby("_cell")["_sc"].sum()
            denom = tot.replace(0, np.nan)
            ru = (uc / denom).fillna(0.0)
            rs = (sc / denom).fillna(0.0)
            ratio_u[ru.index.to_numpy()] = ru.to_numpy()
            ratio_s[rs.index.to_numpy()] = rs.to_numpy()

            for cell_i, sub in ex.groupby("_cell", sort=False):
                su = sorted({int(v) for v in sub["su"].dropna().tolist()})
                ss = sorted({int(v) for v in sub["ss"].dropna().tolist()})
                shapes_u[cell_i] = su
                shapes_s[cell_i] = ss

    out = df_cells.copy()
    out["underlined_ratio"] = ratio_u
    out["is_underline"] = ratio_u >= config.cell_flag_min_ratio
    out["shape_id_underline"] = shapes_u
    out["strikethrough_ratio"] = ratio_s
    out["is_strikethrough"] = ratio_s >= config.cell_flag_min_ratio
    out["shape_id_strikethrough"] = shapes_s
    return out


# ================================================================================
# STEP 3 — CELL HORIZONTAL RULES (above / below)
# ================================================================================

def _assign_cell_hrules(
    df_cells: pd.DataFrame,
    df_shapes: pd.DataFrame,
    config: UnderlineConfig,
) -> pd.DataFrame:
    """
    For each cell, find the horizontal rule bounding it above (nearest to y_top)
    and below (nearest to y_bottom). Returns a frame indexed like df_cells with
    columns shape_id_hr_above, shape_id_hr_below (Int64, NA when none is near).

    Candidate rules are all horizontal line shapes (grids included), minus the
    ones step 1 tagged as underline / strikethrough. The cell casts an unbounded
    vertical ray up and down within its x-band: a rule must overlap the cell
    horizontally, and the nearest one above (y_top) and below (y_bottom) wins,
    however far — cells are blind to intervening text. A rule may cross
    config.hr_cross_tol *into* the cell (tight tables draw it a hair inside).
    """
    n = len(df_cells)
    empty = pd.DataFrame(
        {
            "shape_id_hr_above": pd.array([pd.NA] * n, dtype="Int64"),
            "shape_id_hr_below": pd.array([pd.NA] * n, dtype="Int64"),
        },
        index=df_cells.index,
    )
    need = {"page_number", "shape_id", "x_left", "x_right", "y_top", "y_bottom"}
    if df_shapes is None or df_shapes.empty or not need.issubset(df_shapes.columns):
        return empty

    df = df_shapes
    mask = np.ones(len(df), dtype=bool)
    if "shape_orientation" in df.columns:
        mask &= (df["shape_orientation"].astype("string") == "horizontal").to_numpy()
    if "shape_type" in df.columns:
        mask &= (df["shape_type"].astype("string") == "line").to_numpy()
    if "is_underline" in df.columns:
        mask &= ~df["is_underline"].fillna(False).to_numpy(bool)
    if "is_strikethrough" in df.columns:
        mask &= ~df["is_strikethrough"].fillna(False).to_numpy(bool)

    if not mask.any():
        return empty

    lines = pd.DataFrame({
        "page_number": df.loc[mask, "page_number"].to_numpy(),
        "shape_id":    df.loc[mask, "shape_id"].to_numpy(),
        "line_x_left":  df.loc[mask, "x_left"].to_numpy(float),
        "line_x_right": df.loc[mask, "x_right"].to_numpy(float),
        "line_y": (df.loc[mask, "y_top"].to_numpy(float)
                   + df.loc[mask, "y_bottom"].to_numpy(float)) / 2.0,
    })

    cells = pd.DataFrame({
        "_row": np.arange(n),
        "page_number": df_cells["page_number"].to_numpy(),
        "c_x_left":   df_cells["x_left"].to_numpy(float),
        "c_x_right":  df_cells["x_right"].to_numpy(float),
        "c_y_top":    df_cells["y_top"].to_numpy(float),
        "c_y_bottom": df_cells["y_bottom"].to_numpy(float),
    })

    pairs = cells.merge(lines, on="page_number", how="inner")
    if pairs.empty:
        return empty

    # Must overlap the cell horizontally (a table row rule spans it; a narrow
    # under-number rule covers just that column — both overlap).
    h_ok = ((pairs["line_x_left"] <= pairs["c_x_right"])
            & (pairs["line_x_right"] >= pairs["c_x_left"])).to_numpy()

    tol = config.hr_cross_tol
    dy_top = (pairs["line_y"] - pairs["c_y_top"]).to_numpy()      # +into cell, -above
    dy_bot = (pairs["line_y"] - pairs["c_y_bottom"]).to_numpy()   # +below, -into cell

    # Unbounded outward: a rule at/above y_top (up to hr_cross_tol into the cell)
    # is an "above" candidate; at/below y_bottom likewise for "below". Nearest wins.
    above_ok = h_ok & (dy_top <= tol)
    below_ok = h_ok & (dy_bot >= -tol)

    out = empty.copy()

    if above_ok.any():
        a = pairs.loc[above_ok].assign(_d=np.abs(dy_top[above_ok]))
        a = a.sort_values("_d", kind="mergesort").drop_duplicates("_row", keep="first")
        out.iloc[a["_row"].to_numpy(), out.columns.get_loc("shape_id_hr_above")] = \
            a["shape_id"].to_numpy()
    if below_ok.any():
        b = pairs.loc[below_ok].assign(_d=np.abs(dy_bot[below_ok]))
        b = b.sort_values("_d", kind="mergesort").drop_duplicates("_row", keep="first")
        out.iloc[b["_row"].to_numpy(), out.columns.get_loc("shape_id_hr_below")] = \
            b["shape_id"].to_numpy()

    return out


# ================================================================================
# ENTRY POINT
# ================================================================================

def marry_shapes(
    df_words: pd.DataFrame,
    df_cells: pd.DataFrame,
    df_shapes: pd.DataFrame,
    df_grid_cells: pd.DataFrame = None,
    config: UnderlineConfig = CONFIG,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Annotate words / cells / shapes with shape relationships:
      - underline & strikethrough (needs df_shapes)
      - grid_cell_id / table_grid_id per word (needs df_grid_cells)

    Returns (df_words, df_cells, df_shapes), each a copy with the new columns
    added. No-ops gracefully (adds default columns) when an input is empty or
    missing the geometry it needs.
    """
    need_words = {"word_id", "cell_id", "x_left", "x_right", "y_top", "y_bottom"}
    need_shapes = {"page_number", "shape_id", "x_left", "x_right", "y_top", "y_bottom"}

    df_words = df_words.copy() if df_words is not None else df_words
    df_cells = df_cells.copy() if df_cells is not None else df_cells
    df_shapes = df_shapes.copy() if df_shapes is not None else df_shapes

    words_ok = (df_words is not None and not df_words.empty
                and need_words.issubset(df_words.columns))
    shapes_ok = (df_shapes is not None and not df_shapes.empty
                 and need_shapes.issubset(df_shapes.columns))

    if words_ok and shapes_ok:
        deco = _detect_word_decorations(df_words, df_shapes, config)
    else:
        n = len(df_words) if df_words is not None else 0
        deco = pd.DataFrame({
            "is_underline": np.zeros(n, dtype=bool),
            "shape_id_underline": pd.array([pd.NA] * n, dtype="Int64"),
            "is_strikethrough": np.zeros(n, dtype=bool),
            "shape_id_strikethrough": pd.array([pd.NA] * n, dtype="Int64"),
        }, index=df_words.index if df_words is not None else None)

    if df_words is not None:
        for col in deco.columns:
            df_words[col] = deco[col]        # index-aligned; keeps Int64 dtype

    # Words: tag the grid cell that fully contains each word (step 2).
    if df_words is not None and not df_words.empty:
        df_words = add_grid_cell_relationships(
            df_words, df_grid_cells, contain_tol=config.grid_contain_tol
        )

    # Shapes: flag lines matched to at least one word.
    if df_shapes is not None:
        matched_u = set(pd.Series(deco["shape_id_underline"]).dropna().astype(int).tolist())
        matched_s = set(pd.Series(deco["shape_id_strikethrough"]).dropna().astype(int).tolist())
        sid = df_shapes["shape_id"].to_numpy() if "shape_id" in df_shapes.columns else np.array([])
        df_shapes["is_underline"] = np.isin(sid, list(matched_u)) if len(sid) else False
        df_shapes["is_strikethrough"] = np.isin(sid, list(matched_s)) if len(sid) else False

    # Cells: char-weighted underline/strikethrough roll-up via word_ids, then the
    # bounding horizontal rules above/below (step 3 — after shapes are flagged, so
    # underline/strikethrough rules are excluded from the rule search).
    if df_cells is not None and not df_cells.empty:
        df_cells = _aggregate_to_cells(df_cells, df_words, config)
        hr = _assign_cell_hrules(df_cells, df_shapes, config)
        for col in hr.columns:
            df_cells[col] = hr[col]          # index-aligned; keeps Int64 dtype

    return df_words, df_cells, df_shapes
