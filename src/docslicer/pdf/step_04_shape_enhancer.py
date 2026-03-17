# step_04_shape_enhancer.py

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Literal, Optional, Union, Tuple, List, Iterable

import numpy as np
import pandas as pd

# =============================
# Config
# =============================

# Merge Shapes
_GAP_TOL_PX = 1.5           # max gap between horizontal segments to still merge
_CHAIN_TOL_PX = 1.5        # vertical tolerance for merging horizontal lines

# Rect to Line Conversion
LINE_HEIGHT_MAX_PX = 3          # max height to treat rect as "line"

# Table Detection from raw Lines
TABLE_MIN_ROWS = 2
TABLE_MIN_COLS = 2

# Separator Line & Background Rect Detection
SEPARATOR_MIN_WIDTH_RATIO = 0.7   # 70% of page width
BACKGROUND_MIN_WIDTH_RATIO = 0.8  # 80% of page width

# =============================
# Data Classes
# =============================

ShapeType = Literal["rect", "line", "curve", "unknown"]
ShapeSemantic = Literal["table_grid", "underline", "separator", "background_band", "other"]
Orientation = Literal["horizontal", "vertical", "unknown"]

ColorValue = Union[
    float,                         # grayscale (0–1)
    Tuple[float, float, float],    # RGB
    Tuple[float, float, float, float],  # CMYK
]


@dataclass
class EnhancedShape:
    # Identity
    page_number: int
    shape_id: int
    raw_shape_ids: List[int]
    candidate_group_id: int

    # Geometry
    x_left: float
    x_right: float
    y_top: float
    y_bottom: float
    width: float
    height: float
    area: float

    # Raw drawing info
    raw_shape_type: ShapeType
    linewidth: Optional[float]
    fill: Optional[bool]
    stroke: Optional[bool]
    paint_op: Optional[str]
    non_stroking_color: Optional[ColorValue]
    stroking_color: Optional[ColorValue]

    # Derived / enhanced
    shape_type: ShapeType        # 'rect' may change to 'line'
    shape_orientation: Orientation              # horiz / vert / unknown
    table_id: Optional[int]               # None if not part of table grid
    shape_semantic: ShapeSemantic         # table_grid / underline / separator / background_band / other

    # Intersection analysis (for table detection)
    has_intersection: bool                # True if this line intersects with perpendicular lines
    intersection_count: int               # Number of perpendicular lines this line intersects
    intersecting_line_ids: List[int]      # enhanced_shape_ids of intersecting lines

    # Color meta (for debugging / heuristics)
    color_hex: Optional[str]
    color_label: Optional[str]            # "light_gray", "brand_purple", etc.


@dataclass
class CandidateGroup:
    # Identity
    group_id: int
    page_number: int
    raw_shape_ids: List[int]
    group_orientation: Literal["horizontal", "vertical"]

    # Group bounding box (union of all shapes in the group)
    x_left: float
    x_right: float
    y_top: float
    y_bottom: float

    @property
    def width(self) -> float:
        return self.x_right - self.x_left

    @property
    def height(self) -> float:
        return self.y_bottom - self.y_top

    @property
    def is_singleton(self) -> bool:
        return len(self.raw_shape_ids) == 1

    @classmethod
    def from_shapes_df(
        cls,
        *,
        group_id: int,
        page_number: int,
        shape_orientation: Literal["horizontal", "vertical"],
        shapes_df: pd.DataFrame,
    ) -> "CandidateGroup":
        """
        `shapes_df` is the subset of the shapes table belonging to this group
        (already filtered by page and orientation).
        Must contain: shape_id, x0, x1, top, bottom.
        """
        x_left = float(shapes_df["x_left"].min())
        x_right = float(shapes_df["x_right"].max())
        y_top = float(shapes_df["y_top"].min())
        y_bottom = float(shapes_df["y_bottom"].max())

        return cls(
            group_id=group_id,
            page_number=page_number,
            raw_shape_ids=shapes_df["raw_shape_id"].astype(int).tolist(),
            group_orientation=shape_orientation,
            x_left=x_left,
            x_right=x_right,
            y_top=y_top,
            y_bottom=y_bottom,
        )


# =============================
# HELPERS
# =============================

def _ensure_shape_columns(
    df: pd.DataFrame,
    *,
    step_name: str = "enhance_shapes",
    required_cols: Iterable[str] | None = None,
) -> None:
    """
    Ensure all required columns for shape processing are present.

    Args:
        df: Input DataFrame to validate.
        step_name: Name of the step/function for error context.
        required_cols: Optional override for required columns. If None,
            uses the default shape schema.

    Raises:
        ValueError: If one or more required columns are missing.
    """
    if required_cols is None:
        required_cols = [
            "page_number",
            "raw_shape_id",
            "raw_shape_type",
            "x_left",
            "y_top",
            "x_right",
            "y_bottom",
            "width",
            "height",
            "area",
            "non_stroking_color",
            "stroking_color",
            "linewidth",
            "fill",
            "stroke",
            "paint_op",
        ]

    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"{step_name}: missing required columns: {missing}")


def add_raw_orientation(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a raw_orientation column based on shape dimensions.
    
    Returns:
        DataFrame with additional 'raw_orientation' column:
        - 'horizontal': width > height
        - 'vertical': height > width  
        - 'square': width ≈ height (within 1px tolerance)
    """
    df = df.copy()
    
    def classify_orientation(row):
        if abs(row['width'] - row['height']) <= 1.0:
            return 'square'
        elif row['width'] > row['height']:
            return 'horizontal'
        else:
            return 'vertical'
    
    df['raw_orientation'] = df.apply(classify_orientation, axis=1)
    
    return df


# =============================
# STEP 1: Merge Shapes
# =============================

# ---- Build Horizontal Candidate Groups ----

def make_horizontal_candidate_groups(
    df: pd.DataFrame,
    *,
    start_group_id: int = 1,
) -> Tuple[List[CandidateGroup], int]:
    """
    Returns:
        Tuple of (groups, next_group_id) where next_group_id is the next available ID
    """
    groups: List[CandidateGroup] = []
    group_id = start_group_id

    horiz = df[df["raw_orientation"].isin(["horizontal", "square"])].copy()

    for page, page_df in horiz.groupby("page_number"):
        remaining = page_df.sort_values("raw_shape_id").copy()

        while len(remaining):
            anchor = remaining.iloc[0]
            anchor_top = anchor["y_top"]
            anchor_bottom = anchor["y_bottom"]

            mask = (
                (remaining["y_top"] - anchor_top).abs() <= _GAP_TOL_PX
            ) & (
                (remaining["y_bottom"] - anchor_bottom).abs() <= _GAP_TOL_PX
            )

            band_df = remaining[mask]

            group = CandidateGroup.from_shapes_df(
                group_id=group_id,
                page_number=page,
                shape_orientation="horizontal",
                shapes_df=band_df,
            )
            groups.append(group)
            group_id += 1

            remaining = remaining[~remaining["raw_shape_id"].isin(group.raw_shape_ids)]

    return groups, group_id

from typing import List, Set


# ---- Build Vertical Candidate Groups ----

# Small helper: take vertical + square shapes that were considered in horizontal pass, but ended up as singletons
def get_vertical_candidate_df(
    df: pd.DataFrame,
    horizontal_enhanced: List[EnhancedShape],
) -> pd.DataFrame:
    """
    Build the dataframe used for vertical grouping:
    - all shapes with raw_orientation == 'vertical'
    - plus shapes with raw_orientation == 'square' that only formed
      a singleton EnhancedShape in the horizontal pass.
    """
    # all square shapes in original df
    square_ids: Set[int] = set(
        df.loc[df["raw_orientation"] == "square", "raw_shape_id"].astype(int)
    )

    # shapes that are singletons after horizontal enhanced-pass
    singleton_ids: Set[int] = {
        sid
        for es in horizontal_enhanced
        if len(es.raw_shape_ids) == 1
        for sid in es.raw_shape_ids
    }

    square_singletons: Set[int] = square_ids & singleton_ids

    vertical_candidates = df[
        (df["raw_orientation"] == "vertical")
        | (df["raw_shape_id"].isin(square_singletons))
    ].copy()

    return vertical_candidates


def make_vertical_candidate_groups(
    df: pd.DataFrame,
    horizontal_enhanced: List[EnhancedShape],
    *,
    start_group_id: int = 1,
) -> Tuple[List[CandidateGroup], int]:
    """
    Make vertical CandidateGroups from:
    - all vertical shapes
    - plus square shapes that were singleton EnhancedShapes
      after the horizontal pass.
    
    Returns:
        Tuple of (groups, next_group_id) where next_group_id is the next available ID
    """
    vertical_df = get_vertical_candidate_df(df, horizontal_enhanced)

    groups: List[CandidateGroup] = []
    group_id = start_group_id

    for page, page_df in vertical_df.groupby("page_number"):
        remaining = page_df.sort_values("raw_shape_id").copy()

        while len(remaining):
            anchor = remaining.iloc[0]
            anchor_x0 = anchor["x_left"]
            anchor_x1 = anchor["x_right"]

            # SAME IDEA as horizontal: both x0 and x1 must be within tolerance
            mask = (
                (remaining["x_left"] - anchor_x0).abs() <= _GAP_TOL_PX
            ) & (
                (remaining["x_right"] - anchor_x1).abs() <= _GAP_TOL_PX
            )

            band_df = remaining[mask]

            group = CandidateGroup.from_shapes_df(
                group_id=group_id,
                page_number=page,
                shape_orientation="vertical",
                shapes_df=band_df,
            )
            groups.append(group)
            group_id += 1

            remaining = remaining[~remaining["raw_shape_id"].isin(group.raw_shape_ids)]

    return groups, group_id


# ---- Split Candidate Groups into EnhancedShapes ----

def _build_enhanced_shape_from_run(
    df: pd.DataFrame,
    group: CandidateGroup,
    raw_shape_ids: List[int],
    shape_id: int,
) -> EnhancedShape:
    """
    Build a fully-populated EnhancedShape from:
    - the parent CandidateGroup
    - the list of constituent shape_ids
    - the original shapes dataframe

    For metadata (color, linewidth, etc.), we take the first shape in the run
    as the representative.
    """
    sub = df[df["raw_shape_id"].isin(raw_shape_ids)].copy()
    sub = sub.sort_values("raw_shape_id")

    rep = sub.iloc[0]

    # Geometry: use the union of the shapes (in case it differs from group bbox)
    x_left = float(sub["x_left"].min())
    x_right = float(sub["x_right"].max())
    y_top = float(sub["y_top"].min())
    y_bottom = float(sub["y_bottom"].max())
    width = x_right - x_left
    height = y_bottom - y_top
    area = width * height

    shape_orientation: Orientation
    if group.group_orientation == "horizontal":
        shape_orientation = "horizontal"
    elif group.group_orientation == "vertical":
        shape_orientation = "vertical"
    else:
        shape_orientation = "unknown"

    # Color & drawing info: take from representative row (index 0)
    non_stroking_color = rep.get("non_stroking_color")
    stroking_color = rep.get("stroking_color")
    linewidth = rep.get("linewidth")
    fill = rep.get("fill")
    stroke = rep.get("stroke")
    paint_op = rep.get("paint_op")

    # Adjust shape_type: convert thin rects/curves to lines based on orientation
    raw_shape_type: ShapeType = rep["raw_shape_type"]
    shape_type: ShapeType = raw_shape_type
    
    if shape_type in ("rect", "curve"):
        if shape_orientation == "horizontal" and height <= LINE_HEIGHT_MAX_PX:
            shape_type = "line"
        elif shape_orientation == "vertical" and width <= LINE_HEIGHT_MAX_PX:
            shape_type = "line"

    # Defaults for things you'll fill in later in the pipeline
    table_id: Optional[int] = None
    shape_semantic: ShapeSemantic = "other"
    has_intersection: bool = False
    intersection_count: int = 0
    intersecting_line_ids: List[int] = []
    color_hex: Optional[str] = None
    color_label: Optional[str] = None

    return EnhancedShape(
        # Identity
        page_number=int(rep["page_number"]),
        shape_id=shape_id,
        raw_shape_ids=[int(sid) for sid in raw_shape_ids],
        candidate_group_id=group.group_id,

        # Geometry
        x_left=x_left,
        x_right=x_right,
        y_top=y_top,
        y_bottom=y_bottom,
        width=width,
        height=height,
        area=area,

        # Raw drawing info (from rep)
        raw_shape_type=raw_shape_type,
        linewidth=float(linewidth) if linewidth is not None else None,
        fill=bool(fill) if fill is not None else None,
        stroke=bool(stroke) if stroke is not None else None,
        paint_op=str(paint_op) if paint_op is not None else None,
        non_stroking_color=non_stroking_color,
        stroking_color=stroking_color,

        # Derived / enhanced
        shape_type=shape_type,
        shape_orientation=shape_orientation,
        table_id=table_id,
        shape_semantic=shape_semantic,

        # Intersection analysis
        has_intersection=has_intersection,
        intersection_count=intersection_count,
        intersecting_line_ids=intersecting_line_ids,

        # Color meta
        color_hex=color_hex,
        color_label=color_label,
    )


def _split_candidate_group(
    df: pd.DataFrame,
    group: CandidateGroup,
    *,
    start_id: int,
    sort_col: str,
    gap_ref_col: str,
    gap_to_col: str,
) -> List[EnhancedShape]:
    sub = df[df["raw_shape_id"].isin(group.raw_shape_ids)].copy()
    sub = sub.sort_values(sort_col)

    enhanced: List[EnhancedShape] = []
    current_ids: List[int] = []
    current_x0 = current_x1 = current_top = current_bottom = None
    prev_gap_to: float | None = None
    next_id = start_id

    for _, row in sub.iterrows():
        sid = int(row["raw_shape_id"])
        sx0 = float(row["x_left"])
        sx1 = float(row["x_right"])
        sy_top = float(row["y_top"])
        sy_bottom = float(row["y_bottom"])

        gap_ref = float(row[gap_ref_col])
        gap_to = float(row[gap_to_col])

        if prev_gap_to is None:
            current_ids = [sid]
            current_x0 = sx0
            current_x1 = sx1
            current_top = sy_top
            current_bottom = sy_bottom
        else:
            gap = gap_ref - prev_gap_to
            if gap <= _CHAIN_TOL_PX:
                current_ids.append(sid)
                current_x0 = min(current_x0, sx0)
                current_x1 = max(current_x1, sx1)
                current_top = min(current_top, sy_top)
                current_bottom = max(current_bottom, sy_bottom)
            else:
                # close previous run -> build EnhancedShape
                enhanced.append(
                    _build_enhanced_shape_from_run(
                        df=df,
                        group=group,
                        raw_shape_ids=current_ids,
                        shape_id=next_id,
                    )
                )
                next_id += 1

                # start new run
                current_ids = [sid]
                current_x0 = sx0
                current_x1 = sx1
                current_top = sy_top
                current_bottom = sy_bottom

        # Update prev_gap_to to the maximum extent of the current run, not just the current shape
        # For horizontal: use current_x1 (max x1 of run), for vertical: use current_bottom (max bottom of run)
        prev_gap_to = current_x1 if gap_to_col == "x_right" else current_bottom

    if current_ids:
        enhanced.append(
            _build_enhanced_shape_from_run(
                df=df,
                group=group,
                raw_shape_ids=current_ids,
                shape_id=next_id,
            )
        )

    return enhanced



def enhanced_shapes_from_horizontal_candidate_group(
    df: pd.DataFrame,
    group: CandidateGroup,
    *,
    start_id: int = 1,
) -> List[EnhancedShape]:
    # horizontal: chain along x, sort by x0, gap = x0 - prev_x1
    return _split_candidate_group(
        df,
        group,
        start_id=start_id,
        sort_col="x_left",
        gap_ref_col="x_left",
        gap_to_col="x_right",
    )


def enhanced_shapes_from_vertical_candidate_group(
    df: pd.DataFrame,
    group: CandidateGroup,
    *,
    start_id: int = 1,
) -> List[EnhancedShape]:
    # vertical: chain along y, sort by top, gap = top - prev_bottom
    return _split_candidate_group(
        df,
        group,
        start_id=start_id,
        sort_col="y_top",
        gap_ref_col="y_top",
        gap_to_col="y_bottom",
    )

# ---- Merge Shapes Handler ----

def _merge_shapes(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    High-level wrapper to merge low-level shapes into EnhancedShapes.
    
    Processes each page sequentially (horizontal + vertical) to ensure
    enhanced_shape_id and candidate_group_id increment naturally page-by-page.

    Returns:
        DataFrame with one row per EnhancedShape, columns taken from
        the EnhancedShape dataclass.
    """
    df = df.copy()
    df = add_raw_orientation(df)

    all_enhanced: List[EnhancedShape] = []
    next_group_id = 1
    next_shape_id = 1

    # Process each page completely (horizontal + vertical) before moving to next
    for page_number in sorted(df['page_number'].unique()):
        page_df = df[df['page_number'] == page_number].copy()
        
        # 1) Horizontal candidate groups for this page
        horizontal_groups, next_group_id = make_horizontal_candidate_groups(
            page_df, start_group_id=next_group_id
        )
        
        # 2) Create horizontal EnhancedShapes for this page
        page_horizontal_enhanced: List[EnhancedShape] = []
        for g in horizontal_groups:
            group_shapes = enhanced_shapes_from_horizontal_candidate_group(
                page_df,
                g,
                start_id=next_shape_id,
            )
            page_horizontal_enhanced.extend(group_shapes)
            next_shape_id += len(group_shapes)
        
        # 3) Vertical candidate groups for this page (vertical + square singletons)
        vertical_groups, next_group_id = make_vertical_candidate_groups(
            page_df, page_horizontal_enhanced, start_group_id=next_group_id
        )
        
        # 4) Create vertical EnhancedShapes for this page
        for g in vertical_groups:
            group_shapes = enhanced_shapes_from_vertical_candidate_group(
                page_df,
                g,
                start_id=next_shape_id,
            )
            page_horizontal_enhanced.extend(group_shapes)
            next_shape_id += len(group_shapes)
        
        # Add all enhanced shapes from this page to the master list
        all_enhanced.extend(page_horizontal_enhanced)

    # 5) Convert to DataFrame
    records = [asdict(es) for es in all_enhanced]
    result_df = pd.DataFrame.from_records(records)

    return result_df


#============================
# STEP 2: Detect Line Intersections
#============================

def _detect_line_intersections(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect intersections between horizontal and vertical lines.
    
    Updates has_intersection, intersection_count, and intersecting_line_ids
    for each line that intersects with perpendicular lines.
    
    Args:
        df: DataFrame with EnhancedShapes (must have adjusted_shape_type='line')
    
    Returns:
        DataFrame with intersection properties populated
    """
    if df.empty:
        return df
    
    df = df.copy()
    
    # Only check lines (not other shape types)
    lines_df = df[df['shape_type'] == 'line'].copy()
    
    # Process page by page for efficiency
    for page_number in lines_df['page_number'].unique():
        page_lines = lines_df[lines_df['page_number'] == page_number]
        
        # Separate horizontal and vertical lines
        h_lines = page_lines[page_lines['shape_orientation'] == 'horizontal']
        v_lines = page_lines[page_lines['shape_orientation'] == 'vertical']
        
        # Check each horizontal line against each vertical line
        for h_idx, h_line in h_lines.iterrows():
            h_x0, h_x1 = h_line['x_left'], h_line['x_right']
            h_y = (h_line['y_top'] + h_line['y_bottom']) / 2  # midpoint
            intersecting_ids = []
            
            for v_idx, v_line in v_lines.iterrows():
                v_x = (v_line['x_left'] + v_line['x_right']) / 2  # midpoint
                v_y0, v_y1 = v_line['y_top'], v_line['y_bottom']
                
                # Check if lines intersect
                if h_x0 <= v_x <= h_x1 and v_y0 <= h_y <= v_y1:
                    intersecting_ids.append(int(v_line['shape_id']))
            
            # Update horizontal line
            if intersecting_ids:
                df.loc[h_idx, 'has_intersection'] = True
                df.loc[h_idx, 'intersection_count'] = len(intersecting_ids)
                df.loc[h_idx, 'intersecting_line_ids'] = str(intersecting_ids)
        
        # Check each vertical line against each horizontal line (bi-directional)
        for v_idx, v_line in v_lines.iterrows():
            v_x = (v_line['x_left'] + v_line['x_right']) / 2
            v_y0, v_y1 = v_line['y_top'], v_line['y_bottom']
            intersecting_ids = []
            
            for h_idx, h_line in h_lines.iterrows():
                h_x0, h_x1 = h_line['x_left'], h_line['x_right']
                h_y = (h_line['y_top'] + h_line['y_bottom']) / 2
                
                # Check if lines intersect
                if h_x0 <= v_x <= h_x1 and v_y0 <= h_y <= v_y1:
                    intersecting_ids.append(int(h_line['shape_id']))
            
            # Update vertical line
            if intersecting_ids:
                df.loc[v_idx, 'has_intersection'] = True
                df.loc[v_idx, 'intersection_count'] = len(intersecting_ids)
                df.loc[v_idx, 'intersecting_line_ids'] = str(intersecting_ids)
    
    return df




# =============================
# Public API
# =============================

def enhance_shapes(
    df_shapes: pd.DataFrame,
    *,
    merge_lines: bool = True,
    add_color_meta: bool = False,
) -> pd.DataFrame:
    """
    Enhance raw shapes DataFrame from pdfplumber with:
    - normalized / adjusted shape_type and orientation
    - merged logical lines (optional)
    - rudimentary table detection (table_id)
    - simple line semantics (table_grid / separator / background_band / other)
    - color meta (hex, label, is_grayscale)

    Expected input columns (from extract_shapes step):

        page_number,
        shape_id
        shape_kind,   # 'rect', 'line', 'curve', ...
        x_left, y_top, x_right, y_bottom, width, height,
        area,
        non_stroking_color, stroking_color,
        linewidth, fill, stroke, paint_op

    Returns:
        DataFrame with all original columns plus:
        - shape_type
        - shape_orientation
        - enhanced_shape_id
        - raw_shape_ids (list[int])
        - table_id
        - shape_semantic
        - color_hex (optional) - not added for now
        - color_label (optional) - not added for now
    """
    if df_shapes.empty:
        return df_shapes.copy()

    df = df_shapes.copy()

    _ensure_shape_columns(df)

    # Add grouping traceability for original shapes
    df["raw_shape_ids"] = df["raw_shape_id"].astype(int).map(lambda v: [v])

    # Optionally merge line-like shapes into logical shapes
    if merge_lines:
        df = _merge_shapes(df)
    
    # Detect line intersections (for table detection)
    #df = _detect_line_intersections(df)

    return df

