import pandas as pd
import numpy as np
from typing import Any

from ._utils.table_type_column_assigner import add_table_type_column

# =====================
# Config
# =====================

_MAX_VERTICAL_GAP = 8.0  # max vertical gap between alike horizontal bands to allow merging

# =====================
# Assign column layout to cells
# =====================

def assign_column_layout(df: pd.DataFrame) -> pd.DataFrame:

    def ranges_overlap(a_left: float, a_right: float,
                       b_left: float, b_right: float) -> bool:
        # strict math: 0 overlap → False, any positive overlap → True
        return a_left < b_right and b_left < a_right
    
    def find_overlapping_cols(x_left: float, x_right: float, columns: list[dict]) -> list[int]:
        overlapping = []
        for idx, col in enumerate(columns):
            if ranges_overlap(x_left, x_right, col["x_left"], col["x_right"]):
                overlapping.append(idx)
        return overlapping

    def process_cell(x_left: float, x_right: float, columns: list[dict]) -> list[int]:
        """
        Process a cell against existing columns.

        Rules:
        - 0 hits   → NEW COLUMN (inserted at correct x position)
        - 1 hit    → EXTEND that single column (union of intervals)
        - >= 2 hits → COLSPAN: DO NOT mutate columns, just return the hits
        """
        overlapping = find_overlapping_cols(x_left, x_right, columns)

        # NEW COLUMN
        if len(overlapping) == 0:
            insert_pos = len(columns)
            for idx, col in enumerate(columns):
                if x_right <= col["x_left"]:
                    insert_pos = idx
                    break
            columns.insert(insert_pos, {"x_left": x_left, "x_right": x_right})
            return [insert_pos]

        # EXTEND (exactly one hit)
        if len(overlapping) == 1:
            idx = overlapping[0]
            col = columns[idx]
            col["x_left"] = min(x_left, col["x_left"])
            col["x_right"] = max(x_right, col["x_right"])
            return overlapping

        # COLSPAN: two or more hits → do not change boundaries
        return overlapping

    def maybe_split_columns_for_line(line_cells: pd.DataFrame, columns: list[dict]) -> None:
        """
        Detect SPLIT situations for this line and mutate `columns` accordingly.

        SPLIT rule:
        - For a given column index k, if in this line there are >=2 cells
          that each hit ONLY column k, and their intervals are disjoint,
          we split that column into multiple subcolumns, one per disjoint segment.
        """
        # Collect sole-hit intervals per column index
        sole_hits_by_col: dict[int, list[tuple[float, float]]] = {}

        for cell in line_cells.itertuples():
            hits = find_overlapping_cols(cell.x_left, cell.x_right, columns)
            if len(hits) == 1:
                col_idx = hits[0]
                sole_hits_by_col.setdefault(col_idx, []).append(
                    (float(cell.x_left), float(cell.x_right))
                )

        # Determine which columns need splitting
        splits: list[tuple[int, list[tuple[float, float]]]] = []

        for col_idx, intervals in sole_hits_by_col.items():
            if len(intervals) < 2:
                continue

            # Sort intervals by left edge
            intervals_sorted = sorted(intervals, key=lambda t: t[0])

            # Merge overlapping intervals into segments; count disjoint segments
            segments: list[tuple[float, float]] = []
            cur_left, cur_right = intervals_sorted[0]

            for a, b in intervals_sorted[1:]:
                if ranges_overlap(cur_left, cur_right, a, b):
                    # extend current segment
                    cur_left = min(cur_left, a)
                    cur_right = max(cur_right, b)
                else:
                    # disjoint → close current segment and start new
                    segments.append((cur_left, cur_right))
                    cur_left, cur_right = a, b

            segments.append((cur_left, cur_right))

            if len(segments) > 1:
                # We want to split this column into multiple subcolumns
                splits.append((col_idx, segments))

        # Apply splits from right to left so indices don't shift on us
        for col_idx, segments in sorted(splits, key=lambda t: t[0], reverse=True):
            # Replace columns[col_idx] by one new column per segment
            new_cols = [{"x_left": s_left, "x_right": s_right} for (s_left, s_right) in segments]
            # Keep all other columns as-is
            columns[col_idx:col_idx+1] = new_cols  # in-place slice assignment

    # ------------------ main body ------------------ #

    result_df = df.copy()
    result_df["col_start"] = -1
    result_df["col_end"] = -1
    result_df["colspan"] = -1
    result_df["band_total_cols"] = -1

    grouped = result_df.groupby(["page_number", "horizontal_band_id"])

    for (page, band), band_df in grouped:
        line_cell_counts = band_df.groupby("temp_line_id").size()
        max_line_id = line_cell_counts.idxmax()

        # Seed columns from base line
        max_line_cells = (
            band_df[band_df["temp_line_id"] == max_line_id]
            .sort_values("x_left")
        )

        columns: list[dict] = []
        for _, cell in max_line_cells.iterrows():
            columns.append({"x_left": cell["x_left"], "x_right": cell["x_right"]})

        # Decide processing order: first DOWN, then UP
        all_line_ids = sorted(band_df["temp_line_id"].unique())
        max_line_idx = all_line_ids.index(max_line_id)
        lines_below = all_line_ids[max_line_idx + 1:]              # DOWN (larger temp_line_id)
        lines_above = list(reversed(all_line_ids[:max_line_idx]))  # UP   (smaller temp_line_id)

        # -------- PHASE 1: DOWN --------
        for line_id in lines_below:
            line_cells = band_df[band_df["temp_line_id"] == line_id].sort_values("x_left")

            # SPLIT detection for this line (based on sole hits)
            maybe_split_columns_for_line(line_cells, columns)

            for cell in line_cells.itertuples():
                hits = find_overlapping_cols(cell.x_left, cell.x_right, columns)
                hit_cols = [h for h in hits]  # 0-based

            # Now mutate columns (extend / new column) for these cells
            for _, cell in line_cells.iterrows():
                _ = process_cell(float(cell["x_left"]), float(cell["x_right"]), columns)

        # -------- PHASE 2: UP --------
        for line_id in lines_above:
            line_cells = band_df[band_df["temp_line_id"] == line_id].sort_values("x_left")

            # SPLIT detection for this line too
            maybe_split_columns_for_line(line_cells, columns)

            for cell in line_cells.itertuples():
                hits = find_overlapping_cols(cell.x_left, cell.x_right, columns)
                hit_cols = [h for h in hits]  # 0-based

            for _, cell in line_cells.iterrows():
                _ = process_cell(float(cell["x_left"]), float(cell["x_right"]), columns)

        # -------- FINAL ASSIGNMENT for this band --------
        total_cols = len(columns)
        band_mask = ((result_df["page_number"] == page) &
                     (result_df["horizontal_band_id"] == band))

        # For every cell in this band, assign col_start/end/span
        for cell in band_df.itertuples():
            hits = find_overlapping_cols(cell.x_left, cell.x_right, columns)

            if not hits:
                # In clean data this shouldn't happen; if it does, treat as new rightmost col
                col_start = total_cols
                col_end = col_start
            else:
                col_start = hits[0]  # 0-based
                col_end = hits[-1]

            colspan = col_end - col_start + 1

            result_df.loc[result_df["cell_id"] == cell.cell_id, "col_start"] = col_start
            result_df.loc[result_df["cell_id"] == cell.cell_id, "col_end"] = col_end
            result_df.loc[result_df["cell_id"] == cell.cell_id, "colspan"] = colspan

        result_df.loc[band_mask, "band_total_cols"] = total_cols

    return result_df

# =====================
# Merge alike horizontal bands in 1
# =====================


def find_mergeable_bands(df, max_vertical_gap=_MAX_VERTICAL_GAP):
    """
    Identifies which horizontal_band_ids can be merged together.
    Bands with band_total_cols=1 are always treated as separate layouts.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with columns: page_number, horizontal_band_id, temp_line_id,
        col_start, col_end, band_total_cols, y_top, y_bottom
    max_vertical_gap : float
        Maximum vertical distance (in PDF points) between bands to allow merging (default: 8)
    
    Returns:
    --------
    dict
        Mapping of {original_band_id: layout_id}
        layout_id counts up from 1 sequentially
    """
    
    def get_column_positions(band_cells):
        """Get sorted unique column boundaries for a band."""
        positions = set()
        for _, cell in band_cells.iterrows():
            positions.add(cell['col_start'])
            positions.add(cell['col_end'] + 1)  # +1 to get right boundary
        return sorted(positions)
    
    band_to_group = {}  # Maps band_id to merge group identifier
    layout_counter = 0
    
    # Group by page_number first
    for page, page_df in df.groupby('page_number'):
        # Get all bands for this page, sorted
        bands = sorted(page_df['horizontal_band_id'].unique())
        
        current_merge_group = None
        
        for i in range(len(bands)):
            current_band_id = bands[i]
            current_band = page_df[page_df['horizontal_band_id'] == current_band_id]
            current_total = current_band['band_total_cols'].iloc[0]
            
            # Check if this band has only 1 column - always separate
            if current_total == 1:
                layout_counter += 1
                band_to_group[current_band_id] = layout_counter
                current_merge_group = None  # Reset merge group
                continue
            
            # First multi-column band or can't merge with previous
            can_merge = False
            
            if i > 0 and current_merge_group is not None:
                prev_band_id = bands[i - 1]
                prev_band = page_df[page_df['horizontal_band_id'] == prev_band_id]
                prev_total = prev_band['band_total_cols'].iloc[0]
                
                # Only try to merge if previous band wasn't single-column
                if prev_total > 1:
                    # Check 1: Are they consecutive?
                    if current_band_id == prev_band_id + 1:
                        # Check 2: Do they have equal band_total_cols?
                        if current_total == prev_total:
                            # Check 3: Do their columns align perfectly?
                            prev_positions = get_column_positions(prev_band)
                            current_positions = get_column_positions(current_band)
                            
                            if current_positions == prev_positions:
                                # Check 4: Is the vertical gap small enough?
                                # Get the maximum y_bottom from previous band
                                prev_y_bottom = prev_band['y_bottom'].max()
                                
                                # Get the minimum y_top from current band
                                current_y_top = current_band['y_top'].min()
                                
                                # Calculate vertical gap
                                vertical_gap = current_y_top - prev_y_bottom
                                
                                if vertical_gap <= max_vertical_gap:
                                    # All checks passed - can merge!
                                    can_merge = True
            
            if can_merge:
                # Merge with previous group
                band_to_group[current_band_id] = current_merge_group
            else:
                # Start new layout
                layout_counter += 1
                band_to_group[current_band_id] = layout_counter
                current_merge_group = layout_counter
    
    return band_to_group


def apply_band_merge(df, band_mapping):
    """
    Apply the band merge mapping to the dataframe.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Original dataframe
    band_mapping : dict
        Mapping from find_mergeable_bands()
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with layout_id column added
    """
    result_df = df.copy()
    result_df['layout_id'] = result_df['horizontal_band_id'].map(band_mapping)
    return result_df


# =====================
# Classify multi-column layout_id's as table or text_multicol
# =====================

def add_average_table_score(df):
    """
    Calculates the average table_row_score for each layout_id and adds it as a column.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with columns: layout_id, temp_line_id, table_row_score
        (table_row_score is the same for all cells within a temp_line_id)
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with added column: average_table_score
    """
    result_df = df.copy()
    
    # Get unique line scores for each layout
    # Since all cells in a line have the same table_row_score, we can just take one per line
    line_scores = (df.groupby(['layout_id', 'temp_line_id'])['table_row_score']
                   .first()  # Get one score per line (they're all the same)
                   .reset_index())
    
    # Calculate average score per layout
    layout_avg_scores = (line_scores.groupby('layout_id')['table_row_score']
                         .mean()
                         .reset_index()
                         .rename(columns={'table_row_score': 'average_table_score'}))
    
    # Merge back to original dataframe
    result_df = result_df.merge(layout_avg_scores, on='layout_id', how='left')
    
    return result_df


def classify_layout_types(df):
    """
    Classifies each layout_id into layout_type based on column count and scoring.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with columns: layout_id, band_total_cols, temp_line_id, 
        shape_id_underline, average_table_score
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with added column: layout_type
    """
    
    def score_layout(layout_df):
        """Calculate score for a multi-column layout."""
        score = 0
        
        # Get band_total_cols (same for all cells in layout)
        band_total_cols = layout_df['band_total_cols'].iloc[0]
        
        # Criterion 1: Band total columns
        if 2 <= band_total_cols <= 6:
            score += 1
        elif band_total_cols >= 7:
            score += 2
        
        # Criterion 2: Distinct shape_id_underline values
        if 'shape_id_underline' in layout_df.columns:
            distinct_underline_ids = layout_df['shape_id_underline'].nunique()
            
            if distinct_underline_ids >= 2:
                score += 1
        
        # Criterion 3: Average table score
        if 'average_table_score' in layout_df.columns:
            avg_table_score = layout_df['average_table_score'].iloc[0]
            
            if avg_table_score < 1:
                score += 0
            elif 1 <= avg_table_score < 2:
                score += 1
            elif avg_table_score >= 2:
                score += 2
        
        return score
    
    result_df = df.copy()
    
    # Group by layout_id and classify
    layout_types = {}
    
    for layout_id, layout_df in result_df.groupby('layout_id'):
        band_total_cols = layout_df['band_total_cols'].iloc[0]
        
        if band_total_cols == 1:
            layout_types[layout_id] = 'text_singlecol'
        else:
            # Multi-column layout - calculate score
            score = score_layout(layout_df)
            
            if score == 1:
                layout_types[layout_id] = 'text_multicol'
            else:  # score >= 2
                layout_types[layout_id] = 'table'
    
    # Map layout types back to dataframe
    result_df['layout_type'] = result_df['layout_id'].map(layout_types)
    
    # Update block_role for table layouts
    if 'block_role' not in result_df.columns:
        result_df['block_role'] = pd.NA
    
    result_df.loc[result_df['layout_type'] == 'table', 'block_role'] = 'table'
    
    return result_df


# =====================
# Convert single-line text_multicol to text_singlecol
# =====================

#TODO: Recalculation logic should be centralized

def convert_single_line_multicol_to_singlecol(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert layouts that consist of only 1 temp_line_id with layout_type = text_multicol
    to text_singlecol by merging all cells into a single cell.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with columns: layout_id, temp_line_id, layout_type, cell_id,
        and all cell properties (x_left, x_right, y_top, y_bottom, text, etc.)
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with converted layouts and merged cells
    """
    result_df = df.copy()
    
    # Find layouts that need conversion
    # Group by layout_id and check if they have only 1 unique temp_line_id and are text_multicol
    layout_stats = result_df.groupby('layout_id').agg({
        'temp_line_id': 'nunique',
        'layout_type': 'first'
    }).reset_index()
    
    # Filter: only 1 temp_line_id AND layout_type = text_multicol
    layouts_to_convert = layout_stats[
        (layout_stats['temp_line_id'] == 1) & 
        (layout_stats['layout_type'] == 'text_multicol')
    ]['layout_id'].tolist()
    
    if not layouts_to_convert:
        # No layouts to convert
        return result_df
    
    # Helper function to merge cells
    def _mode_or_first(series: pd.Series) -> Any:
        """Get mode or first non-null value."""
        if series.empty:
            return None
        vc = series.value_counts(dropna=True)
        if not vc.empty:
            return vc.index[0]
        return series.dropna().iloc[0] if series.dropna().size else None
    
    # Process each layout that needs conversion
    cells_to_remove = []
    new_cells = []
    
    for layout_id in layouts_to_convert:
        layout_cells = result_df[result_df['layout_id'] == layout_id].copy()
        
        if layout_cells.empty:
            continue
        
        # Mark all cells in this layout for removal
        cells_to_remove.extend(layout_cells['cell_id'].tolist())
        
        # Merge all cells into one
        # Get the first cell as a base (to preserve most metadata)
        first_cell = layout_cells.iloc[0].copy()
        
        # Recalculate geometry
        first_cell['x_left'] = layout_cells['x_left'].min()
        first_cell['x_right'] = layout_cells['x_right'].max()
        first_cell['y_top'] = layout_cells['y_top'].min()
        first_cell['y_bottom'] = layout_cells['y_bottom'].max()
        first_cell['width'] = first_cell['x_right'] - first_cell['x_left']
        first_cell['height'] = first_cell['y_bottom'] - first_cell['y_top']
        
        # Merge text (join with spaces, filtering out empty strings)
        texts = layout_cells['text'].astype(str)
        first_cell['text'] = ' '.join(t.strip() for t in texts if t.strip() and t.strip() != 'nan')
        
        # Sum numeric counts
        sum_cols = ['char_count', 'alpha_count', 'digit_count', 'uppercase_count',
                   'word_count', 'alpha_word_count', 'capitalized_word_count']
        for col in sum_cols:
            if col in layout_cells.columns:
                first_cell[col] = layout_cells[col].sum()
        
        # Recalculate bold/italic ratios if we have char_count
        if 'char_count' in first_cell and first_cell['char_count'] > 0:
            # Estimate bold/italic char counts from ratios
            if 'bold_ratio' in layout_cells.columns:
                bold_char_est = (layout_cells['bold_ratio'].fillna(0.0) * 
                               layout_cells['char_count'].fillna(0.0)).sum()
                first_cell['bold_ratio'] = bold_char_est / first_cell['char_count']
            if 'italic_ratio' in layout_cells.columns:
                italic_char_est = (layout_cells['italic_ratio'].fillna(0.0) * 
                                 layout_cells['char_count'].fillna(0.0)).sum()
                first_cell['italic_ratio'] = italic_char_est / first_cell['char_count']
        
        # Use mode/first for other properties
        mode_cols = ['font_name', 'font_family', 'font_size', 'non_stroking_color',
                    'stroking_color', 'text_orientation']
        for col in mode_cols:
            if col in layout_cells.columns:
                first_cell[col] = _mode_or_first(layout_cells[col])
        
        # Merge word_ids if present
        if 'word_ids' in layout_cells.columns:
            all_word_ids = []
            for word_id_list in layout_cells['word_ids']:
                if isinstance(word_id_list, list):
                    all_word_ids.extend(word_id_list)
                elif pd.notna(word_id_list):
                    all_word_ids.append(word_id_list)
            first_cell['word_ids'] = all_word_ids if all_word_ids else []
        
        # Update layout-related columns
        first_cell['col_start'] = 1
        first_cell['col_end'] = 0
        first_cell['colspan'] = 1
        first_cell['band_total_cols'] = 1
        first_cell['layout_type'] = 'text_singlecol'
        
        # Preserve other columns from first cell (like doc_name, page_number, etc.)
        # but ensure we keep the layout_id
        first_cell['layout_id'] = layout_id
        
        new_cells.append(first_cell)
    
    # Remove old cells and add new merged cells
    if cells_to_remove:
        result_df = result_df[~result_df['cell_id'].isin(cells_to_remove)]
    
    if new_cells:
        # Convert list of Series to DataFrame
        # Each Series in new_cells is already a row with all necessary columns
        new_cells_df = pd.DataFrame(new_cells)
        
        # Ensure new_cells_df has all columns from result_df (in case of missing columns)
        for col in result_df.columns:
            if col not in new_cells_df.columns:
                # Use appropriate default based on column dtype
                if result_df[col].dtype == 'object':
                    new_cells_df[col] = None
                else:
                    new_cells_df[col] = pd.NA
        
        # Reorder columns to match result_df exactly
        new_cells_df = new_cells_df[result_df.columns]
        
        # Concatenate new cells
        result_df = pd.concat([result_df, new_cells_df], ignore_index=True)
    
    return result_df


# =====================
# Public API
# =====================

def build_layout(df_cells):
    df_cells_out = assign_column_layout(df_cells)
    band_mapping = find_mergeable_bands(df_cells_out)
    df_cells_out = apply_band_merge(df_cells_out, band_mapping)
    df_cells_out = add_average_table_score(df_cells_out)
    df_cells_out = classify_layout_types(df_cells_out)
    
    # Convert single-line text_multicol layouts to text_singlecol
    df_cells_out = convert_single_line_multicol_to_singlecol(df_cells_out)

    # =====================
    # Classify table types ["standard", "matrix", "narrative"]
    # =====================

    # 1) Work only on table layouts
    df_tables = df_cells_out[df_cells_out["layout_type"] == "table"].copy()

    # 2) Add table_type per layout_id (only if there are tables)
    if not df_tables.empty:
        df_tables = add_table_type_column(
            df_tables,
            group_cols=("layout_id",),   # ← runs classification per layout_id
        )

        # 3) Merge back (if you want table_type on the full df)
        df_cells_out = df_cells_out.merge(
            df_tables[["cell_id", "table_type"]],
            on="cell_id",
            how="left",
        )
    else:
        # No tables in document, add empty table_type column
        df_cells_out["table_type"] = None
    
    return df_cells_out
