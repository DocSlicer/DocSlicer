#!/usr/bin/env python3
"""
Pivot normalized table_cells_df into human-readable wide format.

Takes the normalized cell-based DataFrame and converts it back to
traditional table layout with proper headers and colspan handling.
"""
import pandas as pd
from pathlib import Path
import sys


def pivot_table_cells(df_cells: pd.DataFrame) -> pd.DataFrame:
    """
    Convert normalized table cells to pivoted wide format.
    
    Args:
        df_cells: DataFrame with columns: page_number, table_id, table_cell_id,
                  row_start, col_start, rowspan, colspan, text, role, page_label
    
    Returns:
        DataFrame in wide format with tables stacked vertically, separated by headers
    """
    # Handle empty DataFrames or missing required columns
    if df_cells.empty:
        return pd.DataFrame()
    
    required_cols = ['page_number', 'page_label', 'table_id']
    if not all(col in df_cells.columns for col in required_cols):
        return pd.DataFrame()
    
    all_tables = []
    
    # Group by page_number and table_id
    for (page_num, page_label, table_id), group in df_cells.groupby(
        ['page_number', 'page_label', 'table_id'], sort=True
    ):
        # Add separator row with page/table info
        separator_row = {
            'table_info': f">>> Page {page_label} | Table {table_id} <<<",
        }
        separator_df = pd.DataFrame([separator_row])
        all_tables.append(separator_df)
        
        # Get table dimensions
        max_row = group['row_start'].max() + 1
        max_col = group['col_start'].max() + group['colspan'].max()
        
        # Build grid matrix (store cell text)
        grid = [['' for _ in range(max_col)] for _ in range(max_row)]
        
        # Fill grid with cell content, handling colspan and rowspan
        for _, cell in group.iterrows():
            r_start = cell['row_start']
            c_start = cell['col_start']
            colspan = cell['colspan']
            rowspan = cell['rowspan'] if 'rowspan' in cell and pd.notna(cell['rowspan']) else 1
            text = cell['text'] if pd.notna(cell['text']) else ''
            
            # Duplicate text across both rowspan and colspan
            for dr in range(int(rowspan)):
                r = r_start + dr
                if r >= max_row:
                    continue
                for dc in range(int(colspan)):
                    c = c_start + dc
                    if c < max_col:
                        grid[r][c] = text
        
        # Convert grid to DataFrame
        col_names = [f"col_{i}" for i in range(max_col)]
        table_df = pd.DataFrame(grid, columns=col_names)
        
        # Add metadata columns
        table_df.insert(0, 'page_number', page_num)
        table_df.insert(1, 'page_label', page_label)
        table_df.insert(2, 'table_id', table_id)
        table_df.insert(3, 'row_num', range(len(table_df)))
        
        all_tables.append(table_df)
        
        # Add blank separator row
        blank_row = pd.DataFrame([['' for _ in range(len(table_df.columns))]], 
                                columns=table_df.columns)
        all_tables.append(blank_row)
    
    if not all_tables:
        return pd.DataFrame()
    
    # Concatenate all tables
    result = pd.concat(all_tables, ignore_index=True)
    return result


def main():
    """Load table_cells_df and create pivoted version."""
    
    # Default paths
    input_csv = Path("backend/app/services/parsing/html/test_output/table_cells_df.csv")
    output_csv = Path("backend/app/services/parsing/html/test_output/tables_pivoted.csv")
    
    # Allow command line override
    if len(sys.argv) > 1:
        input_csv = Path(sys.argv[1])
    if len(sys.argv) > 2:
        output_csv = Path(sys.argv[2])
    
    # Check if input exists
    if not input_csv.exists():
        print(f"❌ Input file not found: {input_csv}")
        print(f"   Usage: python {sys.argv[0]} [input_csv] [output_csv]")
        sys.exit(1)
    
    print(f"📖 Reading: {input_csv}")
    df_cells = pd.read_csv(input_csv)
    
    # Check required columns
    required_cols = ['page_number', 'table_id', 'row_start', 'col_start', 'colspan', 'text']
    missing_cols = [col for col in required_cols if col not in df_cells.columns]
    if missing_cols:
        print(f"❌ Missing required columns: {missing_cols}")
        sys.exit(1)
    
    # Add rowspan if missing (default to 1)
    if 'rowspan' not in df_cells.columns:
        df_cells['rowspan'] = 1
    
    # Add page_label if missing
    if 'page_label' not in df_cells.columns:
        df_cells['page_label'] = df_cells['page_number']
    
    print(f"   Found {len(df_cells)} cells in {df_cells['table_id'].nunique()} tables")
    
    # Pivot tables
    print(f"\n🔄 Pivoting tables...")
    df_pivoted = pivot_table_cells(df_cells)
    
    # Save output
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df_pivoted.to_csv(output_csv, index=False, encoding='utf-8-sig')
    
    print(f"✅ Saved pivoted tables to: {output_csv}")
    print(f"   Output has {len(df_pivoted)} rows")
    print(f"\n📊 Preview (first 20 rows):")
    print("=" * 100)
    
    # Show preview
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', 40)
    print(df_pivoted.head(20).to_string(index=False))
    print("=" * 100)


if __name__ == "__main__":
    main()
