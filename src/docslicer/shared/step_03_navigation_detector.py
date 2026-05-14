# d03_navigation_detector.py
from __future__ import annotations

import pandas as pd


def detect_navigation_blocks(lines_df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect and mark navigation blocks based on repeated internal links.
    
    Logic:
    1. Filter lines_df on link_type = internal
    2. Filter out block_type = toc
    3. Check if there's a 'text' that repeats 3 or more times
    4. Mark all those rows as block_type = navigation
    
    Args:
        lines_df: DataFrame containing document lines
        Required columns: None (function returns unchanged df if required columns missing)
        Optional columns:
            - link_type: type of link (internal, external, etc.) - required for detection
            - block_type: current role of the block
            - text: text content of the line
            
    Returns:
        Modified DataFrame with navigation blocks marked (or unchanged if link_type column missing)
    """
    df = lines_df.copy()
    
    # If link_type column not present, return unchanged (document has no links)
    if 'link_type' not in df.columns:
        return df
    
    # Step 1: Filter on link_type = internal
    internal_links_mask = df['link_type'] == 'internal'
    
    # Step 2: Filter out block_type = toc (if block_type column exists)
    if 'block_type' in df.columns:
        not_toc_mask = df['block_type'] != 'toc'
    else:
        not_toc_mask = pd.Series([True] * len(df), index=df.index)
    
    # Combine filters
    candidates_mask = internal_links_mask & not_toc_mask
    
    # Step 3: Check if there's a 'text' that repeats 3 or more times
    # Get the candidate rows
    candidates_df = df[candidates_mask]
    
    if not candidates_df.empty and 'text' in candidates_df.columns:
        # Count occurrences of each text value among candidates
        text_counts = candidates_df['text'].value_counts()
        
        # Find texts that appear 3 or more times
        repeated_texts = text_counts[text_counts >= 3].index.tolist()
        
        # Step 4: Mark all rows with those repeated texts as navigation
        if repeated_texts:
            # Ensure block_type column exists
            if 'block_type' not in df.columns:
                df['block_type'] = None
            
            navigation_mask = candidates_mask & df['text'].isin(repeated_texts)
            df.loc[navigation_mask, 'block_type'] = 'navigation'
    
    return df

