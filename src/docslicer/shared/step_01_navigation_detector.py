# d03_navigation_detector.py
from __future__ import annotations

import pandas as pd


def detect_navigation_blocks(lines_df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect and mark navigation blocks based on repeated internal links.
    
    Logic:
    1. Filter lines_df on link_type = internal
    2. Check if there's a 'text' that repeats 3 or more times
    3. Mark all those rows as block_type = navigation
    
    Args:
        lines_df: DataFrame containing document lines
        Required columns: None (function returns unchanged df if required columns missing)
        Optional columns:
            - link_type: type of link (internal, external, etc.) - required for detection
            - block_type: current role of the block
            - text: text content of the line
            - page_number: page the line is on; when present, a text is counted
              at most once per page (multiple occurrences on the same page
              count as 1)
            
    Returns:
        Modified DataFrame with navigation blocks marked (or unchanged if link_type column missing)
    """
    df = lines_df.copy()
    
    # If link_type column not present, return unchanged (document has no links)
    if 'link_type' not in df.columns:
        return df
    
    # Step 1: Filter on link_type = internal
    candidates_mask = df['link_type'] == 'internal'

    # Step 2: Check if there's a 'text' that repeats 3 or more times
    # Get the candidate rows
    candidates_df = df[candidates_mask]
    
    if not candidates_df.empty and 'text' in candidates_df.columns:
        if 'page_number' in candidates_df.columns:
            # Count distinct pages each text appears on, so a text repeated
            # multiple times on the same page only counts once.
            text_counts = (
                candidates_df.drop_duplicates(subset=['text', 'page_number'])['text']
                .value_counts()
            )
        else:
            # Count occurrences of each text value among candidates
            text_counts = candidates_df['text'].value_counts()

        # Find texts that appear 3 or more times
        repeated_texts = text_counts[text_counts >= 3].index.tolist()
        
        # Step 3: Mark all rows with those repeated texts as navigation
        if repeated_texts:
            # Ensure block_type column exists
            if 'block_type' not in df.columns:
                df['block_type'] = None
            
            navigation_mask = candidates_mask & df['text'].isin(repeated_texts)
            df.loc[navigation_mask, 'block_type'] = 'navigation'
    
    return df

