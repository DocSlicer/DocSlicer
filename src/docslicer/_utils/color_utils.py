import re
from typing import Any, Sequence, Tuple, Optional

import numpy as np
import pandas as pd
from scipy.spatial import KDTree
import matplotlib.colors as mcolors


# =========================
# Color Conversion for PDF Extractors
# =========================

def pdf_color_to_hex(color_value: Any) -> Optional[str]:
    """
    Convert PDF color values (from PyMuPDF or pdfplumber) to hex format.
    
    Handles multiple input formats from different PDF libraries:
      - Integer: RGB packed as single int (e.g., 16777215 = 0xFFFFFF) [PyMuPDF]
      - Tuple/List: (R, G, B) in 0-1 or 0-255 range [both libraries]
      - Tuple/List: (C, M, Y, K) in 0-1 or 0-100 range [CMYK]
      - Hex string: '#RRGGBB' (pass-through)
      - Grayscale: single float/int
      - None or invalid: returns None
    
    Returns:
      Hex string like '#rrggbb' or None for invalid/missing colors
      
    Examples:
        >>> pdf_color_to_hex(16777215)
        '#ffffff'
        >>> pdf_color_to_hex((1.0, 0.0, 0.0))
        '#ff0000'
        >>> pdf_color_to_hex((255, 0, 0))
        '#ff0000'
        >>> pdf_color_to_hex(0.5)
        '#808080'
        >>> pdf_color_to_hex(None)
        None
    """
    if color_value is None:
        return None
    
    try:
        # Integer (PyMuPDF's typical format)
        if isinstance(color_value, (int, np.integer)):
            # Handle negative values or out-of-range
            if color_value < 0 or color_value > 0xFFFFFF:
                return None
            
            # Extract RGB from packed integer
            r = (color_value >> 16) & 0xFF
            g = (color_value >> 8) & 0xFF
            b = color_value & 0xFF
            
            # Convert to [0,1] range and then to hex
            rgb_01 = (r / 255.0, g / 255.0, b / 255.0)
            return mcolors.to_hex(rgb_01)
        
        # Tuple/List (RGB or CMYK or grayscale)
        elif isinstance(color_value, (tuple, list, np.ndarray)):
            arr = np.asarray(color_value, dtype=float).ravel()
            n = arr.size
            
            if n == 0:
                return None
            
            # Grayscale
            if n == 1:
                v = arr[0]
                if v > 1.0:
                    v = v / 255.0
                v = float(np.clip(v, 0.0, 1.0))
                return mcolors.to_hex((v, v, v))
            
            # RGB
            if n == 3:
                if arr.max() > 1.0:
                    arr = arr / 255.0
                arr = np.clip(arr, 0.0, 1.0)
                return mcolors.to_hex(tuple(arr))
            
            # CMYK
            if n == 4:
                if arr.max() > 1.0:
                    arr = arr / 100.0
                c, m, y, k = np.clip(arr, 0.0, 1.0)
                r = (1.0 - c) * (1.0 - k)
                g = (1.0 - m) * (1.0 - k)
                b = (1.0 - y) * (1.0 - k)
                return mcolors.to_hex((r, g, b))
            
            return None
        
        # String (hex or color name)
        elif isinstance(color_value, str):
            s = color_value.strip()
            if not s:
                return None
            
            # Try matplotlib parsing (handles hex, names, rgb(), etc.)
            try:
                rgb = mcolors.to_rgb(s)
                return mcolors.to_hex(rgb)
            except ValueError:
                return None
        
        # Float (grayscale)
        elif isinstance(color_value, (float, np.floating)):
            if np.isnan(color_value):
                return None
            v = float(color_value)
            if v > 1.0:
                v = v / 255.0
            v = np.clip(v, 0.0, 1.0)
            return mcolors.to_hex((v, v, v))
        
        return None
    
    except Exception:
        # Any conversion error -> return None
        return None


# =========================
# CSS color library (built once)
# =========================

_CSS_NAMED_COLORS = mcolors.CSS4_COLORS  # ~140 colors
_CSS_NAMES = np.array(list(_CSS_NAMED_COLORS.keys()))
_CSS_HEX = np.array(list(_CSS_NAMED_COLORS.values()))
_CSS_RGB = np.array([mcolors.to_rgb(hex_code) for hex_code in _CSS_HEX], dtype=float)
_CSS_TREE = KDTree(_CSS_RGB)


# =========================
# Helpers
# =========================

def _nums_to_rgb01(nums: Sequence[float]) -> Tuple[float, float, float]:
    """
    Convert 1/3/4 numeric values into RGB in [0,1]:
      - 1 value  -> grayscale
      - 3 values -> RGB (0-1 or 0-255)
      - 4 values -> CMYK (0-1 or 0-100)
    """
    arr = np.asarray(nums, dtype=float).ravel()
    n = arr.size

    if n == 0:
        return (np.nan, np.nan, np.nan)

    # Grayscale
    if n == 1:
        v = arr[0]
        if v > 1.0:
            v = v / 255.0
        v = float(np.clip(v, 0.0, 1.0))
        return (v, v, v)

    # RGB
    if n == 3:
        if arr.max() > 1.0:
            arr = arr / 255.0
        arr = np.clip(arr, 0.0, 1.0)
        return tuple(float(x) for x in arr)

    # CMYK
    if n == 4:
        # treat as 0-1 or 0-100
        if arr.max() > 1.0:
            arr = arr / 100.0
        c, m, y, k = np.clip(arr, 0.0, 1.0)
        r = (1.0 - c) * (1.0 - k)
        g = (1.0 - m) * (1.0 - k)
        b = (1.0 - y) * (1.0 - k)
        return (float(r), float(g), float(b))

    # Fallback: invalid length
    return (np.nan, np.nan, np.nan)


_NUM_RE = re.compile(r"[-+]?\d*\.?\d+")


def _parse_color_to_rgb01(val: Any) -> Tuple[float, float, float]:
    """
    Convert many possible formats to RGB in [0,1]:

    Supported:
      - Named CSS colors: 'red', 'LightGray', ...
      - Hex: '#RRGGBB', '#RGB'
      - 'rgb(...)', 'rgba(...)'
      - 'cmyk(...)'
      - 'R, G, B' or 'R G B'
      - single grayscale value: '128', 0.5, (0.5,), ...
      - tuples/lists/np.arrays of length 1 / 3 / 4 (grayscale / RGB / CMYK)
    """
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return (np.nan, np.nan, np.nan)

    # Tuples / lists / arrays
    if isinstance(val, (tuple, list, np.ndarray)):
        return _nums_to_rgb01(val)

    # Strings
    if isinstance(val, str):
        s = val.strip()
        if not s:
            return (np.nan, np.nan, np.nan)

        # Try direct matplotlib parsing first (handles names, hex, rgb(), etc.)
        try:
            r, g, b = mcolors.to_rgb(s)
            return (float(r), float(g), float(b))
        except ValueError:
            pass

        # Fallback: extract all numbers and interpret
        nums = [float(x) for x in _NUM_RE.findall(s)]
        return _nums_to_rgb01(nums)

    # Numbers (grayscale)
    if isinstance(val, (int, float, np.integer, np.floating)):
        return _nums_to_rgb01([val])

    # Unknown type
    return (np.nan, np.nan, np.nan)


# =========================
# Public API
# =========================

def add_color_columns(df: pd.DataFrame, column_name: str, *, prefix: str | None = None) -> pd.DataFrame:
    """
    For a given color column (rgb/greyscale/cmyk/mixed), add:

      - `<prefix>_hex` : hex string for the original color (e.g. '#aabbcc')
      - `<prefix>_css` : closest CSS4 color name by Euclidean distance in RGB

    The function:
      1) Detects format and converts to RGB in [0,1]
      2) Converts RGB -> hex
      3) Uses a KDTree over the 140 CSS colors for fast nearest-neighbor lookup

    Returns the same DataFrame with 2 new columns.
    """
    if prefix is None:
        prefix = column_name

    col = df[column_name]

    # 1) Convert to RGB (0-1) for every row
    rgb_list = [ _parse_color_to_rgb01(v) for v in col.values ]
    rgb_arr = np.array(rgb_list, dtype=float)   # shape (n, 3)

    # 2) Build mask for valid colors
    valid_mask = ~np.isnan(rgb_arr).any(axis=1)

    # Prepare outputs
    n = len(df)
    out_hex = np.full(n, np.nan, dtype=object)
    out_css_name = np.full(n, None, dtype=object)

    # 3) Hex for all valid colors
    if valid_mask.any():
        valid_rgb = rgb_arr[valid_mask]
        # hex for each row
        out_hex[valid_mask] = [mcolors.to_hex(tuple(row)) for row in valid_rgb]

        # 4) Nearest CSS color (vectorized KDTree lookup)
        distances, indices = _CSS_TREE.query(valid_rgb, k=1, workers=-1)
        out_css_name[valid_mask] = _CSS_NAMES[indices]

    # 5) Attach to df
    df[f"{prefix}_hex"] = out_hex
    df[f"{prefix}_css"] = out_css_name

    return df


# =========================
# Usage example
# =========================
if __name__ == "__main__":
    # Example df
    data = {
        "non_stroking_color": [
            "rgb(128, 0, 128)",
            "#ff0000",
            (0.2, 0.8, 0.1),
            (0, 0, 0, 0.5),        # CMYK
            200,                   # grayscale 0-255
            "0.3, 0.3, 0.3",       # custom comma-separated
        ],
        "stroking_color": [
            "blue",
            (255, 255, 0),
            (0.1,),                # grayscale 0-1
            "cmyk(50, 0, 0, 0)",
            None,
            "rgb(10, 200, 220)",
        ],
    }
    df_example = pd.DataFrame(data)

    df_example = add_color_columns(df_example, "non_stroking_color")
    df_example = add_color_columns(df_example, "stroking_color")

    print(df_example)
