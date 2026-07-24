# SPDX-FileCopyrightText: 2026 Market Framer Inc.
# SPDX-License-Identifier: AGPL-3.0-only
"""
Shared sub/superscript detection thresholds.

Script detection runs twice in the pipeline, at different granularities:

  - step_01_word_extractor: per *character*, while assembling words from the
    raw glyph stream (a size/baseline change mid-word starts a script word).
  - step_10_cell_builder: per *word*, within each assembled cell (a whole
    word that arrived separately from pdfium can still be a footnote ref).

Both passes must agree on what counts as "smaller" and "baseline-shifted",
so the thresholds live here rather than being mirrored in each step.

All SIZE thresholds are ratios of the reference (body-text) font size; the
Y factor is a fraction of that reference size applied to the baseline shift.
"""

SCRIPT_Y_FACTOR  = 0.20   # baseline must shift > 20% of ref font-size to be a candidate
SCRIPT_SIZE_MIN  = 0.40   # candidate must be ≥ 40% of ref size (excludes drop caps)
SCRIPT_SIZE_DOWN = 0.82   # candidate < 82% of ref size  → entering script
SCRIPT_SIZE_INV  = 1.22   # incoming char > 122% of word size → current word was script
SCRIPT_SIZE_UP   = 0.88   # in script word: incoming ≥ 88% of ref size → back to normal
