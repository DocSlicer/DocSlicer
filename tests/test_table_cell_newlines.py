"""Table cell text must never leak a raw newline into a row-per-line format.

A cell whose content wrapped over several physical lines is stored with its
fragments joined by "\n" (pdf/step_14_table_builder._build_table_cells_df).
That newline is structural in markdown/melted output: it splits one
"| a | b |" row in two and corrupts every row after it.
"""

import pandas as pd
import pytest

from docslicer.shared.step_07_block_merger import (
    _format_table_jsonl,
    _format_table_markdown,
    _format_table_melted,
)


@pytest.fixture
def wrapped_cell_table():
    """2x2 table whose header and one data cell wrapped over two lines."""
    return pd.DataFrame(
        [
            {"row_start": 0, "col_start": 0, "rowspan": 1, "colspan": 1,
             "table_cell_role": "header", "text": "Metric"},
            {"row_start": 0, "col_start": 1, "rowspan": 1, "colspan": 1,
             "table_cell_role": "header", "text": "Total credit provided\nand capital raised"},
            {"row_start": 1, "col_start": 0, "rowspan": 1, "colspan": 1,
             "table_cell_role": "row_label", "text": "$3.3\ntrillion"},
            {"row_start": 1, "col_start": 1, "rowspan": 1, "colspan": 1,
             "table_cell_role": "data", "text": "$ 71,477\n$ 119,333"},
        ]
    )


# ── Formatters ────────────────────────────────────────────────────────────────

def test_markdown_row_count_matches_grid(wrapped_cell_table):
    """Two grid rows + one header separator — wrapped cells add no extra lines."""
    md = _format_table_markdown(wrapped_cell_table)
    assert len(md.split("\n")) == 3


def test_markdown_every_line_is_a_table_row(wrapped_cell_table):
    md = _format_table_markdown(wrapped_cell_table)
    assert all(line.startswith("|") for line in md.split("\n"))
    assert "<br>" in md, "wrapped cell should keep its break as <br>"


def test_melted_line_per_fact(wrapped_cell_table):
    """One data cell → exactly one melted line, with the newline collapsed."""
    melted = _format_table_melted(wrapped_cell_table)
    assert melted.split("\n") == [
        "$3.3 trillion | Total credit provided and capital raised | $ 71,477 $ 119,333"
    ]


def test_jsonl_has_no_newline_in_values(wrapped_cell_table):
    import json

    lines = _format_table_jsonl(wrapped_cell_table).split("\n")
    assert len(lines) == 1
    obj = json.loads(lines[0])
    assert not any("\n" in k or "\n" in v for k, v in obj.items())


# ── End-to-end invariant ──────────────────────────────────────────────────────

@pytest.mark.parametrize("fixture", ["pdf_result", "html_result", "docx_result"])
def test_no_table_block_has_a_broken_row(fixture, request):
    """Every non-blank line of a rendered table block is a markdown row."""
    result = request.getfixturevalue(fixture)
    for block in result.blocks:
        if block.type != "table":
            continue
        broken = [
            line for line in (block.text or "").split("\n")
            if line.strip() and not line.lstrip().startswith("|")
        ]
        assert not broken, f"block {block.id} has non-row lines: {broken[:2]}"


@pytest.mark.parametrize("fixture", ["pdf_result", "html_result", "docx_result"])
def test_table_markdown_row_count_matches_cell_rows(fixture, request):
    """Table.markdown has one line per grid row (+1 for the header separator)."""
    result = request.getfixturevalue(fixture)
    for table in result.tables:
        if not table.markdown or not table.cells:
            continue
        n_rows = len({c.row for c in table.cells})
        n_lines = len(table.markdown.split("\n"))
        assert n_lines <= n_rows + 1, f"table {table.id}: {n_lines} lines for {n_rows} rows"
