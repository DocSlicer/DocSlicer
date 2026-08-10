"""Chart extraction tests — PPTX charts, per-point DataFrame, navigation, export.

financial_review.pptx embeds native Office charts (bar/line/pie …). These lock
the Chart / ChartPoint contract: structured series data, per-point flattening,
hierarchy/page navigation, CSV export, and dict round-trip.
"""

import json

import pytest
from docslicer import Chart, ChartPoint


# ── Presence & shape ──────────────────────────────────────────────────────────

def test_pptx_has_charts(pptx_result):
    assert len(pptx_result.charts) > 0, "financial_review.pptx should have charts"
    for chart in pptx_result.charts:
        assert isinstance(chart, Chart)
        assert chart.id
        assert chart.chart_type
        assert chart.page_number > 0
        assert isinstance(chart.markdown, str) and chart.markdown
        assert isinstance(chart.points, list) and chart.points
        assert all(isinstance(p, ChartPoint) for p in chart.points)


def test_chart_has_numeric_series(pptx_result):
    """At least one chart carries a named series and a plotted numeric value."""
    chart = next(
        (c for c in pptx_result.charts if c.series_names and any(p.value is not None for p in c.points)),
        None,
    )
    assert chart is not None, "expected at least one chart with a numeric series"
    assert all(isinstance(name, str) for name in chart.series_names)


def test_pdf_has_no_charts(pdf_result):
    """financial_report.pdf has no embedded charts — the list is empty, not missing."""
    assert pdf_result.charts == []


# ── Per-chart DataFrame ───────────────────────────────────────────────────────

def test_chart_to_dataframe(pptx_result):
    chart = pptx_result.charts[0]
    df = chart.to_dataframe()
    assert len(df) == len(chart.points)


# ── Flat per-point DataFrame ──────────────────────────────────────────────────

def test_charts_df_is_one_row_per_point(pptx_result):
    df = pptx_result.charts_df()
    total_points = sum(len(c.points) for c in pptx_result.charts)
    assert len(df) == total_points
    for col in ("chart_id", "chart_type", "series_name", "value"):
        assert col in df.columns


# ── Navigation ────────────────────────────────────────────────────────────────

def test_charts_by_page(pptx_result):
    chart = pptx_result.charts[0]
    on_page = pptx_result.charts_by_page(chart.page_number)
    assert chart.id in {c.id for c in on_page}
    assert all(c.page_number == chart.page_number for c in on_page)


def test_charts_under_heading(pptx_result):
    node = next(
        (n for n in pptx_result.hierarchy.flatten() if pptx_result.charts_under(n)),
        None,
    )
    if node is None:
        pytest.skip("no hierarchy node with charts")
    charts = pptx_result.charts_under(node)
    assert len(charts) > 0
    assert all(isinstance(c, Chart) for c in charts)


# ── Export ────────────────────────────────────────────────────────────────────

def test_export_charts_csv(pptx_result, tmp_path):
    path = tmp_path / "charts.csv"
    pptx_result.export_charts_csv(path)
    assert path.exists()
    lines = path.read_text(encoding="utf-8").splitlines()
    # header + one row per datapoint
    assert len(lines) == 1 + sum(len(c.points) for c in pptx_result.charts)


# ── Dict round-trip ───────────────────────────────────────────────────────────

def test_chart_dict_roundtrip(pptx_result):
    chart = pptx_result.charts[0]
    restored = Chart.from_dict(json.loads(json.dumps(chart.to_dict())))
    assert restored.id == chart.id
    assert restored.chart_type == chart.chart_type
    assert restored.series_names == chart.series_names
    assert len(restored.points) == len(chart.points)
