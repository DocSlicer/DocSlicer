"""
DOCX chart point extractor.

Builds a datapoint-level dataframe from embedded DrawingML chart parts. Each row
represents one plotted point/slice/bubble, with chart and series metadata
repeated so chart-level summaries can be derived by grouping.

DOCX charts use the exact same ``c:`` (DrawingML chart) schema as PPTX charts,
so the XML cache-parsing helpers here mirror those in
``pptx/step_03_chart_point_builder.py``. The DOCX-specific part is discovery:
charts are found via the ``image_ref`` runs produced by the run extractor, which
already carry the DOCX-computed ``page_number`` and ``source_part`` for each
embedded drawing.
"""

from __future__ import annotations

import posixpath
from dataclasses import dataclass
from typing import Any

import pandas as pd
from lxml import etree

from .step_01_package_reader import DocxPackage


NS = {
    "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
    "c": "http://schemas.openxmlformats.org/drawingml/2006/chart",
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
}

A = f"{{{NS['a']}}}"
C = f"{{{NS['c']}}}"
R = f"{{{NS['r']}}}"

_CHART_TYPE_TAGS = {
    "areaChart",
    "area3DChart",
    "barChart",
    "bar3DChart",
    "bubbleChart",
    "doughnutChart",
    "lineChart",
    "line3DChart",
    "ofPieChart",
    "pieChart",
    "pie3DChart",
    "radarChart",
    "scatterChart",
    "surfaceChart",
    "surface3DChart",
}


@dataclass(frozen=True)
class _ChartRef:
    chart_id: int
    page_number: int | None
    source_part: str
    chart_part: str
    rel_id: str
    run_id: int | None
    paragraph_id: int | None
    order_index: int | None
    alt_text: str | None


def _resolve_part(source_part: str, target: str | None) -> str | None:
    if not target:
        return None
    if target.startswith("/"):
        return target.lstrip("/")
    return posixpath.normpath(posixpath.join(posixpath.dirname(source_part), target))


def _local_name(elem: etree._Element) -> str:
    return etree.QName(elem).localname


def _val(elem: etree._Element | None) -> str | None:
    return elem.get("val") if elem is not None else None


def _text_from_rich(elem: etree._Element | None) -> str | None:
    if elem is None:
        return None
    parts = [t.text or "" for t in elem.findall(f".//{A}t")]
    text = "".join(parts).strip()
    return text or None


def _cache_points(parent: etree._Element | None) -> dict[int, str]:
    if parent is None:
        return {}
    cache = parent.find(f".//{C}strCache")
    if cache is None:
        cache = parent.find(f".//{C}numCache")
    if cache is None:
        cache = parent.find(f".//{C}multiLvlStrCache")
    if cache is None:
        return {}

    points: dict[int, str] = {}
    for pt in cache.findall(f".//{C}pt"):
        idx_raw = pt.get("idx")
        v = pt.find(f"{C}v")
        if idx_raw is None or v is None:
            continue
        try:
            idx = int(idx_raw)
        except ValueError:
            continue
        points[idx] = v.text or ""
    return points


def _first_cache_value(parent: etree._Element | None) -> str | None:
    points = _cache_points(parent)
    if not points:
        return None
    return points[min(points)]


def _num(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _format_code(parent: etree._Element | None) -> str | None:
    if parent is None:
        return None
    elem = parent.find(f".//{C}formatCode")
    return elem.text if elem is not None else None


def _solid_color(sp_pr: etree._Element | None) -> str | None:
    if sp_pr is None:
        return None
    solid = sp_pr.find(f".//{A}solidFill")
    if solid is None:
        return None
    srgb = solid.find(f"{A}srgbClr")
    if srgb is not None:
        val = srgb.get("val")
        if val and len(val) == 6:
            return f"#{val.lower()}"
    scheme = solid.find(f"{A}schemeClr")
    if scheme is not None:
        val = scheme.get("val")
        return f"scheme:{val}" if val else None
    return None


def _series_color(ser: etree._Element) -> str | None:
    return _solid_color(ser.find(f"{C}spPr"))


def _point_colors(ser: etree._Element) -> dict[int, str]:
    colors: dict[int, str] = {}
    for dpt in ser.findall(f"{C}dPt"):
        idx_raw = _val(dpt.find(f"{C}idx"))
        try:
            idx = int(idx_raw) if idx_raw is not None else None
        except ValueError:
            idx = None
        color = _solid_color(dpt.find(f"{C}spPr"))
        if idx is not None and color:
            colors[idx] = color
    return colors


def _data_labels(ser: etree._Element) -> dict[int, str]:
    labels: dict[int, str] = {}
    for dlbl in ser.findall(f"{C}dLbls/{C}dLbl"):
        idx_raw = _val(dlbl.find(f"{C}idx"))
        try:
            idx = int(idx_raw) if idx_raw is not None else None
        except ValueError:
            idx = None
        text = _text_from_rich(dlbl.find(f"{C}tx/{C}rich")) or _first_cache_value(dlbl.find(f"{C}tx"))
        if idx is not None and text:
            labels[idx] = text
    return labels


def _chart_title(root: etree._Element) -> str | None:
    title = root.find(f".//{C}chart/{C}title")
    return _text_from_rich(title.find(f"{C}tx/{C}rich") if title is not None else None)


def _axis_titles(root: etree._Element) -> tuple[str | None, str | None]:
    x_title = None
    y_title = None
    for axis_tag, attr_name in ((f"{C}catAx", "x"), (f"{C}dateAx", "x"), (f"{C}valAx", "y")):
        for axis in root.findall(f".//{axis_tag}"):
            title = _text_from_rich(axis.find(f"{C}title/{C}tx/{C}rich"))
            if not title:
                continue
            if attr_name == "x" and x_title is None:
                x_title = title
            elif attr_name == "y" and y_title is None:
                y_title = title
    return x_title, y_title


def _chart_refs(package: DocxPackage, df_runs: pd.DataFrame) -> list[_ChartRef]:
    """Discover embedded charts from the ``image_ref`` runs that carry a chart
    relationship id, resolving each to its chart part and DOCX page number."""
    required = {"chart_rel_id", "chart_id"}
    if df_runs is None or df_runs.empty or not required.issubset(df_runs.columns):
        return []

    chart_runs = df_runs[df_runs["chart_rel_id"].notna() & df_runs["chart_id"].notna()].copy()
    if chart_runs.empty:
        return []
    if "order_index" in chart_runs.columns:
        chart_runs = chart_runs.sort_values("order_index")

    refs: list[_ChartRef] = []
    for _, row in chart_runs.iterrows():
        source_part = row.get("source_part")
        rel_id = row.get("chart_rel_id")
        if not source_part or not rel_id:
            continue
        rel = package.get_relationship(str(source_part), str(rel_id))
        if rel is None or rel.is_external:
            continue
        chart_part = _resolve_part(str(source_part), rel.target)
        if not chart_part:
            continue

        page_raw = row.get("page_number")
        refs.append(_ChartRef(
            chart_id=int(row["chart_id"]),
            page_number=int(page_raw) if pd.notna(page_raw) else None,
            source_part=str(source_part),
            chart_part=chart_part,
            rel_id=str(rel_id),
            run_id=int(row["run_id"]) if pd.notna(row.get("run_id")) else None,
            paragraph_id=int(row["paragraph_id"]) if pd.notna(row.get("paragraph_id")) else None,
            order_index=int(row["order_index"]) if pd.notna(row.get("order_index")) else None,
            alt_text=(str(row["text"]) if pd.notna(row.get("text")) and str(row["text"]).strip() else None),
        ))
    return refs


def _point_count(*point_maps: dict[int, Any]) -> list[int]:
    indices: set[int] = set()
    for point_map in point_maps:
        indices.update(point_map)
    return sorted(indices)


def _extract_series_rows(
    chart_ref: _ChartRef,
    root: etree._Element,
    chart_elem: etree._Element,
) -> list[dict[str, Any]]:
    chart_type = _local_name(chart_elem)
    grouping = _val(chart_elem.find(f"{C}grouping"))
    bar_dir = _val(chart_elem.find(f"{C}barDir"))
    is_stacked = grouping in {"stacked", "percentStacked"}
    is_percent_stacked = grouping == "percentStacked"
    title = _chart_title(root)
    axis_x_title, axis_y_title = _axis_titles(root)

    rows: list[dict[str, Any]] = []
    for ser_pos, ser in enumerate(chart_elem.findall(f"{C}ser")):
        series_index = int(_val(ser.find(f"{C}idx")) or ser_pos)
        series_order = int(_val(ser.find(f"{C}order")) or ser_pos)
        series_name = _text_from_rich(ser.find(f"{C}tx/{C}rich")) or _first_cache_value(ser.find(f"{C}tx"))
        ser_color = _series_color(ser)
        point_colors = _point_colors(ser)
        point_labels = _data_labels(ser)

        categories = _cache_points(ser.find(f"{C}cat"))
        values = _cache_points(ser.find(f"{C}val"))
        x_values = _cache_points(ser.find(f"{C}xVal"))
        y_values = _cache_points(ser.find(f"{C}yVal"))
        bubble_sizes = _cache_points(ser.find(f"{C}bubbleSize"))
        value_parent = ser.find(f"{C}val")
        if value_parent is None:
            value_parent = ser.find(f"{C}yVal")
        value_format = _format_code(value_parent)

        value_map = values or y_values
        indices = _point_count(categories, value_map, x_values, y_values, bubble_sizes, point_labels)
        total = sum((_num(value_map.get(i)) or 0.0) for i in indices)

        for point_index in indices:
            raw_value = value_map.get(point_index)
            numeric_value = _num(raw_value)
            percent = (
                numeric_value / total
                if numeric_value is not None and total not in (0.0, None) and chart_type in {"pieChart", "pie3DChart", "doughnutChart", "ofPieChart"}
                else None
            )
            label = point_labels.get(point_index)
            category = categories.get(point_index)
            x_raw = x_values.get(point_index)
            y_raw = y_values.get(point_index)
            rows.append({
                "chart_id": chart_ref.chart_id,
                "chart_part": chart_ref.chart_part,
                "page_number": chart_ref.page_number,
                "source_part": chart_ref.source_part,
                "run_id": chart_ref.run_id,
                "paragraph_id": chart_ref.paragraph_id,
                "order_index": chart_ref.order_index,
                "alt_text": chart_ref.alt_text,
                "chart_type": chart_type,
                "chart_title": title,
                "axis_x_title": axis_x_title,
                "axis_y_title": axis_y_title,
                "grouping": grouping,
                "bar_direction": bar_dir,
                "is_stacked": is_stacked,
                "is_percent_stacked": is_percent_stacked,
                "series_index": series_index,
                "series_order": series_order,
                "series_name": series_name,
                "point_index": point_index,
                "category": category,
                "label": label,
                "raw_value": raw_value,
                "value": numeric_value,
                "value_format": value_format,
                "x_value": _num(x_raw),
                "x_label": x_raw,
                "y_value": _num(y_raw),
                "y_label": y_raw,
                "bubble_size": _num(bubble_sizes.get(point_index)),
                "percent": percent,
                "color": point_colors.get(point_index) or ser_color,
                "series_color": ser_color,
                "point_color": point_colors.get(point_index),
                "rel_id": chart_ref.rel_id,
            })
    return rows


def build_chart_points(
    package: DocxPackage,
    df_runs: pd.DataFrame,
) -> pd.DataFrame:
    """
    Extract datapoint-level rows from embedded DOCX chart XML caches.

    Args:
        package: Parsed DOCX package.
        df_runs: Run dataframe from the run extractor. ``image_ref`` runs that
            reference a chart (``chart_rel_id`` populated) supply the chart list
            along with each chart's DOCX-computed ``page_number`` / ``source_part``.

    Returns:
        DataFrame with one row per plotted chart datapoint.
    """
    rows: list[dict[str, Any]] = []

    for chart_ref in _chart_refs(package, df_runs):
        root = package.get_xml(chart_ref.chart_part)
        if root is None:
            continue
        plot_area = root.find(f".//{C}plotArea")
        if plot_area is None:
            continue
        for child in plot_area:
            if _local_name(child) not in _CHART_TYPE_TAGS:
                continue
            rows.extend(_extract_series_rows(chart_ref, root, child))

    return pd.DataFrame(rows)


__all__ = ["build_chart_points"]
