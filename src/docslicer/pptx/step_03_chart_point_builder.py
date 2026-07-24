# SPDX-FileCopyrightText: 2026 Market Framer Inc.
# SPDX-License-Identifier: AGPL-3.0-only
"""
PPTX chart point extractor.

Builds a datapoint-level dataframe from chart XML parts. Each row represents
one plotted point/slice/bubble, with chart and series metadata repeated so
chart-level summaries can be derived by grouping.
"""

from __future__ import annotations

import posixpath
from dataclasses import dataclass
from typing import Any

import pandas as pd
from lxml import etree

from .step_01_package_reader import PptxPackage


NS = {
    "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
    "c": "http://schemas.openxmlformats.org/drawingml/2006/chart",
    "p": "http://schemas.openxmlformats.org/presentationml/2006/main",
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
}

A = f"{{{NS['a']}}}"
C = f"{{{NS['c']}}}"
P = f"{{{NS['p']}}}"
R = f"{{{NS['r']}}}"

_REL_CHART = "http://schemas.openxmlformats.org/officeDocument/2006/relationships/chart"

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
    slide_number: int
    slide_index: int
    source_part: str
    chart_part: str
    rel_id: str
    shape_name: str | None


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


def _chart_refs(package: PptxPackage) -> list[_ChartRef]:
    refs: list[_ChartRef] = []
    chart_id = 0
    for slide in package.slides:
        root = package.get_xml(slide.part_name)
        if root is None:
            continue
        for frame in root.findall(f".//{P}graphicFrame"):
            chart = frame.find(f".//{C}chart")
            if chart is None:
                continue
            rel_id = chart.get(f"{R}id")
            rel = package.get_relationship(slide.part_name, rel_id)
            if rel is None or rel.rel_type != _REL_CHART or rel.is_external:
                continue
            chart_part = _resolve_part(slide.part_name, rel.target)
            if not chart_part:
                continue
            nv = frame.find(f"{P}nvGraphicFramePr")
            cnv_pr = nv.find(f"{P}cNvPr") if nv is not None else None
            shape_name = cnv_pr.get("name") if cnv_pr is not None else None
            chart_id += 1
            refs.append(_ChartRef(
                chart_id=chart_id,
                slide_number=slide.slide_number,
                slide_index=slide.slide_index,
                source_part=slide.part_name,
                chart_part=chart_part,
                rel_id=rel_id or "",
                shape_name=shape_name,
            ))
    return refs


def _chart_ref_geometry(df_runs: pd.DataFrame | None) -> dict[int, dict[str, Any]]:
    if df_runs is None or df_runs.empty or "run_type" not in df_runs.columns:
        return {}
    chart_refs = df_runs[df_runs["run_type"].eq("chart_ref")].copy()
    if chart_refs.empty:
        return {}
    if "chart_id" not in chart_refs.columns:
        return {}
    chart_refs = chart_refs[chart_refs["chart_id"].notna()]
    if chart_refs.empty:
        return {}

    geometry: dict[int, dict[str, Any]] = {}
    cols = [
        "shape_id", "x_left", "y_top", "x_right", "y_bottom", "width", "height",
        "shape_name",
    ]
    for _, row in chart_refs.iterrows():
        data = {col: row.get(col) for col in cols}
        geometry[int(row["chart_id"])] = data
    return geometry


def _point_count(*point_maps: dict[int, Any]) -> list[int]:
    indices: set[int] = set()
    for point_map in point_maps:
        indices.update(point_map)
    return sorted(indices)


def _extract_series_rows(
    chart_ref: _ChartRef,
    root: etree._Element,
    chart_elem: etree._Element,
    chart_order_on_slide: int,
    geom: dict[str, Any],
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
                "chart_order_on_slide": chart_order_on_slide,
                "page_number": chart_ref.slide_number,
                "slide_index": chart_ref.slide_index,
                "source_part": chart_ref.source_part,
                "shape_id": geom.get("shape_id"),
                "shape_name": geom.get("shape_name") or chart_ref.shape_name,
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
                "x_left": geom.get("x_left"),
                "y_top": geom.get("y_top"),
                "x_right": geom.get("x_right"),
                "y_bottom": geom.get("y_bottom"),
                "width": geom.get("width"),
                "height": geom.get("height"),
            })
    return rows


def extract_chart_points(
    package: PptxPackage,
    df_runs: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Extract datapoint-level rows from PPTX chart XML caches.

    Args:
        package: Parsed PPTX package.
        df_runs: Optional run dataframe. If provided, chart_ref rows are used to
            attach shape_id and shape-level geometry to each chart point.

    Returns:
        DataFrame with one row per plotted chart datapoint.
    """
    rows: list[dict[str, Any]] = []
    geometry = _chart_ref_geometry(df_runs)
    refs_by_slide: dict[int, int] = {}

    for chart_ref in _chart_refs(package):
        order_on_slide = refs_by_slide.get(chart_ref.slide_number, 0) + 1
        refs_by_slide[chart_ref.slide_number] = order_on_slide
        root = package.get_xml(chart_ref.chart_part)
        if root is None:
            continue
        geom = geometry.get(chart_ref.chart_id, {})
        plot_area = root.find(f".//{C}plotArea")
        if plot_area is None:
            continue
        for child in plot_area:
            if _local_name(child) not in _CHART_TYPE_TAGS:
                continue
            rows.extend(_extract_series_rows(chart_ref, root, child, order_on_slide, geom))

    return pd.DataFrame(rows)


__all__ = ["extract_chart_points"]
