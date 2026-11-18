"""Generate the paper figures: uv run python src/plot.py --help."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import math
import re
import sys
import tempfile
import warnings
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, NamedTuple
from xml.etree import ElementTree

import cairosvg
import click
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.figure import Figure
from matplotlib.font_manager import fontManager
from matplotlib.legend_handler import HandlerBase
from matplotlib.lines import Line2D
from matplotlib.patches import (
    Circle,
    FancyArrowPatch,
    FancyBboxPatch,
    Patch,
    Rectangle,
    Wedge,
)
from matplotlib.ticker import (
    FuncFormatter,
    LogLocator,
    MaxNLocator,
    ScalarFormatter,
    StrMethodFormatter,
)
from matplotlib.transforms import Bbox
from scipy.optimize import curve_fit

import local_data
import metrics as _metrics

# Typography

FIGURE_TYPOGRAPHY_VERSION = "lato-v1"

FIGURE_FONT_FAMILY = "Lato, sans-serif"

MATPLOTLIB_BASE_SIZE = 10.0

FIGURE_GLOBAL_RENDER_SCALE = 1.2

MATPLOTLIB_TITLE_SIZE_RATIO = 1.2

MATPLOTLIB_SMALL_SIZE_RATIO = 5.0 / 6.0

MATPLOTLIB_NOTE_SIZE = MATPLOTLIB_BASE_SIZE * MATPLOTLIB_SMALL_SIZE_RATIO

_SVG_ROOT_RE = re.compile(r"<svg\b[^>]*>", re.IGNORECASE | re.DOTALL)

_CSS_FONT_FAMILY_RE = re.compile(
    r"(?P<prefix>\bfont-family\s*:\s*)(?P<value>[^;\"]+)",
    re.IGNORECASE,
)

_ATTRIBUTE_FONT_FAMILY_RE = re.compile(
    r"(?P<prefix>\bfont-family\s*=\s*)(?P<quote>[\"'])"
    r"(?P<value>.*?)(?P=quote)",
    re.IGNORECASE,
)

_TYPOGRAPHY_MARKER_RE = re.compile(
    r"\bdata-figure-typography\s*=\s*[\"'][^\"']*[\"']", re.IGNORECASE
)

_RENDER_WIDTH_MARKER_RE = re.compile(
    r"\bdata-figure-render-width\s*=\s*[\"'][^\"']*[\"']", re.IGNORECASE
)

_RENDER_SCALE_MARKER_RE = re.compile(
    r"\bdata-figure-render-scale\s*=\s*[\"'][^\"']*[\"']", re.IGNORECASE
)


def matplotlib_typography_rcparams() -> dict[str, float]:
    """Return Matplotlib's fixed default semantic font hierarchy."""
    title_size = MATPLOTLIB_BASE_SIZE * MATPLOTLIB_TITLE_SIZE_RATIO
    return {
        "font.size": MATPLOTLIB_BASE_SIZE,
        "axes.titlesize": title_size,
        "figure.titlesize": title_size,
        "axes.labelsize": MATPLOTLIB_BASE_SIZE,
        "xtick.labelsize": MATPLOTLIB_BASE_SIZE,
        "ytick.labelsize": MATPLOTLIB_BASE_SIZE,
        "legend.fontsize": MATPLOTLIB_BASE_SIZE,
        "legend.title_fontsize": MATPLOTLIB_BASE_SIZE,
    }


def _view_box_width(svg: str, path: Path | None = None) -> float:
    root = ElementTree.fromstring(svg)
    assert root.tag.rsplit("}", maxsplit=1)[-1] == "svg", (
        f"not an SVG root: {path or '<string>'}"
    )
    view_box = root.get("viewBox")
    assert view_box is not None, f"referenced SVG lacks viewBox: {path or '<string>'}"
    values = view_box.replace(",", " ").split()
    assert len(values) == 4, f"invalid SVG viewBox={view_box!r}: {path or '<string>'}"
    width = float(values[2])
    assert math.isfinite(width) and width > 0, (
        f"non-positive SVG viewBox={view_box!r}: {path or '<string>'}"
    )
    return width


def _format_number(value: float) -> str:
    rendered = f"{value:.3f}".rstrip("0").rstrip(".")
    return rendered or "0"


def _normalize_font_families(svg: str) -> str:
    def replace_css(match: re.Match[str]) -> str:
        family = match.group("value").lower()
        if "monospace" in family or "emoji" in family:
            return match.group(0)
        return f"{match.group('prefix')}'Lato', sans-serif"

    def replace_attribute(match: re.Match[str]) -> str:
        family = match.group("value").lower()
        if "monospace" in family or "emoji" in family:
            return match.group(0)
        quote = match.group("quote")
        return f"{match.group('prefix')}{quote}{FIGURE_FONT_FAMILY}{quote}"

    svg = _CSS_FONT_FAMILY_RE.sub(replace_css, svg)
    return _ATTRIBUTE_FONT_FAMILY_RE.sub(replace_attribute, svg)


def _mark_root(
    svg: str, render_width_px: float, render_scale: float | None = None
) -> str:
    match = _SVG_ROOT_RE.search(svg)
    assert match is not None, "SVG has no root element"
    root = match.group(0)
    if _TYPOGRAPHY_MARKER_RE.search(root):
        root = _TYPOGRAPHY_MARKER_RE.sub(
            f'data-figure-typography="{FIGURE_TYPOGRAPHY_VERSION}"', root
        )
    else:
        root = root[:-1] + (f' data-figure-typography="{FIGURE_TYPOGRAPHY_VERSION}">')
    render_width_marker = (
        f'data-figure-render-width="{_format_number(render_width_px)}"'
    )
    if _RENDER_WIDTH_MARKER_RE.search(root):
        root = _RENDER_WIDTH_MARKER_RE.sub(render_width_marker, root)
    else:
        root = root[:-1] + f" {render_width_marker}>"
    if render_scale is not None:
        render_scale_marker = (
            f'data-figure-render-scale="{_format_number(render_scale)}"'
        )
        if _RENDER_SCALE_MARKER_RE.search(root):
            root = _RENDER_SCALE_MARKER_RE.sub(render_scale_marker, root)
        else:
            root = root[:-1] + f" {render_scale_marker}>"
    if not _ATTRIBUTE_FONT_FAMILY_RE.search(root):
        root = root[:-1] + f' font-family="{FIGURE_FONT_FAMILY}">'
    return svg[: match.start()] + root + svg[match.end() :]


def normalize_matplotlib_svg_typography(svg: str) -> str:
    """Apply the one shared whole-figure scale to a Matplotlib SVG."""
    render_width_px = _view_box_width(svg) * FIGURE_GLOBAL_RENDER_SCALE
    normalized = _normalize_font_families(svg)
    return _mark_root(normalized, render_width_px, FIGURE_GLOBAL_RENDER_SCALE)


def normalize_matplotlib_svg_typography_file(path: Path) -> bool:
    """Normalize one Matplotlib SVG in place and report whether it changed."""
    original = path.read_text()
    normalized = normalize_matplotlib_svg_typography(original)
    if normalized == original:
        return False
    path.write_text(normalized)
    return True


# Style

FIGURE_WIDTH = 12.0

LAYOUT_SCALE = 0.74


def figsize(w: float, h: float) -> tuple[float, float]:
    """Return the historical recipe's initial canvas geometry in inches.

    Plot styling is unaffected; fonts, lines, and markers inherit Matplotlib
    defaults and the saved SVG receives the one shared whole-figure scale.
    """
    return (w * LAYOUT_SCALE, h * LAYOUT_SCALE)


def set_square_subplot_height(fig, axes, displayed_height_px: float) -> None:
    """Resize a canvas so square data axes render at ``displayed_height_px``.

    Width stays tied to height. This changes only plot geometry: text and data
    glyphs retain Matplotlib's shared defaults and are scaled later by the one
    global whole-SVG factor.
    """
    assert displayed_height_px > 0
    axes = tuple(axes)
    assert axes
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    extents = [axis.get_window_extent(renderer=renderer) for axis in axes]
    for extent in extents:
        assert np.isclose(extent.width, extent.height, rtol=1e-6), extent
    points_per_display_pixel = 72.0 / fig.dpi
    heights_points = [extent.height * points_per_display_pixel for extent in extents]
    assert max(heights_points) - min(heights_points) < 0.01, heights_points
    current_height_px = heights_points[0] * FIGURE_GLOBAL_RENDER_SCALE
    resize_factor = displayed_height_px / current_height_px

    # Scale the subplot block while preserving its outer gutters in physical
    # points. Fonts and legends therefore retain their standard size instead of
    # being squeezed together when a recipe requests a smaller data area.
    bounds_points = [
        (
            extent.x0 * points_per_display_pixel,
            extent.y0 * points_per_display_pixel,
            extent.width * points_per_display_pixel,
            extent.height * points_per_display_pixel,
        )
        for extent in extents
    ]
    block_x0 = min(x for x, _y, _width, _height in bounds_points)
    block_y0 = min(y for _x, y, _width, _height in bounds_points)
    block_x1 = max(x + width for x, _y, width, _height in bounds_points)
    block_y1 = max(y + height for _x, y, _width, height in bounds_points)
    figure_width_points = fig.get_figwidth() * 72.0
    figure_height_points = fig.get_figheight() * 72.0
    right_gutter = figure_width_points - block_x1
    top_gutter = figure_height_points - block_y1
    new_width_points = block_x0 + (block_x1 - block_x0) * resize_factor + right_gutter
    new_height_points = block_y0 + (block_y1 - block_y0) * resize_factor + top_gutter
    fig.set_size_inches(new_width_points / 72.0, new_height_points / 72.0, forward=True)
    for axis, (x, y, width, height) in zip(axes, bounds_points, strict=True):
        resized_x = block_x0 + (x - block_x0) * resize_factor
        resized_y = block_y0 + (y - block_y0) * resize_factor
        axis.set_position(
            (
                resized_x / new_width_points,
                resized_y / new_height_points,
                width * resize_factor / new_width_points,
                height * resize_factor / new_height_points,
            )
        )
    fig.canvas.draw()
    resized = axes[0].get_window_extent(renderer=fig.canvas.get_renderer())
    resized_height_px = resized.height * 72.0 / fig.dpi * FIGURE_GLOBAL_RENDER_SCALE
    assert np.isclose(resized_height_px, displayed_height_px, atol=0.1), (
        resized_height_px,
        displayed_height_px,
    )


def pack_horizontal_axes(fig, axes, gap_font_sizes: float = 1.0) -> None:
    """Pack axes left-to-right using their rendered labels as boundaries."""
    assert gap_font_sizes >= 0
    axes = tuple(axes)
    assert len(axes) > 1
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    gap_pixels = mpl.rcParams["font.size"] * fig.dpi / 72.0 * gap_font_sizes
    previous_right = axes[0].get_tightbbox(renderer=renderer).x1
    for axis in axes[1:]:
        tight_bbox = axis.get_tightbbox(renderer=renderer)
        shift_pixels = previous_right + gap_pixels - tight_bbox.x0
        position = axis.get_position()
        axis.set_position(
            (
                position.x0 + shift_pixels / fig.bbox.width,
                position.y0,
                position.width,
                position.height,
            )
        )
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        previous_right = axis.get_tightbbox(renderer=renderer).x1


def pack_horizontal_axis_columns(fig, axes, gap_font_sizes: float = 1.0) -> None:
    """Pack a grid's columns while preserving alignment between its rows."""
    assert gap_font_sizes >= 0
    rows = tuple(tuple(row) for row in axes)
    assert rows
    column_count = len(rows[0])
    assert column_count > 1
    assert all(len(row) == column_count for row in rows)

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    gap_pixels = mpl.rcParams["font.size"] * fig.dpi / 72.0 * gap_font_sizes
    previous_right = max(row[0].get_tightbbox(renderer=renderer).x1 for row in rows)
    for column_index in range(1, column_count):
        column = tuple(row[column_index] for row in rows)
        current_left = min(axis.get_tightbbox(renderer=renderer).x0 for axis in column)
        shift_pixels = previous_right + gap_pixels - current_left
        for axis in column:
            position = axis.get_position()
            axis.set_position(
                (
                    position.x0 + shift_pixels / fig.bbox.width,
                    position.y0,
                    position.width,
                    position.height,
                )
            )
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        previous_right = max(
            axis.get_tightbbox(renderer=renderer).x1 for axis in column
        )


def center_axes_block(fig, axes, reference_axes) -> None:
    """Center one axes block on the rendered content of another."""
    axes = tuple(axes)
    reference_axes = tuple(reference_axes)
    assert axes
    assert reference_axes
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    block = Bbox.union([axis.get_tightbbox(renderer=renderer) for axis in axes])
    reference = Bbox.union(
        [axis.get_tightbbox(renderer=renderer) for axis in reference_axes]
    )
    shift_pixels = (reference.x0 + reference.x1) / 2.0 - (block.x0 + block.x1) / 2.0
    for axis in axes:
        position = axis.get_position()
        axis.set_position(
            (
                position.x0 + shift_pixels / fig.bbox.width,
                position.y0,
                position.width,
                position.height,
            )
        )


PARAM_CMAP = LinearSegmentedColormap.from_list(
    "earth", ["#23403f", "#3f6b5e", "#7e8a45", "#b3823f", "#9c4f2f"]
)

HEATMAP_CMAP = LinearSegmentedColormap.from_list(
    "earth_seq", ["#f4ecda", "#e2c089", "#c79150", "#9c6234", "#5e3418"]
)

SCORING_PROTOCOL_COLORS = {
    "llr": "#465c6e",
    "probe": "#9c4f2f",
}

SCORING_PROTOCOL_LINESTYLES = {
    "llr": "--",
    "probe": "-",
}

MODEL_FAMILY_MARKERS = {
    "marindna": "o",
    "evo2": "s",
}

COMPARISON_ERRORBAR_ALPHA = 1.0

EARTH_QUAL = ["#9c4f2f", "#2f6f63", "#c0883a", "#465c6e", "#6f7d3f", "#8a5170"]

LEGEND_Y = 0.10

X_LABEL_PAD = 0

LEGEND_KW = dict(
    frameon=False,
    handletextpad=0.3,
    columnspacing=2.2,
    borderpad=0.4,
)


def palette(param_counts: list[int]) -> dict[int, tuple]:
    cmap = PARAM_CMAP
    if len(param_counts) == 1:
        return {param_counts[0]: cmap(0.5)}
    return {
        p: cmap(0.15 + 0.7 * i / (len(param_counts) - 1))
        for i, p in enumerate(param_counts)
    }


def params_label(num_params: int | float) -> str:
    n = int(num_params)
    if n >= 1_000_000_000:
        return f"{round(n / 1e9)}B"
    return f"{round(n / 1e6)}M"


def shape_legend_handles(include_reference: bool = True):
    """Proxy artists for the marker-shape legend (no axes side-effects)."""
    common = dict(
        color="w",
        markerfacecolor="lightgray",
        markeredgecolor="k",
        linestyle="",
    )
    sweep = Line2D([0], [0], marker="o", **common)
    optimal = Line2D([0], [0], marker="s", **common)
    handles = [sweep, optimal]
    labels = ["Sweep", "Optimal (predicted)"]
    if include_reference:
        handles.append(Line2D([0], [0], marker="D", **common))
        labels.append("Control (reference)")
    return handles, labels


def params_legend_handles(palette: dict, params: list[int]):
    """Proxy artists for the per-scale params legend (square markers, scale color)."""
    sorted_params = sorted(params)
    handles = [
        Line2D(
            [0],
            [0],
            marker="s",
            color="w",
            markerfacecolor=palette[p],
            markeredgecolor="k",
            linestyle="",
        )
        for p in sorted_params
    ]
    labels = [params_label(p) for p in sorted_params]
    return handles, labels


def attach_params_legend_below(
    fig,
    palette: dict,
    params: list[int],
    *,
    width_scale: float = 1.0,
    handlelength: float | None = None,
) -> None:
    """Single horizontal `model params` legend, centered just below the x-axis.

    `width_scale` shrinks/expands the inter-pair gap (columnspacing) — smaller
    values produce a more compact legend. `handlelength` (if given) tightens the
    marker-to-label gap for an even more compact strip.
    """
    p_handles, p_labels = params_legend_handles(palette, params)
    kw = {**LEGEND_KW, "columnspacing": LEGEND_KW["columnspacing"] * width_scale}
    if handlelength is not None:
        kw["handlelength"] = handlelength
    fig.legend(
        p_handles,
        p_labels,
        ncol=len(p_handles),
        title="Model parameters",
        loc="upper center",
        bbox_to_anchor=(0.5, LEGEND_Y),
        **kw,
    )


TWO_LEGEND_KW = {**LEGEND_KW, "columnspacing": 0.6, "handlelength": 0.9}


def attach_legends_below(
    fig,
    palette: dict,
    params: list[int],
    include_reference: bool = True,
    legend_y: float = 0.09,
    gap: float = 0.03,
) -> None:
    """Two horizontal figure-level legends (`model params` + `run type`) placed
    as a single block, centered on x=0.5, just below the axes.

    The pair is centered by measuring each legend's rendered width and seating
    them symmetrically with a fixed `gap` — so the block stays centered no matter
    how many entries each legend has (e.g. with or without the control marker).
    Centering is on the *plotted content* (the axes block incl. their labels),
    not the raw figure: figures save with ``bbox_inches='tight'``, which crops to
    that content, so its center — not x=0.5 — is what reads as centered.
    """
    p_handles, p_labels = params_legend_handles(palette, params)
    s_handles, s_labels = shape_legend_handles(include_reference=include_reference)

    # Seat both at the left provisionally; positions are fixed up after measuring.
    leg_params = fig.legend(
        p_handles,
        p_labels,
        ncol=len(p_handles),
        title="Model parameters",
        loc="upper left",
        bbox_to_anchor=(0.0, legend_y),
        **TWO_LEGEND_KW,
    )
    fig.add_artist(leg_params)
    leg_shape = fig.legend(
        s_handles,
        s_labels,
        ncol=len(s_handles),
        title="Run type",
        loc="upper left",
        bbox_to_anchor=(0.0, legend_y),
        **TWO_LEGEND_KW,
    )

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    fw = fig.bbox.width
    # Visual center of the plotted content (axes + their tick/axis labels).
    content = Bbox.union(
        [a.get_tightbbox(renderer) for a in fig.axes if a.get_visible()]
    )
    center = (content.x0 + content.x1) / 2.0 / fw
    w1 = leg_params.get_window_extent().width / fw
    w2 = leg_shape.get_window_extent().width / fw
    x0 = center - (w1 + gap + w2) / 2.0
    leg_params.set_bbox_to_anchor((x0, legend_y), transform=fig.transFigure)
    leg_shape.set_bbox_to_anchor((x0 + w1 + gap, legend_y), transform=fig.transFigure)


def attach_stacked_legends_right(
    fig,
    ax,
    palette: dict,
    params: list[int],
    *,
    include_reference: bool = True,
) -> None:
    """Stack two one-column legend groups to the right of one subplot."""
    param_handles, param_labels = params_legend_handles(palette, params)
    param_legend = fig.legend(
        param_handles,
        param_labels,
        ncol=1,
        title="Model parameters",
        loc="upper left",
        bbox_to_anchor=(1.04, 1.08),
        bbox_transform=ax.transAxes,
        alignment="left",
        **TWO_LEGEND_KW,
    )
    fig.add_artist(param_legend)
    run_handles, run_labels = shape_legend_handles(include_reference=include_reference)
    fig.legend(
        run_handles,
        run_labels,
        ncol=1,
        title="Run type",
        loc="lower left",
        bbox_to_anchor=(1.04, -0.10),
        bbox_transform=ax.transAxes,
        alignment="left",
        **TWO_LEGEND_KW,
    )


def set_plain_decimal_yticks(ax) -> None:
    """Force y-axis tick labels to plain decimals (no scientific notation).

    Works for both linear and log y-scales. On log scales matplotlib's default
    LogFormatter renders values like ``6×10⁻¹``; in this figure's loss range
    (~0.5–1.5) plain decimals are more readable.
    """
    fmt = ScalarFormatter()
    fmt.set_scientific(False)
    fmt.set_useOffset(False)
    ax.yaxis.set_major_formatter(fmt)
    ax.yaxis.set_minor_formatter(fmt)


# Sweep

CONTROL_ROLES = ("positive-control", "negative-control")


class _AutoGridLocator(MaxNLocator):
    """Choose a readable subset of integer positions on an equal-spaced grid."""

    def __init__(self, grid_size: int) -> None:
        assert grid_size > 1
        self._grid_size = grid_size
        super().__init__(nbins="auto", integer=True, min_n_ticks=min(3, grid_size))

    def tick_values(self, vmin: float, vmax: float):
        ticks = super().tick_values(
            max(vmin, 0.0), min(vmax, float(self._grid_size - 1))
        )
        return [
            tick
            for tick in ticks
            if 0 <= tick < self._grid_size and math.isclose(tick, round(tick))
        ]


def _interpolate_x(
    value: float, grid_values: list[float], grid_xs: list[int], log_scale: bool
) -> float:
    """Map a hparam value to an evenly-spaced grid x (log/linear). Extrapolates."""
    if log_scale:
        t = math.log(value)
        gt = [math.log(g) for g in grid_values]
    else:
        t, gt = value, list(grid_values)
    if t <= gt[0]:
        slope = (grid_xs[1] - grid_xs[0]) / (gt[1] - gt[0])
        return grid_xs[0] + slope * (t - gt[0])
    if t >= gt[-1]:
        slope = (grid_xs[-1] - grid_xs[-2]) / (gt[-1] - gt[-2])
        return grid_xs[-1] + slope * (t - gt[-1])
    for i in range(len(gt) - 1):
        if gt[i] <= t <= gt[i + 1]:
            frac = (t - gt[i]) / (gt[i + 1] - gt[i])
            return grid_xs[i] + frac * (grid_xs[i + 1] - grid_xs[i])
    return grid_xs[-1]


def plot_axis(
    ax,
    df: pd.DataFrame,
    *,
    axis_role: str,
    axis_field: str,
    axis_label: str,
    log_scale: bool,
    palette: dict,
    value_formatter=None,
    native_numeric_axis: bool = False,
    include_negative_control: bool = True,
    y_field: str = "eval_loss",
    y_label: str = "Loss",
) -> None:
    """Plot `y_field` vs `axis_field` for the relevant rows. One series per param scale.

    Includes:
      - sweep points for runs with role == axis_role  (circles)
      - per-scale positive-control                    (square, on grid center)
      - 1B-only negative-control                      (square, x-position interpolated)
    """
    sub = (
        df[df["role"].isin([axis_role, *CONTROL_ROLES])]
        .dropna(subset=[axis_field, y_field])
        .copy()
    )

    # Union x grid: every unique axis value seen in sweep + positive-control. Negative
    # control's value is interpolated on top of this grid (it's untransferred).
    grid_mask = sub["role"].isin([axis_role, "positive-control"])
    grid_values = sorted(sub.loc[grid_mask, axis_field].unique().tolist())
    grid_xs = list(range(len(grid_values)))
    grid_lookup = {v: i for i, v in enumerate(grid_values)}

    for params in sorted(sub["params"].unique()):
        scale = sub[sub["params"] == params]
        color = palette[int(params)]

        # Per-scale line: connect mean y at each axis value, ascending.
        line_pts = (
            scale[scale["role"].isin([axis_role, "positive-control"])]
            .groupby(axis_field, as_index=False)[y_field]
            .mean()
            .sort_values(axis_field)
        )
        if not line_pts.empty:
            line_xs = (
                line_pts[axis_field].to_numpy()
                if native_numeric_axis
                else [grid_lookup[v] for v in line_pts[axis_field]]
            )
            ax.plot(
                line_xs,
                line_pts[y_field].values,
                color=color,
                alpha=0.6,
                zorder=2,
            )

        # Sweep points: circles, raw individual runs.
        sweep = scale[scale["role"] == axis_role]
        if not sweep.empty:
            sweep_xs = (
                sweep[axis_field].to_numpy()
                if native_numeric_axis
                else [grid_lookup[v] for v in sweep[axis_field]]
            )
            ax.scatter(
                sweep_xs,
                sweep[y_field].values,
                color=color,
                marker="o",
                edgecolors="k",
                zorder=3,
            )

        # Positive control: square, sits at the grid center value (same for every scale).
        pos = scale[scale["role"] == "positive-control"]
        if not pos.empty:
            pos_xs = (
                pos[axis_field].to_numpy()
                if native_numeric_axis
                else [grid_lookup[v] for v in pos[axis_field]]
            )
            ax.scatter(
                pos_xs,
                pos[y_field].values,
                color=color,
                marker="s",
                edgecolors="k",
                zorder=4,
            )

        # Negative control (1B only): diamond at the (interpolated) untransferred value.
        neg = (
            scale[scale["role"] == "negative-control"]
            if include_negative_control
            else scale.iloc[0:0]
        )
        if not neg.empty:
            for _, row in neg.iterrows():
                x = (
                    float(row[axis_field])
                    if native_numeric_axis
                    else _interpolate_x(
                        float(row[axis_field]), grid_values, grid_xs, log_scale
                    )
                )
                ax.scatter(
                    [x],
                    [row[y_field]],
                    color=color,
                    marker="D",
                    edgecolors="k",
                    zorder=4,
                )

    if native_numeric_axis:
        if log_scale:
            ax.set_xscale("log")
        ax.margins(x=0.06)
    else:
        assert value_formatter is not None
        # Keep the equal-spaced sweep grid, but let Matplotlib decide how many
        # labels fit at the rendered axis width. The formatter maps the selected
        # integer grid positions back to their underlying hyperparameter values.
        ax.set_xlim(-0.4, len(grid_values) - 0.6)
        ax.xaxis.set_major_locator(_AutoGridLocator(len(grid_values)))
        ax.xaxis.set_major_formatter(
            FuncFormatter(
                lambda position, _index: (
                    value_formatter(grid_values[round(position)])
                    if math.isclose(position, round(position))
                    and 0 <= round(position) < len(grid_values)
                    else ""
                )
            )
        )
        ax.tick_params(axis="x", labelrotation=30)
        for label in ax.get_xticklabels():
            label.set_ha("right")
    ax.set_xlabel(axis_label, labelpad=X_LABEL_PAD)
    ax.set_ylabel(y_label)
    ax.grid(False)


# Learning rate

LEARNING_RATE_SUBPLOT_HEIGHT_PX = 148.0


def plot_learning_rate(df: pd.DataFrame, palette: dict, params: list[int]) -> Figure:
    # A compact near-square canvas matches this single panel's information
    # density. The article declares its exact display width, so typography stays
    # consistent with wider multi-panel figures.
    fig, ax = plt.subplots(figsize=figsize(7.2, 7.2))
    plot_axis(
        ax,
        df,
        axis_role="learning_rate",
        axis_field="learning_rate",
        # Plain text (no Greek symbol) so the whole label renders in the page
        # font, matching the y-axis/legend labels — any mathtext/symbol segment
        # forces the label into DejaVu, which renders at a different size.
        axis_label="Learning rate",
        log_scale=True,
        palette=palette,
        native_numeric_axis=True,
    )
    ax.set_box_aspect(1)
    # A little extra gap so the x-axis label clears the rotated LR tick labels.
    ax.xaxis.labelpad = 6
    fig.tight_layout(rect=(0, 0.21, 1, 1))
    set_square_subplot_height(fig, [ax], LEARNING_RATE_SUBPLOT_HEIGHT_PX)
    # A vertical legend block uses the plot's right side efficiently and avoids
    # a separate column-spacing decision for this compact square panel.
    attach_stacked_legends_right(
        fig,
        ax,
        palette,
        params,
        include_reference=True,
    )
    return fig


# Optimizer

OPTIMIZER_SUBPLOT_HEIGHT_PX = 148.0


def plot_optimizer(df: pd.DataFrame, palette: dict, params: list[int]) -> Figure:
    fig, axes = plt.subplots(1, 2, figsize=figsize(FIGURE_WIDTH, 5.8))
    plot_axis(
        axes[0],
        df,
        axis_role="beta2",
        axis_field="beta2",
        axis_label="β₂",
        log_scale=False,
        palette=palette,
        native_numeric_axis=True,
        include_negative_control=False,
    )
    plot_axis(
        axes[1],
        df,
        axis_role="epsilon",
        axis_field="epsilon",
        axis_label="ε",
        log_scale=True,
        palette=palette,
        native_numeric_axis=True,
        include_negative_control=False,
    )
    for ax in axes:
        ax.set_box_aspect(1)
    fig.tight_layout(rect=(0, 0.31, 1, 1), w_pad=0)
    set_square_subplot_height(fig, axes, OPTIMIZER_SUBPLOT_HEIGHT_PX)
    pack_horizontal_axes(fig, axes)
    attach_legends_below(
        fig,
        palette,
        params,
        include_reference=False,
        legend_y=0.35,
    )
    return fig


# Regional

REGIONAL_SUBPLOT_HEIGHT_PX = 148.0

_REGION_ROWS: tuple[tuple[str, str], ...] = (
    ("cds", "CDS"),
    ("upstream", "Upstream"),
    ("downstream", "Downstream"),
)

_HYPER_COLS: tuple[tuple[str, str, str, bool], ...] = (
    ("learning_rate", "learning_rate", "Learning rate (η)", True),
    ("beta2", "beta2", "β₂", False),
    ("epsilon", "epsilon", "ε", True),
)


def plot_regional(df: pd.DataFrame, palette: dict, params: list[int]) -> Figure:
    """3x3 grid: per-region transfer loss vs each tuned hyper.

    Rows are genomic regions (CDS / upstream / downstream); columns are the
    swept hypers (learning rate / β₂ / ε). y is `eval/val_<region>/loss` from
    the corresponding region column in the transfer CSV. Free y-axis per cell.
    Negative controls are omitted (this figure focuses on transfer-vs-direct
    sweep shapes per region).
    """
    fig, axes = plt.subplots(3, 3, figsize=figsize(FIGURE_WIDTH, 8.8))
    for r, (region_key, region_label) in enumerate(_REGION_ROWS):
        y_field = f"eval_loss_{region_key}"
        for c, (axis_role, axis_field, axis_label, log_scale) in enumerate(_HYPER_COLS):
            ax = axes[r, c]
            plot_axis(
                ax,
                df,
                axis_role=axis_role,
                axis_field=axis_field,
                axis_label=axis_label if r == 2 else "",
                log_scale=log_scale,
                palette=palette,
                native_numeric_axis=True,
                include_negative_control=False,
                y_field=y_field,
                y_label=f"{region_label} loss" if c == 0 else "",
            )
            if r != 2:
                ax.tick_params(axis="x", labelbottom=False)
                ax.set_xlabel("")
            ax.set_box_aspect(1)
    fig.tight_layout(rect=(0, 0.15, 1, 1), h_pad=0.5, w_pad=0)
    set_square_subplot_height(fig, axes.flat, REGIONAL_SUBPLOT_HEIGHT_PX)
    pack_horizontal_axis_columns(fig, axes)
    attach_legends_below(
        fig,
        palette,
        params,
        include_reference=False,
        legend_y=0.13,
    )
    return fig


# Loss scaling

LOSS_SCALING_SUBPLOT_HEIGHT_PX = 268.5


def _fit_kaplan_law(
    params: np.ndarray, losses: np.ndarray, report_path: Path
) -> tuple[float, float, float]:
    """Fit L(N) = A * N^(-alpha) + L_inf via nonlinear least squares.

    Returns (A, alpha, L_inf). L_inf is the irreducible-loss asymptote.

    Raises RuntimeError if the optimizer did not converge, if any scipy warning
    fires during the call, if the parameter covariance is non-finite (parameters
    unidentifiable), or if any fit parameter lands within `bound_tol_frac` of a
    bound (which would indicate the bound is binding rather than the data
    determining the parameter).
    """

    def kaplan(N, A, alpha, L_inf):
        return A * N ** (-alpha) + L_inf

    L_min = float(np.min(losses))
    L_inf0 = max(0.0, L_min - 0.1)
    alpha0 = 0.1
    N_med = float(np.median(params))
    A0 = max((float(np.median(losses)) - L_inf0) * (N_med**alpha0), 1.0)

    lower = np.array([0.0, 0.0, 0.0])
    upper = np.array([np.inf, 5.0, L_min])
    names = ("A", "alpha", "L_inf")

    with warnings.catch_warnings():
        # Treat any scipy/numpy warning during the fit (covariance estimation,
        # convergence, divide-by-zero, etc.) as a hard failure.
        warnings.simplefilter("error")
        popt, pcov, infodict, mesg, ier = curve_fit(
            kaplan,
            params,
            losses,
            p0=[A0, alpha0, L_inf0],
            bounds=(lower, upper),
            maxfev=10000,
            full_output=True,
        )

    # ier ∈ {1, 2, 3, 4} indicates a successful termination of the underlying
    # least-squares algorithm; everything else (0, 5, ...) is a non-convergence.
    if ier not in (1, 2, 3, 4):
        raise RuntimeError(f"Kaplan fit did not converge: ier={ier}, mesg={mesg!r}")

    if not np.all(np.isfinite(pcov)):
        raise RuntimeError(
            f"Kaplan fit covariance is non-finite (parameters unidentifiable): pcov={pcov!r}"
        )

    # Reject fits where any parameter is glued to a bound. For finite bounds use a
    # fraction of the bound interval; for the semi-infinite A upper bound use a
    # fraction of the parameter scale.
    bound_tol_frac = 0.01
    for val, lo, hi, name in zip(popt, lower, upper, names, strict=True):
        scale = (hi - lo) if np.isfinite(hi) else max(abs(val), 1.0)
        margin = bound_tol_frac * scale
        if val - lo < margin or (np.isfinite(hi) and hi - val < margin):
            raise RuntimeError(
                f"Kaplan fit parameter {name}={val:.6g} is at bound "
                f"(lo={lo}, hi={hi}, margin={margin:.3g})"
            )

    A_fit, alpha_fit, L_inf_fit = float(popt[0]), float(popt[1]), float(popt[2])
    _write_kaplan_fit_report(
        params,
        losses,
        A_fit,
        alpha_fit,
        L_inf_fit,
        pcov,
        lower,
        upper,
        names,
        report_path,
    )
    return A_fit, alpha_fit, L_inf_fit


def _write_kaplan_fit_report(
    params: np.ndarray,
    losses: np.ndarray,
    A: float,
    alpha: float,
    L_inf: float,
    pcov: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    names: tuple[str, ...],
    report_path: Path,
) -> None:
    """Write Kaplan fit parameters, bounds, std-errs, and residual stats to a report file."""
    pred = A * params ** (-alpha) + L_inf
    resid = losses - pred
    rmse = float(np.sqrt(np.mean(resid**2)))
    max_abs = float(np.max(np.abs(resid)))
    perr = np.sqrt(np.diag(pcov))
    values = (A, alpha, L_inf)

    lines = [
        "Kaplan scaling-law fit",
        "======================",
        "",
        "L(N) = A · N^(-α) + L_∞",
        "",
        f"data points : {len(params)}",
        f"RMSE        : {rmse:.4g}",
        f"max |resid| : {max_abs:.4g}",
        "",
        f"{'param':<6} {'value':>14} {'± std err':>14}    bounds",
    ]
    for name, val, err, lo, hi in zip(names, values, perr, lower, upper, strict=True):
        hi_str = "∞" if not np.isfinite(hi) else f"{hi:.4g}"
        lines.append(f"{name:<6} {val:>14.6g} {err:>14.3g}    [{lo:.4g}, {hi_str}]")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n")


def plot_loss_scaling(
    history: pd.DataFrame, results: pd.DataFrame, palette: dict, report_path: Path
) -> Figure:
    """1x2: train/eval loss vs step. Inset on eval panel: final loss vs params with Kaplan fit."""
    name_to_params = dict(zip(results["run_name"], results["params"], strict=True))
    history = history.assign(params=history["run_name"].map(name_to_params))
    # Log-x cannot show step=0; the step-0 eval is pre-training and not informative anyway.
    history = history[history["step"] > 0]

    fig, axes = plt.subplots(1, 2, figsize=figsize(FIGURE_WIDTH, 6.5))
    panels = [
        (axes[0], "train/loss", "Train loss"),
        (axes[1], "eval/loss", "Validation loss"),
    ]
    # Eval-loss lines need a high zorder so they draw over the inset on the right panel.
    line_zorder = {axes[0]: 2, axes[1]: 20}
    for ax, metric, ylabel in panels:
        sub = history[history["metric"] == metric]
        for params in sorted(sub["params"].unique()):
            line = sub[sub["params"] == params].sort_values("step")
            ax.plot(
                line["step"],
                line["value"],
                color=palette[int(params)],
                zorder=line_zorder[ax],
            )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Step", labelpad=X_LABEL_PAD)
        ax.set_ylabel(ylabel)
        ax.grid(False)
        ax.set_box_aspect(1)
        # Loss values span ~0.5–1.5; render as plain decimals rather than 6×10⁻¹ etc.
        set_plain_decimal_yticks(ax)

    # Inset on eval-loss panel: final eval loss vs params with Kaplan scaling-law fit.
    # Sits in the bottom-left empty region (low-step / low-loss has no curve data there).
    _attach_kaplan_inset(axes[1], results, palette, report_path)

    fig.tight_layout(rect=(0, 0.08, 1, 1))
    set_square_subplot_height(fig, axes, LOSS_SCALING_SUBPLOT_HEIGHT_PX)

    params_present = sorted({int(p) for p in history["params"].dropna().unique()})
    attach_params_legend_below(
        fig, palette, params_present, width_scale=0.4, handlelength=1.0
    )
    return fig


def _attach_kaplan_inset(
    parent_ax, results: pd.DataFrame, palette: dict, report_path: Path
) -> None:
    """Embed a small params-vs-loss panel with Kaplan fit inside `parent_ax`.

    Inset axes uses parent-axes-fraction coords. The fit equation/constants sit
    in the open space just to the right of the inset.
    """
    fit_df = results.dropna(subset=["params", "eval_loss"]).sort_values("params")
    P = fit_df["params"].astype(float).to_numpy()
    L = fit_df["eval_loss"].astype(float).to_numpy()
    A, alpha, L_inf = _fit_kaplan_law(P, L, report_path)
    pred = A * P ** (-alpha) + L_inf
    ss_res = float(np.sum((L - pred) ** 2))
    ss_tot = float(np.sum((L - L.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    # zorder=0 puts the inset under parent's eval-loss lines (which use zorder=20),
    # so the curves visually cross over the inset like a framed window beneath them.
    # Keep the bottom (y0) where it is; the height controls only the top edge.
    inset_bounds = (0.10, 0.17, 0.27, 0.40)
    inset = parent_ax.inset_axes(list(inset_bounds), zorder=0)

    # Replace the default rectangular background/spines with a rounded FancyBboxPatch
    # placed in parent axes-fraction coords, so the inset reads as a card with rounded edges.
    inset.patch.set_visible(False)
    for spine in inset.spines.values():
        spine.set_visible(False)
    rounded_bg = FancyBboxPatch(
        (inset_bounds[0], inset_bounds[1]),
        inset_bounds[2],
        inset_bounds[3],
        boxstyle="round,pad=0,rounding_size=0.025",
        transform=parent_ax.transAxes,
        facecolor="none",
        edgecolor="black",
        linewidth=0.2,
        zorder=0,
        clip_on=False,
    )
    parent_ax.add_patch(rounded_bg)

    # Y-range: zoom to the data points with a little padding. The asymptote and
    # fit details live in a text block to the right, so the inset needn't reach
    # all the way down to L_inf.
    dataspan = float(L.max() - L.min()) or 1.0
    inset.set_ylim(L.min() - 0.12 * dataspan, L.max() + 0.12 * dataspan)

    P_grid = np.geomspace(P.min() * 0.92, P.max() * 1.08, 200)
    inset.plot(
        P_grid,
        A * P_grid ** (-alpha) + L_inf,
        color="0.2",
        alpha=0.92,
        zorder=2,
    )
    for p, loss in zip(P, L, strict=True):
        inset.scatter(
            [p],
            [loss],
            color=palette[int(p)],
            edgecolors="k",
            zorder=3,
        )

    inset.set_xscale("log")
    inset.set_xlabel("Parameters (N)", labelpad=2)
    inset.set_ylabel("")
    inset.tick_params(length=2.5, pad=1)
    inset.minorticks_off()
    inset.grid(False)
    set_plain_decimal_yticks(inset)

    # Fit equation + constants in the open space just right of the inset (the
    # empty low-loss/early-step corner), bottom-aligned to the inset plot. No
    # background box (transparent) so it doesn't mask the loss curves; high
    # zorder keeps the text itself on top.
    parent_ax.text(
        inset_bounds[0] + inset_bounds[2] + 0.03,
        inset_bounds[1],
        r"$L(N) = A \cdot N^{-\alpha} + L_{\infty}$"
        "\n"
        rf"$A = {A:.3g},\ \alpha = {alpha:.3f}$"
        "\n"
        rf"$L_{{\infty}} = {L_inf:.3f},\ R^2 = {r2:.3f}$",
        transform=parent_ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=MATPLOTLIB_NOTE_SIZE,
        color="0.15",
        linespacing=1.6,
        zorder=25,
    )


# Scaling

FINAL_STEP = 215573

SCALING_SUBPLOT_HEIGHT_PX = 112.0

REDUNDANT_LABEL_HEIGHT_FONT_SIZES = 1.4

PROBE_RESULTS = (
    "s3://oa-bolinas/snakemake/analysis/evals_v2/results/probe/"
    "{model}/{dataset}.parquet"
)

MENDELIAN_SUBSETS: tuple[tuple[str, str], ...] = (
    ("missense_variant", "Missense"),
    ("splicing", "Splicing"),
    ("synonymous_variant", "Synonymous"),
    ("tss_proximal", "Promoter"),
    ("5_prime_UTR_variant", "5′ UTR"),
    ("3_prime_UTR_variant", "3′ UTR"),
)

SGE_SUBSETS: tuple[tuple[str, str], ...] = (
    ("missense_variant", "Missense"),
    ("splicing", "Splicing"),
)

SCALING_SUBSETS = {
    "mendelian_traits": MENDELIAN_SUBSETS,
    "sge": SGE_SUBSETS,
}

READOUTS: tuple[tuple[str, str, str], ...] = (
    ("minus_llr_avg", "Zero-shot LLR", "llr"),
    ("probe_score", "Linear probe", "probe"),
)


def _model_rows() -> pd.DataFrame:
    """Return the eight endpoint model IDs and their parameter counts."""
    metadata = pd.read_csv(local_data.SCALING_RESULTS_PATH)[
        ["run_name", "params"]
    ].copy()
    assert len(metadata) == 8, f"expected 8 scaling models, got {len(metadata)}"
    assert metadata["run_name"].is_unique
    assert metadata["params"].is_unique
    metadata["model"] = (
        metadata["run_name"].str.removeprefix("dna-bolinas-") + f"-step-{FINAL_STEP}"
    )
    return metadata.sort_values("params").reset_index(drop=True)


def _read_paired_metrics(model: str, dataset: str) -> pd.DataFrame:
    """Compute both readouts' AUPRC on identical finite probe-score rows."""
    path = PROBE_RESULTS.format(model=model, dataset=dataset)
    print(f"Reading {path}")
    data = local_data.read_pandas(path)
    required = {
        "label",
        "subset",
        "chrom",
        "probe_score",
        "llr_fwd",
        "llr_rc",
    }
    assert required.issubset(data.columns), (
        f"{path} missing columns {sorted(required - set(data.columns))}"
    )

    # A probe can be absent for a subset that failed its training-data gate.
    # Restrict both readouts to the exact rows on which the probe is defined.
    finite_probe = np.isfinite(data["probe_score"].to_numpy(dtype=float))
    data = data.loc[finite_probe].copy()
    assert len(data) > 0, f"{path} has no finite probe predictions"
    data["minus_llr_avg"] = (
        -(data["llr_fwd"].to_numpy(dtype=float) + data["llr_rc"].to_numpy(dtype=float))
        / 2
    )
    assert np.isfinite(data["minus_llr_avg"]).all()

    metrics = _metrics.per_chrom_ap_table(
        data,
        ["minus_llr_avg", "probe_score"],
        n_bootstrap=1000,
        rng=0,
    )
    wanted = {subset for subset, _ in SCALING_SUBSETS[dataset]}
    metrics = metrics[metrics["subset"].isin(wanted)].copy()
    expected = {(score, subset) for score, *_ in READOUTS for subset in wanted}
    observed = set(zip(metrics["score_type"], metrics["subset"], strict=True))
    assert observed == expected, (
        f"{model}/{dataset}: expected metric cells {sorted(expected)}, "
        f"got {sorted(observed)}"
    )
    assert metrics["value"].between(0, 1).all()
    assert metrics["se"].notna().all()
    return metrics


def load_parameter_scaling_metrics() -> pd.DataFrame:
    """Load all eight endpoints for both datasets and both paired readouts."""
    frames: list[pd.DataFrame] = []
    for row in _model_rows().itertuples(index=False):
        for dataset in SCALING_SUBSETS:
            metrics = _read_paired_metrics(row.model, dataset)
            metrics["dataset"] = dataset
            metrics["params"] = int(row.params)
            metrics["model"] = row.model
            frames.append(metrics)
    result = pd.concat(frames, ignore_index=True)
    expected_rows = (
        8 * sum(len(subsets) for subsets in SCALING_SUBSETS.values()) * len(READOUTS)
    )
    assert len(result) == expected_rows, (
        f"expected {expected_rows} endpoint metric rows, got {len(result)}"
    )
    return result


def _scaling_plot_panel(
    ax: plt.Axes,
    data: pd.DataFrame,
    *,
    title: str,
    show_xlabel: bool,
    show_ylabel: bool,
) -> None:
    """Draw paired readout scaling curves for one consequence."""
    assert set(data["score_type"]) == {score for score, *_ in READOUTS}
    for score_type, _label, protocol in READOUTS:
        series = data[data["score_type"] == score_type].sort_values("params")
        assert len(series) == 8
        color = SCORING_PROTOCOL_COLORS[protocol]
        ax.plot(
            series["params"],
            series["value"] * 100.0,
            color=color,
            linestyle=SCORING_PROTOCOL_LINESTYLES[protocol],
            zorder=3,
        )
        ax.errorbar(
            series["params"],
            series["value"] * 100.0,
            yerr=series["se"] * 100.0,
            fmt="none",
            ecolor=color,
            alpha=COMPARISON_ERRORBAR_ALPHA,
            capsize=0,
            zorder=2,
        )
    ax.set_xscale("log")
    ax.set_title(title)
    if show_xlabel:
        ax.set_xlabel("Model parameters", labelpad=X_LABEL_PAD)
    if show_ylabel:
        ax.set_ylabel("AUPRC (%)")
    ax.grid(False)
    ax.margins(x=0.06)
    ax.set_box_aspect(1)


def plot_scaling(data: pd.DataFrame | None = None) -> Figure:
    """Build the combined Mendelian + SGE parameter-scaling figure."""
    if data is None:
        data = load_parameter_scaling_metrics()

    mosaic = [
        [
            "m_missense",
            "m_missense",
            "m_splicing",
            "m_splicing",
            "m_synonymous",
            "m_synonymous",
        ],
        ["m_promoter", "m_promoter", "m_5utr", "m_5utr", "m_3utr", "m_3utr"],
        [".", "s_missense", "s_missense", "s_splicing", "s_splicing", "."],
    ]
    fig, axes = plt.subplot_mosaic(
        mosaic,
        figsize=figsize(10.0, 9.2),
        gridspec_kw={"hspace": 0.52, "wspace": 0.16},
    )

    mendelian_axes = (
        ("m_missense", "missense_variant", "Missense"),
        ("m_splicing", "splicing", "Splicing"),
        ("m_synonymous", "synonymous_variant", "Synonymous"),
        ("m_promoter", "tss_proximal", "Promoter"),
        ("m_5utr", "5_prime_UTR_variant", "5′ UTR"),
        ("m_3utr", "3_prime_UTR_variant", "3′ UTR"),
    )
    for index, (axis_name, subset, title) in enumerate(mendelian_axes):
        panel = data[
            (data["dataset"] == "mendelian_traits") & (data["subset"] == subset)
        ]
        _scaling_plot_panel(
            axes[axis_name],
            panel,
            title=title,
            show_xlabel=index >= 3,
            show_ylabel=index in (0, 3),
        )

    sge_axes = (
        ("s_missense", "missense_variant", "Missense"),
        ("s_splicing", "splicing", "Splicing"),
    )
    for index, (axis_name, subset, title) in enumerate(sge_axes):
        panel = data[(data["dataset"] == "sge") & (data["subset"] == subset)]
        _scaling_plot_panel(
            axes[axis_name],
            panel,
            title=title,
            show_xlabel=True,
            show_ylabel=index == 0,
        )

    set_square_subplot_height(fig, axes.values(), SCALING_SUBPLOT_HEIGHT_PX)
    mendelian_rows = (
        tuple(axes[axis_name] for axis_name, *_ in mendelian_axes[:3]),
        tuple(axes[axis_name] for axis_name, *_ in mendelian_axes[3:]),
    )
    pack_horizontal_axis_columns(fig, mendelian_rows)
    sge_row = tuple(axes[axis_name] for axis_name, *_ in sge_axes)
    pack_horizontal_axes(fig, sge_row)
    center_axes_block(fig, sge_row, (axis for row in mendelian_rows for axis in row))
    freed_label_height = (
        plt.rcParams["font.size"]
        * REDUNDANT_LABEL_HEIGHT_FONT_SIZES
        / (fig.get_figheight() * 72.0)
    )
    for axis_name, *_ in (*mendelian_axes[3:], *sge_axes):
        position = axes[axis_name].get_position()
        axes[axis_name].set_position(
            (
                position.x0,
                position.y0 + freed_label_height,
                position.width,
                position.height,
            )
        )

    handles = [
        Line2D(
            [0],
            [0],
            color=SCORING_PROTOCOL_COLORS[protocol],
            linestyle=SCORING_PROTOCOL_LINESTYLES[protocol],
        )
        for _score, _label, protocol in READOUTS
    ]
    labels = [label for _score, label, *_rest in READOUTS]
    fig.legend(
        handles,
        labels,
        title="Scoring protocol",
        ncol=2,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.985),
        frameon=False,
        handletextpad=0.5,
        columnspacing=1.8,
    )
    # Dataset headers sit above their respective facet blocks. Their vertical
    # positions are derived from the live axes rather than hard-coded pixels.
    fig.canvas.draw()
    mendelian_top = max(axes[name].get_position().y1 for name, *_ in mendelian_axes)
    sge_top = max(axes[name].get_position().y1 for name, *_ in sge_axes)
    fig.text(0.02, mendelian_top + 0.012, "Mendelian", weight="bold")
    fig.text(0.02, sge_top + 0.012, "SGE", weight="bold")
    return fig


# Likelihood

MATCHED_LOSS_COLUMN = {
    "missense_variant": "eval_loss_cds",
    "splicing": "eval_loss_cds",
    "synonymous_variant": "eval_loss_cds",
    "tss_proximal": "eval_loss_upstream",
    "5_prime_UTR_variant": "eval_loss_upstream",
    "3_prime_UTR_variant": "eval_loss_downstream",
}

LOSS_COLUMNS = tuple(sorted(set(MATCHED_LOSS_COLUMN.values())))

ANNOTATION_HEADROOM_SUBSETS = frozenset({"synonymous_variant", "3_prime_UTR_variant"})

ANNOTATION_HEADROOM_FRACTION = 0.3


def load_loss_vs_auprc(
    results: pd.DataFrame | None = None,
    metrics: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Join paired AUPRCs to each consequence's matched validation LL."""
    if results is None:
        results = pd.read_csv(local_data.SCALING_RESULTS_PATH)
    if metrics is None:
        metrics = load_parameter_scaling_metrics()

    required_results = {"params", *LOSS_COLUMNS}
    assert required_results.issubset(results.columns), (
        f"scaling results missing {sorted(required_results - set(results.columns))}"
    )
    metadata = results[["params", *LOSS_COLUMNS]].copy()
    assert len(metadata) == 8
    assert metadata["params"].is_unique
    assert np.isfinite(metadata[list(LOSS_COLUMNS)].to_numpy(dtype=float)).all()

    data = metrics.merge(metadata, on="params", how="left", validate="many_to_one")
    assert len(data) == len(metrics)
    data["ll"] = np.nan
    for subset, loss_column in MATCHED_LOSS_COLUMN.items():
        mask = data["subset"] == subset
        data.loc[mask, "ll"] = -data.loc[mask, loss_column]
    assert data["ll"].notna().all()
    assert np.isfinite(data["ll"].to_numpy(dtype=float)).all()
    return data


def _likelihood_plot_panel(
    ax: plt.Axes,
    data: pd.DataFrame,
    *,
    title: str,
    add_annotation_headroom: bool,
    show_xlabel: bool,
    show_ylabel: bool,
) -> None:
    """Draw paired protocol fits and endpoint AUPRCs for one consequence."""
    assert set(data["score_type"]) == {score for score, *_ in READOUTS}
    marker = MODEL_FAMILY_MARKERS["marindna"]
    for index, (score_type, _label, protocol) in enumerate(READOUTS):
        series = data[data["score_type"] == score_type].sort_values("ll")
        assert len(series) == 8
        assert series["ll"].is_unique
        color = SCORING_PROTOCOL_COLORS[protocol]
        xs = series["ll"].to_numpy(dtype=float)
        ys = series["value"].to_numpy(dtype=float) * 100.0
        ax.errorbar(
            xs,
            ys,
            yerr=series["se"] * 100.0,
            fmt="none",
            ecolor=color,
            alpha=COMPARISON_ERRORBAR_ALPHA,
            capsize=0,
            zorder=1,
        )
        ax.scatter(
            xs,
            ys,
            color=color,
            marker=marker,
            edgecolors="#1f1e1b",
            zorder=3,
        )
        slope, intercept = np.polyfit(xs, ys, 1)
        x_line = np.array([xs.min(), xs.max()])
        ax.plot(
            x_line,
            slope * x_line + intercept,
            color=color,
            linestyle=SCORING_PROTOCOL_LINESTYLES[protocol],
            zorder=2,
        )
        correlation = float(np.corrcoef(xs, ys)[0, 1])
        assert np.isfinite(correlation)
        ax.text(
            0.04,
            0.96 - index * 0.16,
            f"$r$ =\u2009{correlation:.2f}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=MATPLOTLIB_NOTE_SIZE,
            color=color,
        )

    ax.set_title(title)
    if show_xlabel:
        ax.set_xlabel("LL (−loss)", labelpad=X_LABEL_PAD)
    if show_ylabel:
        ax.set_ylabel("AUPRC (%)")
    ax.grid(False)
    ax.margins(x=0.08)
    if add_annotation_headroom:
        ymin, ymax = ax.get_ylim()
        ax.set_ylim(ymin, ymax + (ymax - ymin) * ANNOTATION_HEADROOM_FRACTION)
    ax.set_box_aspect(1)


def plot_likelihood(
    results: pd.DataFrame | None = None,
    metrics: pd.DataFrame | None = None,
) -> Figure:
    """Build the combined Mendelian + SGE LL-versus-AUPRC figure."""
    data = load_loss_vs_auprc(results, metrics)

    mosaic = [
        [
            "m_missense",
            "m_missense",
            "m_splicing",
            "m_splicing",
            "m_synonymous",
            "m_synonymous",
        ],
        ["m_promoter", "m_promoter", "m_5utr", "m_5utr", "m_3utr", "m_3utr"],
        [".", "s_missense", "s_missense", "s_splicing", "s_splicing", "."],
    ]
    fig, axes = plt.subplot_mosaic(
        mosaic,
        figsize=figsize(10.0, 9.2),
        gridspec_kw={"hspace": 0.52, "wspace": 0.16},
    )

    mendelian_axes = (
        ("m_missense", "missense_variant", "Missense"),
        ("m_splicing", "splicing", "Splicing"),
        ("m_synonymous", "synonymous_variant", "Synonymous"),
        ("m_promoter", "tss_proximal", "Promoter"),
        ("m_5utr", "5_prime_UTR_variant", "5′ UTR"),
        ("m_3utr", "3_prime_UTR_variant", "3′ UTR"),
    )
    assert tuple((subset, title) for _, subset, title in mendelian_axes) == tuple(
        MENDELIAN_SUBSETS
    )
    for index, (axis_name, subset, title) in enumerate(mendelian_axes):
        panel = data[
            (data["dataset"] == "mendelian_traits") & (data["subset"] == subset)
        ]
        _likelihood_plot_panel(
            axes[axis_name],
            panel,
            title=title,
            add_annotation_headroom=subset in ANNOTATION_HEADROOM_SUBSETS,
            show_xlabel=index >= 3,
            show_ylabel=index in (0, 3),
        )

    sge_axes = (
        ("s_missense", "missense_variant", "Missense"),
        ("s_splicing", "splicing", "Splicing"),
    )
    assert tuple((subset, title) for _, subset, title in sge_axes) == tuple(SGE_SUBSETS)
    for index, (axis_name, subset, title) in enumerate(sge_axes):
        panel = data[(data["dataset"] == "sge") & (data["subset"] == subset)]
        _likelihood_plot_panel(
            axes[axis_name],
            panel,
            title=title,
            add_annotation_headroom=False,
            show_xlabel=True,
            show_ylabel=index == 0,
        )

    set_square_subplot_height(fig, axes.values(), SCALING_SUBPLOT_HEIGHT_PX)
    mendelian_rows = (
        tuple(axes[axis_name] for axis_name, *_ in mendelian_axes[:3]),
        tuple(axes[axis_name] for axis_name, *_ in mendelian_axes[3:]),
    )
    # Loss tick labels are wider than the parameter ticks in Figure 12. Pack
    # their rendered bounds flush so the square axes themselves keep the same
    # column spacing in both figures.
    pack_horizontal_axis_columns(fig, mendelian_rows, gap_font_sizes=0.0)
    sge_row = tuple(axes[axis_name] for axis_name, *_ in sge_axes)
    pack_horizontal_axes(fig, sge_row)
    center_axes_block(fig, sge_row, (axis for row in mendelian_rows for axis in row))
    freed_label_height = (
        plt.rcParams["font.size"]
        * REDUNDANT_LABEL_HEIGHT_FONT_SIZES
        / (fig.get_figheight() * 72.0)
    )
    for axis_name, *_ in (*mendelian_axes[3:], *sge_axes):
        position = axes[axis_name].get_position()
        axes[axis_name].set_position(
            (
                position.x0,
                position.y0 + freed_label_height,
                position.width,
                position.height,
            )
        )

    handles = [
        Line2D(
            [0],
            [0],
            color=SCORING_PROTOCOL_COLORS[protocol],
            linestyle=SCORING_PROTOCOL_LINESTYLES[protocol],
        )
        for _score, _label, protocol in READOUTS
    ]
    labels = [label for _score, label, *_rest in READOUTS]
    fig.legend(
        handles,
        labels,
        title="Scoring protocol",
        ncol=2,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.985),
        frameon=False,
        handletextpad=0.5,
        columnspacing=1.8,
    )

    fig.canvas.draw()
    mendelian_top = max(axes[name].get_position().y1 for name, *_ in mendelian_axes)
    sge_top = max(axes[name].get_position().y1 for name, *_ in sge_axes)
    fig.text(0.02, mendelian_top + 0.012, "Mendelian", weight="bold")
    fig.text(0.02, sge_top + 0.012, "SGE", weight="bold")
    return fig


# Missense

MISSENSE_DATASET = "mendelian_traits"

SUBSET = "missense_variant"

MISSENSE_SUBPLOT_HEIGHT_PX = 164.0

MARIN = "MarinDNA"

EVO2 = "Evo 2"

FAMILIES = (MARIN, EVO2)

EVO2_MODELS: tuple[tuple[str, float], ...] = (
    ("evo2_1b_base", 1.0e9),
    ("evo2_7b", 7.0e9),
    ("evo2_40b", 40.0e9),
)


def _format_parameter_count(value: float, _position: int) -> str:
    """Compact labels for Matplotlib-selected log-scale parameter ticks."""
    if value >= 1e9:
        return f"{value / 1e9:g}B"
    if value >= 1e6:
        return f"{value / 1e6:g}M"
    return f"{value:g}"


def _evo2_path(model: str) -> str:
    short = local_data.EVO2_DATASET_SHORT[MISSENSE_DATASET]
    return f"{local_data.EVO2_PROBE_METRICS_GIST_BASE}/{short}_{model}_train_probe_metrics.parquet"


def load_missense_comparison(
    marin_metrics: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Load paired missense AUPRCs for both model families."""
    if marin_metrics is None:
        marin_metrics = load_parameter_scaling_metrics()
    required = {"dataset", "subset", "score_type", "value", "se", "n", "params"}
    assert required.issubset(marin_metrics.columns), (
        f"MarinDNA metrics missing {sorted(required - set(marin_metrics.columns))}"
    )
    marin = marin_metrics[
        (marin_metrics["dataset"] == MISSENSE_DATASET)
        & (marin_metrics["subset"] == SUBSET)
    ][["score_type", "value", "se", "n", "params"]].copy()
    assert len(marin) == 8 * len(READOUTS)
    marin["family"] = MARIN

    evo_frames: list[pd.DataFrame] = []
    for model, params in EVO2_MODELS:
        path = _evo2_path(model)
        print(f"Reading {path}")
        frame = local_data.read_polars(path).filter(
            (pl.col("subset") == SUBSET)
            & pl.col("score_type").is_in([score for score, *_ in READOUTS])
            & (pl.col("split") == "train")
        )
        assert frame.height == len(READOUTS), (
            f"{model}: expected {len(READOUTS)} missense score rows, got {frame.height}"
        )
        evo = frame.select("score_type", "value", "se", "n").to_pandas()
        evo["params"] = params
        evo["family"] = EVO2
        evo_frames.append(evo)

    result = pd.concat([marin, *evo_frames], ignore_index=True)
    assert len(result) == (8 + len(EVO2_MODELS)) * len(READOUTS)
    assert result["value"].between(0, 1).all()
    assert result["se"].notna().all()
    assert (result["se"] >= 0).all()
    assert result["n"].nunique() == 1, (
        f"MarinDNA and Evo 2 missense rows differ: {sorted(result['n'].unique())}"
    )
    expected_cells = {
        (family, score) for family in FAMILIES for score, *_rest in READOUTS
    }
    assert (
        set(zip(result["family"], result["score_type"], strict=True)) == expected_cells
    )
    return result


def plot_missense(marin_metrics: pd.DataFrame | None = None) -> Figure:
    """Build the combined MarinDNA + Evo 2 missense scaling figure."""
    data = load_missense_comparison(marin_metrics)
    fig, ax = plt.subplots(figsize=figsize(8.4, 7.2))

    for family in FAMILIES:
        family_key = "marindna" if family == MARIN else "evo2"
        marker = MODEL_FAMILY_MARKERS[family_key]
        for score_type, _label, protocol in READOUTS:
            series = data[
                (data["family"] == family) & (data["score_type"] == score_type)
            ].sort_values("params")
            expected = 8 if family == MARIN else len(EVO2_MODELS)
            assert len(series) == expected
            color = SCORING_PROTOCOL_COLORS[protocol]
            ax.plot(
                series["params"],
                series["value"] * 100.0,
                color=color,
                linestyle=SCORING_PROTOCOL_LINESTYLES[protocol],
                marker=marker,
                markeredgecolor="#1f1e1b",
                zorder=3,
            )
            ax.errorbar(
                series["params"],
                series["value"] * 100.0,
                yerr=series["se"] * 100.0,
                fmt="none",
                ecolor=color,
                alpha=COMPARISON_ERRORBAR_ALPHA,
                capsize=0,
                zorder=2,
            )

    ax.set_xscale("log")
    ax.xaxis.set_major_locator(LogLocator(base=10))
    ax.xaxis.set_major_formatter(FuncFormatter(_format_parameter_count))
    ax.set_xlabel("Model parameters", labelpad=X_LABEL_PAD)
    ax.set_ylabel("AUPRC (%)")
    ax.grid(False)
    ax.margins(x=0.04)
    ax.set_box_aspect(1)
    fig.subplots_adjust(left=0.16, right=0.62, top=0.95, bottom=0.16)
    set_square_subplot_height(fig, (ax,), MISSENSE_SUBPLOT_HEIGHT_PX)

    protocol_handles = [
        Line2D(
            [0],
            [0],
            color=SCORING_PROTOCOL_COLORS[protocol],
            linestyle=SCORING_PROTOCOL_LINESTYLES[protocol],
            label=label,
        )
        for _score, label, protocol in READOUTS
    ]
    family_handles = [
        Line2D(
            [0],
            [0],
            color="#4b4b4b",
            marker=MODEL_FAMILY_MARKERS["marindna" if family == MARIN else "evo2"],
            markerfacecolor="#b8b0a3",
            markeredgecolor="#1f1e1b",
            linestyle="none",
            label=family,
        )
        for family in FAMILIES
    ]
    axis_position = ax.get_position()
    legend_x = axis_position.x1 + 0.04
    protocol_legend = fig.legend(
        handles=protocol_handles,
        title="Scoring protocol",
        ncol=1,
        loc="upper left",
        bbox_to_anchor=(legend_x, axis_position.y1),
        frameon=False,
        handlelength=2.4,
    )
    fig.add_artist(protocol_legend)
    fig.canvas.draw()
    protocol_bottom = (
        protocol_legend.get_window_extent(renderer=fig.canvas.get_renderer()).y0
        / fig.bbox.height
    )
    fig.legend(
        handles=family_handles,
        title="Model family",
        ncol=1,
        loc="upper left",
        bbox_to_anchor=(legend_x, protocol_bottom - 0.025),
        frameon=False,
        handletextpad=0.4,
    )
    return fig


# Throughput

MACRO_COLUMN = "Macro Avg (8 subsets)"

THROUGHPUT_COLUMN = "Throughput (variants / hour)"

MODEL_NAMES = (
    "exp135-1B-m5.1",
    "Evo 2 (1B base)",
    "Evo 2 (7B)",
    "Evo 2 (40B)",
)

DISPLAY_NAMES = {
    "exp135-1B-m5.1": "MarinDNA 1B",
    "Evo 2 (1B base)": "Evo 2 1B",
    "Evo 2 (7B)": "Evo 2 7B",
    "Evo 2 (40B)": "Evo 2 40B",
}

MARINDNA_COLOR = SCORING_PROTOCOL_COLORS["llr"]

EVO2_COLOR = "#8a7d69"


def _headline_data(
    leaderboard: pd.DataFrame,
    inference_costs: pd.DataFrame,
) -> pd.DataFrame:
    assert leaderboard["Model"].is_unique
    assert inference_costs["Model"].is_unique
    assert set(inference_costs["Model"]) == set(MODEL_NAMES)
    selected = leaderboard[leaderboard["Model"].isin(MODEL_NAMES)][
        ["Model", MACRO_COLUMN]
    ]
    assert set(selected["Model"]) == set(MODEL_NAMES)
    data = inference_costs.merge(selected, on="Model", validate="one_to_one")
    assert len(data) == len(MODEL_NAMES)
    assert data["Time (s / 1M variants)"].gt(0).all()
    data[THROUGHPUT_COLUMN] = 1_000_000 * 60 * 60 / data["Time (s / 1M variants)"]
    assert data[THROUGHPUT_COLUMN].gt(0).all()
    assert data[MACRO_COLUMN].between(0, 100).all()
    return data


def plot_throughput(leaderboard: pd.DataFrame, inference_costs: pd.DataFrame) -> Figure:
    data = _headline_data(leaderboard, inference_costs)
    evo2 = data[data["Family"] == "Evo 2"].sort_values("Parameters (B)")
    marindna = data[data["Family"] == "MarinDNA"]
    assert len(evo2) == 3
    assert len(marindna) == 1

    fig, ax = plt.subplots(figsize=figsize(7.0, 3.2))
    ax.plot(
        evo2[THROUGHPUT_COLUMN],
        evo2[MACRO_COLUMN],
        color=EVO2_COLOR,
        marker=MODEL_FAMILY_MARKERS["evo2"],
        markeredgecolor="#1f1e1b",
        label="Evo 2",
        zorder=2,
    )
    ax.plot(
        marindna[THROUGHPUT_COLUMN],
        marindna[MACRO_COLUMN],
        color=MARINDNA_COLOR,
        marker=MODEL_FAMILY_MARKERS["marindna"],
        markeredgecolor="#1f1e1b",
        linestyle="none",
        label="MarinDNA",
        zorder=3,
    )

    label_offsets = {
        "exp135-1B-m5.1": (-8, 0, "right", "center"),
        "Evo 2 (1B base)": (8, 0, "left", "center"),
        "Evo 2 (7B)": (8, 0, "left", "center"),
        "Evo 2 (40B)": (8, 0, "left", "center"),
    }
    for _, row in data.iterrows():
        model = str(row["Model"])
        score = float(row[MACRO_COLUMN])
        dx, dy, ha, va = label_offsets[model]
        ax.annotate(
            DISPLAY_NAMES[model],
            xy=(float(row[THROUGHPUT_COLUMN]), score),
            xytext=(dx, dy),
            textcoords="offset points",
            ha=ha,
            va=va,
            color=MARINDNA_COLOR if model == "exp135-1B-m5.1" else EVO2_COLOR,
            fontweight="bold" if model == "exp135-1B-m5.1" else "normal",
        )

    ax.set_xscale("log")
    ax.set_xlabel("Throughput (variants/hour)")
    ax.set_ylabel("VEP AUPRC (%)")
    ax.grid(which="major", color="#d9ccba", linewidth=0.8)

    fig.tight_layout()
    return fig


# Lineage

TOKENS_PER_STEP = 8192 * 256

RESULTS_ROOT = "s3://oa-bolinas/snakemake/analysis/evals_v2/results"


@dataclass(frozen=True)
class OfflineRun:
    """One lineage stage and its existing HF-exported checkpoint steps."""

    mix: str
    index: int
    steps: tuple[int, ...]


RUNS: tuple[OfflineRun, ...] = (
    OfflineRun("uniform", 0, (10000, 20000, 25004)),
    OfflineRun("uniform_to_uniform_1", 18, (10000, 20000, 29579)),
    OfflineRun("exp135-zoonomia-m5.1", 24, (30000, 40000, 50000, 59158)),
    OfflineRun("exp135-zoonomia-m1", 20, (10000, 20000, 29579)),
    OfflineRun("exp135-zoonomia-m1.1", 26, (30000, 40000, 50000, 53243)),
    OfflineRun("exp135-zoonomia-m1.2", 28, (50000, 60000, 70000, 70991)),
    OfflineRun("exp135-zoonomia-m1.3", 30, (60000, 70000, 80000, 82823)),
    OfflineRun("exp135-zoonomia-m3", 22, (10000, 20000, 29578)),
    OfflineRun("exp135-zoonomia-m3.1", 25, (30000, 40000, 50000, 53238)),
    OfflineRun("exp135-zoonomia-m3.2", 27, (50000, 60000, 70000, 70983)),
    OfflineRun("exp135-zoonomia-m3.3", 29, (60000, 70000, 80000, 82813)),
)

BY_MIX = {run.mix: run for run in RUNS}

SHORT_NAMES = {
    "exp135-zoonomia-m5.1": "exp135-m5.1",
}

LINEAGES: tuple[tuple[str, str, str, str], ...] = (
    ("exp135-zoonomia-m5.1", "m5.1", EARTH_QUAL[0], "o"),
    ("exp135-zoonomia-m1.3", "m1.3", EARTH_QUAL[3], "o"),
    ("exp135-zoonomia-m3.3", "m3.3", EARTH_QUAL[4], "o"),
)

PANELS: tuple[tuple[str, str], ...] = (
    ("missense_variant", "Missense"),
    ("splicing", "Splicing"),
    ("synonymous_variant", "Synonymous"),
    ("tss_proximal", "Promoter"),
    ("5_prime_UTR_variant", "5' UTR"),
    ("3_prime_UTR_variant", "3' UTR"),
    ("distal", "Distal"),
    ("non_coding_transcript_exon_variant", "ncRNA"),
)

ALL_SUBSETS = tuple(subset for subset, _label in PANELS)

MACRO_SUBSET = "_macro_avg_"

ALL_METRIC_SUBSETS = (*ALL_SUBSETS, MACRO_SUBSET)

WORLD_CONFIG = {
    "llr": ("metrics", "minus_llr_avg", "pooled AUPRC", "zero-shot LLR"),
    "probe": (
        "probe_metrics",
        "probe_score",
        "chromosome-weighted AUPRC",
        "linear probe",
    ),
}

MACRO_ACCENT = "#59636e"

MACRO_FILL = "#f1eee8"


def _model_id(mix: str, step: int) -> str:
    run = BY_MIX[mix]
    short = SHORT_NAMES.get(mix, mix)
    return f"mix-v0.9-p1B-i{run.index}-{short}-step-{step}"


@lru_cache(maxsize=None)
def _read_checkpoint(kind: str, mix: str, step: int) -> dict[str, tuple[float, float]]:
    """Return stored ``(AUPRC, SE)`` values for the eight subsets and their macro."""
    result_kind, score_type, _metric_label, _method_label = WORLD_CONFIG[kind]
    model_id = _model_id(mix, step)
    path = f"{RESULTS_ROOT}/{result_kind}/{model_id}/mendelian_traits.parquet"
    data = (
        local_data.read_polars(path)
        .filter(
            (pl.col("score_type") == score_type)
            & (pl.col("split") == "train")
            & pl.col("subset").is_in(ALL_METRIC_SUBSETS)
        )
        .select("subset", "value", "se")
    )
    assert data.height == len(ALL_METRIC_SUBSETS), (
        f"{model_id}/{kind}: expected {len(ALL_METRIC_SUBSETS)} metric rows, "
        f"got {data.height}"
    )
    assert data["subset"].n_unique() == len(ALL_METRIC_SUBSETS)
    assert data["value"].is_not_null().all()
    assert data["se"].is_not_null().all()
    assert data["value"].is_between(0, 1, closed="both").all()
    assert data["se"].is_between(0, 1, closed="both").all()
    return {subset: (float(value), float(se)) for subset, value, se in data.iter_rows()}


def _chain(leaf: str) -> list[str]:
    """Return the root-to-leaf stage sequence for one lineage."""
    chain: list[str] = []
    mix: str | None = leaf
    while mix is not None:
        chain.append(mix)
        mix = local_data.BY_MIX[mix].parent
    return chain[::-1]


def _phase_start_step(mix: str, results: pd.DataFrame) -> float:
    row = results.loc[mix]
    own_steps = round(float(row["tokens"]) / TOKENS_PER_STEP)
    return float(row["num_train_steps"]) - own_steps


def _composed_curve(
    kind: str,
    leaf: str,
    subset: str,
    results: pd.DataFrame,
    own_tokens: dict[str, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compose existing offline checkpoints on the cumulative-token axis."""
    chain = _chain(leaf)
    offsets = [
        sum(local_data.inherited_components(mix, own_tokens).values()) for mix in chain
    ]
    tokens: list[float] = []
    values: list[float] = []
    ses: list[float] = []

    for phase_index, mix in enumerate(chain):
        start = _phase_start_step(mix, results)
        cutoff = (
            offsets[phase_index + 1] + TOKENS_PER_STEP
            if phase_index + 1 < len(chain)
            else np.inf
        )
        for step in BY_MIX[mix].steps:
            # Drop the final pre-shift m5.1 checkpoint: it is only 0.9B tokens
            # before the first post-shift point, which aligns with both controls.
            if (
                leaf == "exp135-zoonomia-m5.1"
                and mix == "uniform_to_uniform_1"
                and step == 29579
            ):
                continue
            cumulative_tokens = offsets[phase_index] + (step - start) * TOKENS_PER_STEP
            if cumulative_tokens > cutoff:
                continue
            value, se = _read_checkpoint(kind, mix, step)[subset]
            tokens.append(cumulative_tokens)
            values.append(value)
            ses.append(se)

    order = np.argsort(tokens)
    result_tokens = np.asarray(tokens)[order]
    result_values = np.asarray(values)[order]
    result_ses = np.asarray(ses)[order]
    expected_checkpoints = 8 if leaf == "exp135-zoonomia-m5.1" else 9
    assert len(result_tokens) == expected_checkpoints, (
        f"{kind}/{leaf}: expected {expected_checkpoints} on-path checkpoints, "
        f"got {len(result_tokens)}"
    )
    assert np.all(np.diff(result_tokens) > 0), (
        f"{kind}/{leaf}: token order is not strict"
    )
    return result_tokens, result_values, result_ses


def _lineage_draw_panel(
    ax: plt.Axes,
    *,
    kind: str,
    subset: str,
    results: pd.DataFrame,
    own_tokens: dict[str, float],
    token_cutoff: float,
) -> None:
    """Draw raw offline checkpoints with capless ±1 SE bars."""
    for leaf, _label, color, marker in LINEAGES:
        tokens, values, ses = _composed_curve(kind, leaf, subset, results, own_tokens)
        if leaf != "exp135-zoonomia-m5.1":
            keep = tokens <= token_cutoff
            if keep.any() and not keep.all():
                keep[np.argmax(~keep)] = True
            tokens = tokens[keep]
            values = values[keep]
            ses = ses[keep]
        ax.errorbar(
            tokens / 1e9,
            values * 100.0,
            yerr=ses * 100.0,
            color=color,
            marker=marker,
            ecolor=color,
            capsize=0,
            zorder=3,
        )


def _highlight_macro(ax: plt.Axes) -> None:
    ax.add_patch(
        Rectangle(
            (0, 0),
            1,
            1,
            transform=ax.transAxes,
            facecolor=MACRO_FILL,
            edgecolor="none",
            zorder=-10,
        )
    )


def _attach_legend(fig: plt.Figure) -> None:
    handles = [
        Line2D(
            [0],
            [0],
            color=color,
            marker=marker,
        )
        for _leaf, _label, color, marker in LINEAGES
    ]
    labels = [label for _leaf, label, _color, _marker in LINEAGES]
    fig.legend(
        handles,
        labels,
        title="Mixture lineage",
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.95),
        **LEGEND_KW,
    )


def plot_lineage(kind: str, results_df: pd.DataFrame | None = None) -> Figure:
    """Build one nine-panel offline prototype without smoothing."""
    assert kind in WORLD_CONFIG
    if results_df is None:
        results_df = local_data.load_mixture()
    results = results_df.set_index("mix")
    own_tokens = {mix: float(results.loc[mix, "tokens"]) for mix in results.index}

    m5_tokens, _m5_values, _m5_ses = _composed_curve(
        kind,
        "exp135-zoonomia-m5.1",
        MACRO_SUBSET,
        results,
        own_tokens,
    )
    token_cutoff = float(m5_tokens.max())
    shift_x = (
        sum(
            local_data.inherited_components(
                "exp135-zoonomia-m5.1",
                own_tokens,
            ).values()
        )
        / 1e9
    )

    grid_panels: list[tuple[str, str, bool]] = [(MACRO_SUBSET, "Macro Avg", True)]
    grid_panels.extend((subset, label, False) for subset, label in PANELS)

    fig, axes = plt.subplots(
        3,
        3,
        sharex=True,
        figsize=figsize(FIGURE_WIDTH * 0.70, FIGURE_WIDTH * 0.80),
    )
    fig.subplots_adjust(
        left=0.115,
        right=0.99,
        bottom=0.07,
        top=0.84,
        wspace=0.18,
        hspace=0.16,
    )

    for ax, (subset, title, is_macro) in zip(axes.flat, grid_panels, strict=True):
        _lineage_draw_panel(
            ax,
            kind=kind,
            subset=subset,
            results=results,
            own_tokens=own_tokens,
            token_cutoff=token_cutoff,
        )
        ax.grid(False)
        ax.yaxis.set_major_locator(MaxNLocator(nbins="auto", integer=True))
        ax.yaxis.set_major_formatter(StrMethodFormatter("{x:.0f}"))
        ax.set_box_aspect(1)
        for name, spine in ax.spines.items():
            spine.set_visible(name in {"left", "bottom"})
            if spine.get_visible():
                spine.set_linewidth(0.6)
        if is_macro:
            _highlight_macro(ax)
        ax.set_title(
            title,
            fontweight="bold" if is_macro else "normal",
            y=1.03,
            pad=0,
        )

        ax.axvline(
            shift_x,
            color="#6b7280",
            linestyle=(0, (4, 2)),
            linewidth=0.9,
            zorder=1,
        )
        if is_macro:
            ax.annotate(
                "m5.1 shift\n3→5 regions",
                xy=(shift_x, 0.02),
                xycoords=ax.get_xaxis_transform(),
                xytext=(4, 0),
                textcoords="offset points",
                ha="left",
                va="bottom",
                fontsize=MATPLOTLIB_NOTE_SIZE,
                color=MACRO_ACCENT,
                zorder=4,
                linespacing=1.35,
            )
    axes[-1, 1].set_xlabel("Tokens (B)", labelpad=4)
    for ax in axes[:, 0]:
        ax.set_ylabel("AUPRC (%)")

    _attach_legend(fig)
    return fig


# Heatmap

HEATMAP_CELL_ASPECT = 0.74

HEATMAP_SUBSET_DISPLAY: dict[str, str] = {
    "missense_variant": "Missense",
    "splicing": "Splicing",
    "5_prime_UTR_variant": "5' UTR",
    "tss_proximal": "Promoter",
    "non_coding_transcript_exon_variant": "ncRNA",
    "3_prime_UTR_variant": "3' UTR",
    "distal": "Distal",
    "synonymous_variant": "Synonymous",
}

SUBSET_ORDER = list(HEATMAP_SUBSET_DISPLAY)


def _luminance(rgba: tuple[float, ...] | np.ndarray) -> float:
    r, g, b = rgba[:3]
    return float(0.2126 * r + 0.7152 * g + 0.0722 * b)


def table_from_normalized(rows: pl.DataFrame) -> pd.DataFrame:
    """Convert normalized leaderboard rows to an AUPRC-percent heatmap table."""
    keep = [_metrics.MACRO_AVG_SUBSET, *SUBSET_ORDER]
    selected = (
        rows.filter(pl.col("subset").is_in(keep))
        .with_columns(value=pl.col("value") * 100.0)
        .select(["method_display", "family", "subset", "value"])
    )
    duplicates = (
        selected.group_by(["method_display", "family", "subset"])
        .len()
        .filter(pl.col("len") != 1)
    )
    assert duplicates.height == 0, f"duplicate leaderboard cells:\n{duplicates}"
    pdf = selected.to_pandas()
    return pdf.pivot(
        index=["method_display", "family"], columns="subset", values="value"
    ).reindex(columns=keep)


def render_heatmap(
    table: pd.DataFrame,
) -> Figure:
    """Render one independently sorted and color-normalized leaderboard."""
    table = table.sort_values(_metrics.MACRO_AVG_SUBSET, ascending=False)
    display_names = [index[0] for index in table.index]
    families = [index[1] for index in table.index]
    column_labels = ["Macro Avg", *HEATMAP_SUBSET_DISPLAY.values()]
    matrix = table[[_metrics.MACRO_AVG_SUBSET, *SUBSET_ORDER]].to_numpy(dtype=float)
    n_rows, n_columns = matrix.shape

    finite = matrix[np.isfinite(matrix)]
    assert finite.size > 0, "leaderboard has no finite values"
    norm = Normalize(float(finite.min()), float(finite.max()))

    fig, ax = plt.subplots(figsize=figsize(10.0, max(4.4, 0.30 * n_rows)))
    # A fixed cell aspect keeps each model row the same height when paired
    # leaderboards contain different numbers of models.
    ax.imshow(matrix, cmap=HEATMAP_CMAP, norm=norm, aspect=HEATMAP_CELL_ASPECT)

    ax.set_xticks(np.arange(-0.5, n_columns), minor=True)
    ax.set_yticks(np.arange(-0.5, n_rows), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.5)
    ax.tick_params(which="both", length=0)

    for row_index in range(n_rows):
        for column_index in range(n_columns):
            value = matrix[row_index, column_index]
            if not np.isfinite(value):
                continue
            text_color = (
                "white" if _luminance(HEATMAP_CMAP(norm(value))) < 0.5 else "black"
            )
            ax.text(
                column_index,
                row_index,
                f"{value:.1f}",
                ha="center",
                va="center",
                fontsize=MATPLOTLIB_NOTE_SIZE,
                color=text_color,
                fontweight="bold" if column_index == 0 else "normal",
            )

    ax.add_patch(
        Rectangle(
            (-0.5, -0.5),
            1,
            n_rows,
            fill=False,
            edgecolor="black",
            linewidth=2.0,
            zorder=5,
        )
    )
    ax.set_xticks(range(n_columns))
    ax.set_xticklabels(column_labels, rotation=30, ha="right")
    ax.get_xticklabels()[0].set_fontweight("bold")
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(display_names)
    for tick, family in zip(ax.get_yticklabels(), families, strict=True):
        if family == "marin_dna":
            tick.set_fontweight("bold")
    for spine in ax.spines.values():
        spine.set_visible(False)

    colorbar = fig.colorbar(
        ScalarMappable(norm=norm, cmap=HEATMAP_CMAP),
        ax=ax,
        fraction=0.025,
        pad=0.02,
    )
    colorbar.set_label("AUPRC (%)")

    fig.tight_layout()
    axes_position = ax.get_position()
    colorbar_position = colorbar.ax.get_position()
    colorbar.ax.set_position(
        (
            colorbar_position.x0,
            axes_position.y0,
            colorbar_position.width,
            axes_position.height,
        )
    )
    return fig


# Leaderboard

ZERO_SHOT_DISPLAYS: frozenset[str] = frozenset(
    {
        "GPN-Star (M)",
        "AlphaGenome",
        "exp135-1B-m5.1",
        "Evo 2 (40B)",
        "Evo 2 (7B)",
        "Evo 2 (1B base)",
    }
)

PROBE_DISPLAYS: frozenset[str] = frozenset(
    {
        "exp135-1B-m5.1",
        "Evo 2 (40B)",
        "Evo 2 (7B)",
        "Evo 2 (1B base)",
    }
)

DISPLAY_OVERRIDES: dict[str, str] = {
    "exp135-1B-m5.1": "MarinDNA (1B/m5.1)",
}


def _table_for_displays(
    rows: pl.DataFrame,
    expected_displays: frozenset[str],
) -> pd.DataFrame:
    """Select and require the exact editorial model set for one panel."""
    table = table_from_normalized(rows)
    mask = table.index.get_level_values("method_display").isin(expected_displays)
    table = table[mask]
    actual_displays = frozenset(table.index.get_level_values("method_display"))
    assert actual_displays == expected_displays, (
        "leaderboard model set mismatch: "
        f"missing={sorted(expected_displays - actual_displays)}, "
        f"unexpected={sorted(actual_displays - expected_displays)}"
    )
    table.index = pd.MultiIndex.from_tuples(
        [
            (DISPLAY_OVERRIDES.get(display, display), family)
            for display, family in table.index
        ],
        names=table.index.names,
    )
    return table


def plot_mendelian_zero_shot_comparison() -> Figure:
    """Render the six canonical-protocol zero-shot headline rows."""
    rows = (
        local_data.normalized_rows("mendelian_traits", ZERO_SHOT_DISPLAYS)
        .with_columns(
            pl.col("family")
            .replace_strict(local_data.DEFAULT_PROTOCOL, default=None)
            .alias("_default")
        )
        .filter(pl.col("protocol") == pl.col("_default"))
    )
    return render_heatmap(
        _table_for_displays(rows, ZERO_SHOT_DISPLAYS),
    )


def plot_mendelian_probe_comparison() -> Figure:
    """Render only the four probe rows that overlap the zero-shot panel."""
    return render_heatmap(
        _table_for_displays(
            local_data.probe_normalized_rows("mendelian_traits", PROBE_DISPLAYS),
            PROBE_DISPLAYS,
        ),
    )


# Specialist

SPECIALIST_DATASET = "mendelian_traits"

SPECIALIST_SUBPLOT_HEIGHT_PX = 100.0

SPECIALIST_INK = "#1f1e1b"

AUPRC_BASELINE_PCT = 10.0

ROLE_ORDER = ("Specialist", "Evo 2 (40B)", "GPN-Star (M)")

BASELINE_COLORS = {
    "Evo 2 (40B)": "#999999",
    "GPN-Star (M)": "#555555",
}

SPECIALIST_REGION_COLORS = {"upstream": "#2f6f63", "cds": "#9c4f2f"}

PANEL_ORDER = (
    "tss_proximal",
    "5_prime_UTR_variant",
    "missense_variant",
    "splicing",
    "synonymous_variant",
)

PANEL_SPECS = {
    "5_prime_UTR_variant": {
        "specialist_id": "exp21-promoters-yolo-step-22000",
        "region": "upstream",
    },
    "tss_proximal": {
        "specialist_id": "exp21-promoters-yolo-step-22000",
        "region": "upstream",
    },
    "missense_variant": {
        "specialist_id": "exp27-cds-yolo-step-34000",
        "region": "cds",
    },
    "splicing": {
        "specialist_id": "exp27-cds-yolo-step-34000",
        "region": "cds",
    },
    "synonymous_variant": {
        "specialist_id": "exp27-cds-yolo-step-34000",
        "region": "cds",
    },
}

SPECIALIST_SUBSET_DISPLAY = {
    "5_prime_UTR_variant": "5′ UTR",
    "tss_proximal": "Promoter",
    "missense_variant": "Missense",
    "splicing": "Splicing",
    "synonymous_variant": "Synonymous",
}

BASELINE_IDS = {
    "Evo 2 (40B)": "evo2_40b",
    "GPN-Star (M)": "GPN-Star-M",
}


def load_plot_data() -> pd.DataFrame:
    """Load the five region-matched comparisons from current evals_v2 metrics."""
    required_ids = {
        *(spec["specialist_id"] for spec in PANEL_SPECS.values()),
        *BASELINE_IDS.values(),
    }
    models = {
        model.id: model
        for model in local_data.models_for_dataset(SPECIALIST_DATASET)
        if model.id in required_ids
    }
    assert set(models) == required_ids, (
        f"missing registered models: {sorted(required_ids - set(models))}"
    )

    records: list[dict[str, Any]] = []
    for subset in PANEL_ORDER:
        spec = PANEL_SPECS[subset]
        role_to_model_id = {
            "Specialist": spec["specialist_id"],
            **BASELINE_IDS,
        }
        for role, model_id in role_to_model_id.items():
            model = models[model_id]
            metrics = local_data.fetch_method_metrics(model, SPECIALIST_DATASET)
            row = metrics.filter(metrics["subset"] == subset)
            assert row.height == 1, (
                f"expected one {subset!r} row for {model_id!r}, got {row.height}"
            )
            values = row.row(0, named=True)
            value = float(values["value"])
            se = float(values["se"])
            n = int(values["n"])
            n_positives = int(values["n_positives"])
            assert 0.0 <= value <= 1.0
            assert 0.0 <= se <= 1.0
            assert 0 < n_positives <= n
            records.append(
                {
                    "subset": subset,
                    "consequence": SPECIALIST_SUBSET_DISPLAY[subset],
                    "region": spec["region"],
                    "role": role,
                    "model_id": model_id,
                    "protocol": local_data.DEFAULT_PROTOCOL[model.family],
                    "auprc_pct": 100.0 * value,
                    "se_pct": 100.0 * se,
                    "n": n,
                    "n_positives": n_positives,
                }
            )

    frame = pd.DataFrame.from_records(records)
    expected_rows = len(PANEL_ORDER) * len(ROLE_ORDER)
    assert len(frame) == expected_rows
    assert not frame.isna().any().any()
    return frame


def _specialist_draw_panel(ax: Axes, data: pd.DataFrame, subset: str) -> None:
    """Draw one consequence panel on its own y-axis."""
    assert set(data["subset"]) == {subset}
    assert data["n"].nunique() == 1
    region = str(data["region"].iloc[0])
    assert region in SPECIALIST_REGION_COLORS
    palette = {
        "Specialist": SPECIALIST_REGION_COLORS[region],
        **BASELINE_COLORS,
    }
    sns.barplot(
        data=data,
        x="role",
        y="auprc_pct",
        hue="role",
        order=ROLE_ORDER,
        hue_order=ROLE_ORDER,
        palette=palette,
        errorbar=None,
        edgecolor=SPECIALIST_INK,
        saturation=0.9,
        ax=ax,
        legend=False,
        dodge=False,
    )

    for x, role in enumerate(ROLE_ORDER):
        row = data.loc[data["role"] == role]
        assert len(row) == 1
        value = float(row["auprc_pct"].iloc[0])
        se = float(row["se_pct"].iloc[0])
        ax.errorbar(
            x,
            value,
            yerr=se,
            color=SPECIALIST_INK,
            capsize=0,
            zorder=5,
        )
    ax.set_axisbelow(True)
    ax.grid(False)
    ax.set_ylim(bottom=AUPRC_BASELINE_PCT)
    ax.set_xlabel("")
    ax.tick_params(axis="x", bottom=False, labelbottom=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_title(
        SPECIALIST_SUBSET_DISPLAY[subset],
        color=SPECIALIST_REGION_COLORS[region],
        pad=7,
    )
    ax.set_box_aspect(1)


def plot_specialists(data: pd.DataFrame) -> Figure:
    """Build five independently scaled consequence panels in a 2x3 grid."""
    mpl.rcdefaults()
    mpl.rcParams.update(
        {
            "svg.fonttype": "none",
            "svg.hashsalt": "region_specialist_performance",
            **matplotlib_typography_rcparams(),
            "axes.spines.top": False,
            "axes.spines.right": False,
            "text.color": SPECIALIST_INK,
            "axes.labelcolor": SPECIALIST_INK,
            "axes.edgecolor": SPECIALIST_INK,
            "xtick.color": SPECIALIST_INK,
            "ytick.color": SPECIALIST_INK,
            "axes.titlecolor": SPECIALIST_INK,
        }
    )
    layout_scale = SPECIALIST_SUBPLOT_HEIGHT_PX / 100.0
    figure, axes = plt.subplots(
        2,
        3,
        figsize=(5.0 * layout_scale, 3.2 * layout_scale),
        sharex=False,
        sharey=False,
    )
    panel_axes = (axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1], axes[1, 2])
    for ax, subset in zip(panel_axes, PANEL_ORDER, strict=True):
        _specialist_draw_panel(ax, data.loc[data["subset"] == subset], subset)
    axes[0, 2].axis("off")
    axes[0, 0].set_ylabel("AUPRC (%)")
    axes[1, 0].set_ylabel("AUPRC (%)")
    for ax in (axes[0, 1], axes[1, 1], axes[1, 2]):
        ax.set_ylabel("")

    specialist_handles = [
        Patch(
            facecolor=SPECIALIST_REGION_COLORS["upstream"],
            edgecolor=SPECIALIST_INK,
            label="Upstream",
        ),
        Patch(
            facecolor=SPECIALIST_REGION_COLORS["cds"],
            edgecolor=SPECIALIST_INK,
            label="CDS",
        ),
    ]
    generalist_handles = [
        Patch(
            facecolor=BASELINE_COLORS["Evo 2 (40B)"],
            edgecolor=SPECIALIST_INK,
            label="Evo 2 (40B)",
        ),
        Patch(
            facecolor=BASELINE_COLORS["GPN-Star (M)"],
            edgecolor=SPECIALIST_INK,
            label="GPN-Star (M)",
        ),
    ]
    specialist_legend = axes[0, 2].legend(
        handles=specialist_handles,
        title="Specialists",
        loc="upper left",
        bbox_to_anchor=(0.02, 1.22),
        ncol=1,
        frameon=False,
        handlelength=1.0,
        handletextpad=0.4,
        labelspacing=0.3,
        borderaxespad=0,
    )
    axes[0, 2].add_artist(specialist_legend)
    axes[0, 2].legend(
        handles=generalist_handles,
        title="Generalists",
        loc="upper left",
        bbox_to_anchor=(0.02, 0.68),
        ncol=1,
        frameon=False,
        handlelength=1.0,
        handletextpad=0.4,
        labelspacing=0.3,
        borderaxespad=0,
    )
    figure.subplots_adjust(
        left=0.105,
        right=0.995,
        top=0.96,
        bottom=0.12,
        wspace=0.25,
        hspace=0.32,
    )
    return figure


# Balance


class Arm(NamedTuple):
    prefix: str
    label: str
    upstream_fraction: float
    steps: tuple[int, ...]


ARMS: dict[str, Arm] = {
    "upstream_only": Arm(
        prefix="exp21-promoters-yolo",
        label="Upstream only (100 / 0)",
        upstream_fraction=1.0,
        steps=(2000, 6000, 10000, 12000, 14000, 16000, 18000, 20000, 22000),
    ),
    "balanced": Arm(
        prefix="exp13-mixture-equal",
        label="Uniform mix (50 / 50)",
        upstream_fraction=0.5,
        steps=(2000, 6000, 10000, 14000, 18000, 22000, 26000),
    ),
    "proportional": Arm(
        prefix="exp13-mixture-proportional",
        label="Proportional mix (10 / 90)",
        upstream_fraction=0.1,
        steps=(2000, 6000, 10000, 14000, 18000, 22000, 26000),
    ),
    "cds_only": Arm(
        prefix="exp27-cds-yolo",
        label="CDS only (0 / 100)",
        upstream_fraction=0.0,
        steps=(2000, 6000, 10000, 14000, 18000, 22000, 26000, 34000),
    ),
}

S3_BASE = "s3://oa-bolinas/snakemake/analysis/evals_v2/results/metrics"

SCORE_TYPE = "minus_llr_avg"

BALANCE_SUBSETS: dict[str, str] = {
    "Promoter": "tss_proximal",
    "Missense": "missense_variant",
}

BALANCE_SUBPLOT_HEIGHT_PX = 120.0

TEXT_COLOR = "#1f1e1b"

BALANCE_REGION_COLORS = {"upstream": "#2f6f63", "cds": "#9c4f2f"}

TITLE_COLORS = {
    "Promoter": BALANCE_REGION_COLORS["upstream"],
    "Missense": BALANCE_REGION_COLORS["cds"],
    "Mean of both": TEXT_COLOR,
}


def blend_region_colors(upstream_fraction: float) -> tuple[float, float, float]:
    """Blend the issue #370 Upstream and CDS colors by mixture fraction."""
    assert 0.0 <= upstream_fraction <= 1.0
    upstream = mpl.colors.to_rgb(BALANCE_REGION_COLORS["upstream"])
    cds = mpl.colors.to_rgb(BALANCE_REGION_COLORS["cds"])
    return tuple(
        upstream_fraction * upstream_channel + (1.0 - upstream_fraction) * cds_channel
        for upstream_channel, cds_channel in zip(upstream, cds, strict=True)
    )


class HandlerLineMarkerWithPie(HandlerBase):
    """Draw a line marker followed by an upstream/CDS composition pie."""

    def __init__(self, upstream_fraction: float, **kwargs) -> None:
        super().__init__(**kwargs)
        assert 0.0 <= upstream_fraction <= 1.0
        self.upstream_fraction = upstream_fraction

    def create_artists(
        self,
        legend,
        orig_handle,
        xdescent,
        ydescent,
        width,
        height,
        fontsize,
        trans,
    ):
        pie_diameter = height
        gap = 0.35 * fontsize
        line_y = height / 2 - ydescent
        line_x0 = -xdescent
        line_x1 = -xdescent + width - pie_diameter - gap

        color = orig_handle.get_color()
        line = Line2D(
            [line_x0, line_x1],
            [line_y, line_y],
            color=color,
            linewidth=orig_handle.get_linewidth(),
            solid_capstyle="butt",
        )
        marker = Line2D(
            [(line_x0 + line_x1) / 2],
            [line_y],
            color=color,
            marker=orig_handle.get_marker(),
            markersize=orig_handle.get_markersize(),
            markeredgecolor=TEXT_COLOR,
            markeredgewidth=orig_handle.get_markeredgewidth(),
            linestyle="None",
        )
        line.set_transform(trans)
        marker.set_transform(trans)
        artists = [line, marker]

        center_x = -xdescent + width - pie_diameter / 2
        radius = pie_diameter / 2 * 0.92
        upstream_fraction = self.upstream_fraction
        if upstream_fraction in (0.0, 1.0):
            region = "upstream" if upstream_fraction == 1.0 else "cds"
            circle = Circle(
                (center_x, line_y),
                radius,
                facecolor=BALANCE_REGION_COLORS[region],
                edgecolor=TEXT_COLOR,
            )
            circle.set_transform(trans)
            artists.append(circle)
        else:
            theta_top = 90.0
            theta_start = theta_top - upstream_fraction * 360.0
            upstream_wedge = Wedge(
                (center_x, line_y),
                radius,
                theta_start,
                theta_top,
                facecolor=BALANCE_REGION_COLORS["upstream"],
                edgecolor=TEXT_COLOR,
            )
            cds_wedge = Wedge(
                (center_x, line_y),
                radius,
                theta_top,
                theta_top + (1.0 - upstream_fraction) * 360.0,
                facecolor=BALANCE_REGION_COLORS["cds"],
                edgecolor=TEXT_COLOR,
            )
            upstream_wedge.set_transform(trans)
            cds_wedge.set_transform(trans)
            artists.extend((upstream_wedge, cds_wedge))
        return artists


def apply_style() -> None:
    """Apply the compact Open Athena blog style used by the draft."""
    mpl.rcParams.update(
        {
            "svg.fonttype": "none",
            "svg.hashsalt": Path(__file__).stem,
            "font.family": "sans-serif",
            **matplotlib_typography_rcparams(),
            "axes.edgecolor": TEXT_COLOR,
            "axes.labelcolor": TEXT_COLOR,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "none",
            "axes.facecolor": "none",
            "xtick.color": TEXT_COLOR,
            "ytick.color": TEXT_COLOR,
            "legend.frameon": False,
            "savefig.facecolor": "none",
            "savefig.transparent": True,
        }
    )


def load_trajectories() -> pl.DataFrame:
    """Load all available checkpoints for the four experiment arms."""
    parts: list[pl.DataFrame] = []
    missing: list[str] = []
    total = sum(len(arm.steps) for arm in ARMS.values())
    for arm_key, arm in ARMS.items():
        for step in arm.steps:
            uri = f"{S3_BASE}/{arm.prefix}-step-{step}/mendelian_traits.parquet"
            try:
                checkpoint = local_data.read_polars(uri)
            except local_data.MissingData:
                raise
            except Exception as exc:
                missing.append(f"  {arm_key} step-{step}: {exc}")
                continue
            parts.append(
                checkpoint.with_columns(
                    pl.lit(arm_key).alias("arm"),
                    pl.lit(step).alias("step"),
                )
            )
    if missing:
        print(
            f"WARNING: {len(missing)} of {total} parquets unreadable:\n"
            + "\n".join(missing),
            file=sys.stderr,
        )
    assert parts, "no metrics parquets loaded"
    return pl.concat(parts)


def shared_step_data(trajectories: pl.DataFrame) -> pl.DataFrame:
    """Select the two focal subsets at steps represented in every arm."""
    common_steps = set.intersection(*(set(arm.steps) for arm in ARMS.values()))
    assert common_steps == {2000, 6000, 10000, 14000, 18000, 22000}

    filtered = trajectories.filter(
        (pl.col("score_type") == SCORE_TYPE)
        & (pl.col("subset").is_in(list(BALANCE_SUBSETS.values())))
        & (pl.col("step").is_in(sorted(common_steps)))
    )
    expected_rows = len(ARMS) * len(BALANCE_SUBSETS) * len(common_steps)
    assert filtered.height == expected_rows, (
        f"expected {expected_rows} arm/subset/step rows, got {filtered.height}"
    )
    assert filtered.select("arm", "subset", "step").unique().height == expected_rows
    assert filtered["value"].is_not_null().all()
    assert filtered["value"].is_between(0.0, 1.0).all()
    return filtered


def plot_balance(data: pl.DataFrame) -> plt.Figure:
    """Create upstream, missense, and two-subset-mean trajectory panels."""
    apply_style()
    axes_height_fraction = 0.70 - 0.09
    figure_height_inches = BALANCE_SUBPLOT_HEIGHT_PX / (
        72.0 * axes_height_fraction * FIGURE_GLOBAL_RENDER_SCALE
    )
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(2.5 * figure_height_inches, figure_height_inches),
        sharex=True,
        sharey=False,
    )

    for axis, (title, subset) in zip(axes[:2], BALANCE_SUBSETS.items(), strict=True):
        for arm_key, arm in ARMS.items():
            arm_data = data.filter(
                (pl.col("arm") == arm_key) & (pl.col("subset") == subset)
            ).sort("step")
            axis.plot(
                arm_data["step"].to_numpy(),
                arm_data["value"].to_numpy() * 100.0,
                marker="o",
                color=blend_region_colors(arm.upstream_fraction),
                label=arm.label,
            )
        axis.set_title(
            title,
            color=TITLE_COLORS[title],
        )

    mean_axis = axes[2]
    for arm_key, arm in ARMS.items():
        arm_data = (
            data.filter(pl.col("arm") == arm_key)
            .group_by("step")
            .agg(pl.col("value").mean().alias("value"))
            .sort("step")
        )
        assert arm_data.height == data["step"].n_unique()
        mean_axis.plot(
            arm_data["step"].to_numpy(),
            arm_data["value"].to_numpy() * 100.0,
            marker="o",
            color=blend_region_colors(arm.upstream_fraction),
            label=arm.label,
        )
    mean_axis.set_title(
        "Mean of both",
        color=TITLE_COLORS["Mean of both"],
    )

    for axis in axes:
        axis.set_xlabel("Training step")
        axis.grid(False)
        axis.set_box_aspect(1)
    axes[0].set_ylabel("AUPRC (%)")

    handles, labels = axes[0].get_legend_handles_labels()
    # Matplotlib fills a two-column legend column-first. This ordering displays
    # the two pure-region arms on the first row and the two mixtures below.
    legend_order = (0, 1, 3, 2)
    handler_map = {
        handle: HandlerLineMarkerWithPie(arm.upstream_fraction)
        for handle, arm in zip(handles, ARMS.values(), strict=True)
    }
    fig.legend(
        [handles[index] for index in legend_order],
        [labels[index] for index in legend_order],
        title="Training data mixture",
        handler_map=handler_map,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.08),
        ncol=2,
        alignment="center",
        columnspacing=0.4,
        handlelength=4.0,
        handletextpad=0.4,
        labelspacing=0.25,
        borderpad=0.0,
    )
    fig.subplots_adjust(bottom=0.09, top=0.70, left=0.09, right=0.98, wspace=0.28)
    return fig


# Overview

OVERVIEW_INK = "#465e70"

GRAY = "#817866"

COLORS = ["#465e70", "#a54e2a", "#2b766b", "#8a7090", "#a17d2f"]


def overview_canvas(height):
    fig, ax = plt.subplots(figsize=(7, height / 100))
    fig.subplots_adjust(0, 0, 1, 1)
    ax.set(xlim=(0, 700), ylim=(height, 0))
    ax.axis("off")
    return fig, ax


def overview_text(ax, x, y, value, size=14, color="#252522", align="left", bold=False):
    return ax.text(
        x,
        y,
        value,
        fontsize=size,
        color=color,
        ha=align,
        va="center",
        weight="bold" if bold else "normal",
    )


def overview_arrow(ax, start, end, color=GRAY):
    ax.add_patch(
        FancyArrowPatch(
            start, end, arrowstyle="-|>", mutation_scale=10, lw=1, color=color
        )
    )


def overview_box(ax, x, y, w, h, edge=OVERVIEW_INK, fill="white"):
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0,rounding_size=5",
            edgecolor=edge,
            facecolor=fill,
            linewidth=0.8,
        )
    )


def plot_multispecies_training_sequences():
    fig, ax = overview_canvas(245)
    overview_text(ax, 15, 22, "Multiple species", bold=True)
    overview_text(ax, 245, 22, "Select regions", bold=True)
    overview_text(ax, 510, 22, "Sample sequences", bold=True)
    for i, seq in enumerate(("ACGTAC…", "TGCACT…", "GACTGA…")):
        y = 67 + i * 50
        overview_text(ax, 15, y, f"Species {i + 1}", size=12, color=GRAY)
        overview_text(ax, 15, y + 20, seq, size=17, color=OVERVIEW_INK)
    for i, name in enumerate(
        ("CDS", "Upstream", "Downstream", "ncRNA", "Enhancer regions")
    ):
        y = 54 + i * 31
        overview_box(ax, 245, y, 178, 26, edge=COLORS[i])
        overview_text(ax, 256, y + 13, name, color=COLORS[i])
    overview_arrow(ax, (170, 128), (231, 128))
    overview_arrow(ax, (435, 128), (493, 128))
    for i, (seq, color) in enumerate(
        zip(("ACGTAC…", "GACTGA…", "TGCACT…"), (COLORS[0], COLORS[3], COLORS[4]))
    ):
        y = 63 + i * 49
        overview_box(ax, 510, y, 164, 33, edge=color)
        overview_text(ax, 592, y + 16.5, seq, size=17, color=color, align="center")
    overview_text(ax, 592, 225, "255 bases per example", size=12, align="center")
    return fig


def plot_next_base_prediction():
    fig, ax = overview_canvas(285)
    overview_text(
        ax,
        350,
        20,
        "Predict each base from preceding bases",
        size=15,
        align="center",
        bold=True,
    )
    xs = [105, 160, 215, 270, 325]
    overview_text(ax, 12, 70, "Target", size=12, color=GRAY)
    overview_text(ax, 12, 204, "Input", size=12, color=GRAY)
    for x, inp, target in zip(
        xs, ("BOS", "A", "C", "G", "…"), ("A", "C", "G", "T", "…")
    ):
        overview_text(ax, x, 70, target, size=17, align="center", color="#a54e2a")
        overview_text(ax, x, 204, inp, size=15, align="center", color=OVERVIEW_INK)
        overview_arrow(ax, (x, 187), (x, 160), color=OVERVIEW_INK)
        overview_arrow(ax, (x, 115), (x, 88), color="#a54e2a")
    overview_box(ax, 68, 115, 297, 45, fill="#f3f6f8")
    overview_text(
        ax,
        216.5,
        137.5,
        "Qwen3 causal Transformer",
        size=14,
        bold=True,
        align="center",
        color=OVERVIEW_INK,
    )
    ax.plot([365, 410, 410], [137.5, 137.5, 100], color=GRAY, lw=1)
    ax.plot([410, 410], [137.5, 177], color=GRAY, lw=1)
    overview_arrow(ax, (410, 100), (450, 100))
    overview_arrow(ax, (410, 177), (450, 177))
    overview_text(ax, 463, 100, "Sequence\nprobabilities", color="#a54e2a")
    overview_text(ax, 463, 177, "Learned\nrepresentations", color="#2b766b")
    overview_text(
        ax,
        216,
        232,
        "255 DNA bases + BOS = 256 tokens",
        size=12,
        color=GRAY,
        align="center",
    )
    overview_text(
        ax,
        350,
        265,
        "DNA training, without variant-effect labels",
        size=13,
        align="center",
    )
    return fig


# Command-line interface and publication output

PAPER_STYLE = {
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
    "mathtext.fontset": "dejavusans",
    "svg.fonttype": "none",
    **matplotlib_typography_rcparams(),
    "axes.spines.top": False,
    "axes.spines.right": False,
    **{
        key: "#1f1e1b"
        for key in (
            "text.color",
            "axes.labelcolor",
            "axes.edgecolor",
            "xtick.color",
            "ytick.color",
            "axes.titlecolor",
        )
    },
}

OVERVIEW_STYLE = {
    **PAPER_STYLE,
    "font.family": "Lato",
    "font.size": 14,
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "text.color": "#252522",
    "svg.hashsalt": "marin-overview",
}


def _publish(staging: Path, output_dir: Path, filenames: list[str]) -> None:
    """Check the complete render before replacing any publication output."""
    for filename in filenames:
        if not (staging / filename).is_file():
            raise RuntimeError(f"Figure did not produce {filename}")
    for filename in filenames:
        (staging / filename).replace(output_dir / filename)


def render(output_dir: Path, name: str, draw, *, overview=False, reports=()) -> None:
    """Render into a temporary directory and publish only after saving succeeds.

    ``draw(staging)`` returns a Figure and may write the named reports there.
    Every command has its own style context, including repeated CLI invocations.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        with tempfile.TemporaryDirectory(
            prefix=".render-", dir=output_dir
        ) as temporary:
            staging = Path(temporary)
            with mpl.rc_context():
                mpl.rcdefaults()
                mpl.rcParams.update(OVERVIEW_STYLE if overview else PAPER_STYLE)
                if overview:
                    for font in Path("/usr/share/fonts/truetype/lato").glob("*.ttf"):
                        fontManager.addfont(font)
                else:
                    mpl.rcParams["svg.hashsalt"] = name
                try:
                    figure = draw(staging)
                    for extension in ("pdf", "svg"):
                        metadata = (
                            {"CreationDate": None, "ModDate": None}
                            if extension == "pdf"
                            else {"Date": None}
                        )
                        options = (
                            {}
                            if overview
                            else {"bbox_inches": "tight", "transparent": True}
                        )
                        path = staging / f"{name}.{extension}"
                        figure.savefig(path, metadata=metadata, **options)
                        if extension == "svg" and not overview:
                            path.write_text(
                                "\n".join(
                                    line.rstrip()
                                    for line in path.read_text().splitlines()
                                )
                                + "\n"
                            )
                            normalize_matplotlib_svg_typography_file(path)
                    _publish(
                        staging, output_dir, [f"{name}.pdf", f"{name}.svg", *reports]
                    )
                finally:
                    plt.close("all")
    except (local_data.MissingData, FileNotFoundError) as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(f"Built {name}")


def render_artwork(output_dir: Path, name: str) -> None:
    """Convert an existing figures/<name>.svg to PDF."""
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        source = local_data.local_path(
            Path(__file__).resolve().parents[1] / "figures" / f"{name}.svg"
        )
        with tempfile.TemporaryDirectory(
            prefix=".render-", dir=output_dir
        ) as temporary:
            staging = Path(temporary)
            cairosvg.svg2pdf(url=str(source), write_to=str(staging / f"{name}.pdf"))
            _publish(staging, output_dir, [f"{name}.pdf"])
    except (local_data.MissingData, FileNotFoundError) as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(f"Built {name}")


def _transfer_figure(draw) -> Figure:
    frame = local_data.load_transfer()
    params = sorted({int(p) for p in frame["params"].dropna().unique()})
    return draw(frame, palette(params), params)


def _loss_scaling_figure(staging: Path) -> Figure:
    results = local_data.load_scaling_results()
    params = sorted({int(p) for p in results["params"].dropna().unique()})
    return plot_loss_scaling(
        local_data.load_scaling_history(),
        results,
        palette(params),
        staging / "sequence_loss_scaling_fit.txt",
    )


@click.group()
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path, file_okay=False),
    default=Path(__file__).resolve().parents[1] / "figures",
    show_default=True,
)
@click.option(
    "--fetch",
    is_flag=True,
    help="Retrieve missing snapshots using configured credentials.",
)
@click.pass_context
def cli(ctx: click.Context, output_dir: Path, fetch: bool) -> None:
    """Generate a paper figure by its manuscript filename (without extension).

    Plots write PDF and SVG; existing SVG diagrams are converted to PDF.
    Builds are offline unless --fetch or
    FIGURE_FETCH=1 is explicitly supplied. Use Make for batches of figures.
    """
    previous = local_data.FETCH
    local_data.FETCH = fetch
    ctx.call_on_close(lambda: setattr(local_data, "FETCH", previous))
    ctx.obj = output_dir


@cli.command("learning_rate_transfer")
@click.pass_obj
def learning_rate_transfer(output_dir: Path) -> None:
    """Transfer-validation loss versus learning rate."""
    render(
        output_dir,
        "learning_rate_transfer",
        lambda _: _transfer_figure(plot_learning_rate),
    )


@cli.command("optimizer_parameter_transfer")
@click.pass_obj
def optimizer_parameter_transfer(output_dir: Path) -> None:
    """Transfer-validation loss versus optimizer parameters."""
    render(
        output_dir,
        "optimizer_parameter_transfer",
        lambda _: _transfer_figure(plot_optimizer),
    )


@cli.command("regional_hyperparameter_transfer")
@click.pass_obj
def regional_hyperparameter_transfer(output_dir: Path) -> None:
    """Hyperparameter transfer across genomic regions."""
    render(
        output_dir,
        "regional_hyperparameter_transfer",
        lambda _: _transfer_figure(plot_regional),
    )


@cli.command("sequence_loss_scaling")
@click.pass_obj
def sequence_loss_scaling(output_dir: Path) -> None:
    """Training/validation loss scaling and the Kaplan fit report."""
    render(
        output_dir,
        "sequence_loss_scaling",
        _loss_scaling_figure,
        reports=("sequence_loss_scaling_fit.txt",),
    )


@cli.command("vep_performance_vs_throughput")
@click.pass_obj
def vep_performance_vs_throughput(output_dir: Path) -> None:
    """Historical VEP performance and inference throughput."""
    render(
        output_dir,
        "vep_performance_vs_throughput",
        lambda _: plot_throughput(
            local_data.load_leaderboard(), local_data.load_inference_costs()
        ),
    )


@cli.command("vep_by_model_size")
@click.pass_obj
def vep_by_model_size(output_dir: Path) -> None:
    """Paired endpoint AUPRC versus model size; requires external predictions."""
    render(output_dir, "vep_by_model_size", lambda _: plot_scaling())


@cli.command("vep_by_sequence_likelihood")
@click.pass_obj
def vep_by_sequence_likelihood(output_dir: Path) -> None:
    """Paired endpoint AUPRC versus sequence loss; requires external predictions."""
    render(output_dir, "vep_by_sequence_likelihood", lambda _: plot_likelihood())


@cli.command("missense_readout_scaling")
@click.pass_obj
def missense_readout_scaling(output_dir: Path) -> None:
    """MarinDNA/Evo 2 missense readouts; requires external predictions/metrics."""
    render(output_dir, "missense_readout_scaling", lambda _: plot_missense())


@cli.command("mixture_zero_shot_trajectories")
@click.pass_obj
def mixture_zero_shot_trajectories(output_dir: Path) -> None:
    """Mixture-lineage zero-shot trajectories; requires external metrics."""
    render(output_dir, "mixture_zero_shot_trajectories", lambda _: plot_lineage("llr"))


@cli.command("mixture_probe_trajectories")
@click.pass_obj
def mixture_probe_trajectories(output_dir: Path) -> None:
    """Mixture-lineage probe trajectories; requires external metrics."""
    render(output_dir, "mixture_probe_trajectories", lambda _: plot_lineage("probe"))


@cli.command("mendelian_zero_shot_comparison")
@click.pass_obj
def mendelian_zero_shot_comparison(output_dir: Path) -> None:
    """Mendelian zero-shot leaderboard; requires external metrics."""
    render(
        output_dir,
        "mendelian_zero_shot_comparison",
        lambda _: plot_mendelian_zero_shot_comparison(),
    )


@cli.command("mendelian_probe_comparison")
@click.pass_obj
def mendelian_probe_comparison(output_dir: Path) -> None:
    """Mendelian probe leaderboard; requires external metrics."""
    render(
        output_dir,
        "mendelian_probe_comparison",
        lambda _: plot_mendelian_probe_comparison(),
    )


@cli.command("region_specialist_performance")
@click.pass_obj
def region_specialist_performance(output_dir: Path) -> None:
    """Region-specialist comparisons; requires external metrics."""
    render(
        output_dir,
        "region_specialist_performance",
        lambda _: plot_specialists(load_plot_data()),
    )


@cli.command("upstream_coding_mixture_balance")
@click.pass_obj
def upstream_coding_mixture_balance(output_dir: Path) -> None:
    """Equal-step upstream/CDS mixture trajectories; requires external metrics."""
    render(
        output_dir,
        "upstream_coding_mixture_balance",
        lambda _: plot_balance(shared_step_data(load_trajectories())),
    )


@cli.command("multispecies_training_sequences")
@click.pass_obj
def multispecies_training_sequences(output_dir: Path) -> None:
    """Multispecies training-sequence schematic."""
    render(
        output_dir,
        "multispecies_training_sequences",
        lambda _: plot_multispecies_training_sequences(),
        overview=True,
    )


@cli.command("next_base_prediction")
@click.pass_obj
def next_base_prediction(output_dir: Path) -> None:
    """Next-base-prediction schematic."""
    render(
        output_dir,
        "next_base_prediction",
        lambda _: plot_next_base_prediction(),
        overview=True,
    )


@cli.command("training_dataset_provenance")
@click.pass_obj
def training_dataset_provenance(output_dir: Path) -> None:
    """Convert the authored training-dataset provenance diagram."""
    render_artwork(output_dir, "training_dataset_provenance")


@cli.command("evaluation_benchmarks")
@click.pass_obj
def evaluation_benchmarks(output_dir: Path) -> None:
    """Convert the authored evaluation-benchmarks diagram."""
    render_artwork(output_dir, "evaluation_benchmarks")


@cli.command("variant_scoring_workflow")
@click.pass_obj
def variant_scoring_workflow(output_dir: Path) -> None:
    """Convert the authored variant-scoring workflow."""
    render_artwork(output_dir, "variant_scoring_workflow")


@cli.command("annotation_derived_training_pool")
@click.pass_obj
def annotation_derived_training_pool(output_dir: Path) -> None:
    """Convert the authored annotation-derived training-pool diagram."""
    render_artwork(output_dir, "annotation_derived_training_pool")


@cli.command("hyperparameter_transfer_workflow")
@click.pass_obj
def hyperparameter_transfer_workflow(output_dir: Path) -> None:
    """Convert the authored hyperparameter-transfer workflow."""
    render_artwork(output_dir, "hyperparameter_transfer_workflow")


@cli.command("training_mixture_histories")
@click.pass_obj
def training_mixture_histories(output_dir: Path) -> None:
    """Convert the authored training-mixture histories."""
    render_artwork(output_dir, "training_mixture_histories")


if __name__ == "__main__":
    cli()
