from numbers import Number

import ipywidgets as widgets
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from IPython.display import display
from ipywidgets import Button, HBox, VBox, Output
from matplotlib.backend_bases import MouseEvent
from matplotlib.lines import Line2D


def print_dataset_variables_summary(ds: xr.Dataset):
    print(f"{'Variable':<20}{'Long name':<50} {'Units':<10}")
    print("-" * 80)
    for var in ds:
        if var == "grid_mapping_1":
            continue
        attrs = ds[var].attrs
        print(f"{var:<20}{attrs['long_name']:<50} {attrs['units']:<10}")


def truncate_cmap(cmap, start=0.0, stop=1.0, n=256) -> mcolors.LinearSegmentedColormap:
    cmap = plt.get_cmap(cmap)
    new_colors = cmap(np.linspace(start, stop, n))
    return mcolors.LinearSegmentedColormap.from_list("trunc_cmap", new_colors)


def estimate_topography_from_dataset(ds: xr.Dataset) -> xr.Dataset:
    topography = (~xr.ufuncs.isnan(ds["P"])).idxmax(dim="z_1")
    topography.attrs = {"units": "m", "long_name": "Topography"}
    return topography


def plot_cross_section_across_dim(
    ds: xr.Dataset,
    dim: str,
    value: Number,
    var: str, var_contour: str | None = None,
    plot_kwargs: dict | None = None,
    figsize: tuple[int, int] = (12, 6),
) -> plt.Figure:
    title_suffix = ""

    if dim == "x_1":
        ds_cross_section = ds.sel(x_1=value)
        x_label = "y_1"
        y_label = "z_1"
    elif dim == "y_1":
        ds_cross_section = ds.sel(y_1=value)
        x_label = "x_1"
        y_label = "z_1"
    elif dim == "z_1":
        ds_cross_section = ds.sel(z_1=value)
        title_suffix = f" at z={value:.0f} m"
        x_label = "x_1"
        y_label = "y_1"
    else:
        raise ValueError(f"Invalid dim: {dim}")

    return _plot_cross_section(ds_cross_section, x_label, y_label, var, var_contour, plot_kwargs, figsize, title_suffix)


def plot_cross_section_between_two_points(
    ds: xr.Dataset,
    point1: tuple[Number, Number],
    point2: tuple[Number, Number],
    var: str, var_contour: str | None = None,
    plot_kwargs: dict | None = None,
    figsize: tuple[int, int] = (12, 6),
) -> plt.Figure:
    x1, y1 = point1
    x2, y2 = point2

    x_step = np.median(np.diff(ds.coords["x_1"]))
    y_step = np.median(np.diff(ds.coords["y_1"]))

    n = int(max(abs(x2 - x1) / x_step, abs(y2 - y1) / y_step))

    t = np.linspace(0, 1, n)
    xs = x1 + t * (x2 - x1)
    ys = y1 + t * (y2 - y1)

    points_label = "points"
    ds_cross_section = ds.interp(x_1=(points_label, xs), y_1=(points_label, ys), method="linear")

    dist = np.sqrt((xs - xs[0]) ** 2 + (ys - ys[0]) ** 2) / 1000
    ds_cross_section = ds_cross_section.assign_coords({points_label: dist})
    ds_cross_section.coords[points_label].attrs = dict(units="km", long_name="Distance along section")

    return _plot_cross_section(ds_cross_section, points_label, "z_1", var, var_contour, plot_kwargs, figsize,
                               title_suffix="")


def _plot_cross_section(
    ds_cross_section: xr.Dataset,
    x_label: str, y_label: str,
    var: str, var_contour: str | None,
    plot_kwargs: dict | None,
    figsize: tuple[int, int],
    title_suffix: str,
) -> plt.Figure:
    if plot_kwargs is None:
        plot_kwargs = {}

    fig, ax = plt.subplots(figsize=figsize)
    ds_cross_section[var].plot(x=x_label, y=y_label, ax=ax, **plot_kwargs)
    title = f"{var} (shaded)"

    if var_contour is not None:
        title += f" + {var_contour} (contours)"
        cs = ds_cross_section[var_contour].plot.contour(x=x_label, y=y_label, ax=ax, colors="black", linewidths=0.5)
        ax.clabel(cs, inline=True, fontsize=10, fmt="%.1f")

    title += title_suffix
    title += f"\n{ds_cross_section["time"].values.astype("datetime64[s]")}"
    plt.title(title)
    plt.close(fig)

    return fig


def mark_and_store_points_onclick(event: MouseEvent, coord_label: widgets.Label,
                                  ds: xr.Dataset,
                                  markers: list[Line2D],
                                  line: Line2D | None,
                                  selected_points: list[tuple[Number, Number]]):
    ax = event.inaxes

    x = xr.ufuncs.abs(ds["x_1"] - event.xdata).idxmin().item()
    y = xr.ufuncs.abs(ds["y_1"] - event.ydata).idxmin().item()

    # If 2 points already exist, reset everything on a 3rd click
    if len(selected_points) == 2:
        selected_points = []
        coord_label.value = "Click two points..."

        # Clear markers
        for m in markers:
            m.remove()
        markers = []

        # Clear line
        if line is not None:
            line.remove()
            line = None

    # Add the selected point
    selected_points.append((x, y))
    coord_label.value = f"Selected points: {selected_points}"

    # Draw the red cross marker
    marker = ax.plot(x, y, marker="x", color="red", markersize=5, mew=2)[0]
    markers.append(marker)

    # If this is the second point, draw the connecting line
    if len(selected_points) == 2:
        (x1, y1), (x2, y2) = selected_points
        line = ax.plot([x1, x2], [y1, y2], color="black", linewidth=2)[0]

    return selected_points, markers, line


def plot_figures(figures: list[plt.Figure]):
    out = Output()
    index = 0

    def show_fig(i):
        with out:
            out.clear_output(wait=True)
            display(figures[i])

    def prev(_):
        nonlocal index
        index = (index - 1) % len(figures)
        show_fig(index)

    def next(_):
        nonlocal index
        index = (index + 1) % len(figures)
        show_fig(index)

    prev_btn = Button(description="⟨ Prev")
    next_btn = Button(description="Next ⟩")

    prev_btn.on_click(prev)
    next_btn.on_click(next)

    display(VBox([HBox([prev_btn, next_btn]), out]))
    show_fig(index)
