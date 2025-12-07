import datetime
import pathlib
from numbers import Number

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import imageio.v2 as imageio
import ipywidgets as widgets
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from IPython.display import display
from ipywidgets import Button, HBox, VBox, Output
from matplotlib.backend_bases import MouseEvent
from matplotlib.lines import Line2D


WIND_U = "wind_u"
WIND_V = "wind_v"


class GIFWriter:
    def __init__(self):
        self._paths: list[pathlib.Path] = []

    def add_path(self, path: pathlib.Path):
        self._paths.append(path)

    def build_gif(self, output_path: pathlib.Path | None = None):
        if output_path is None:
            if len(self._paths) == 0:
                return
            output_path = self._paths[0]
            output_path = output_path.parent / (output_path.stem + ".gif")

        images = []
        for path in self._paths:
            img = imageio.imread(path)
            images.append(img)
        imageio.mimsave(output_path, images, duration=250)


def calculate_wind_uv(ds: xr.Dataset) -> xr.Dataset:
    ds = ds.assign(
        U=-ds.FF * xr.ufuncs.cos(xr.ufuncs.deg2rad(90 - ds.DD)),
        V=-ds.FF * xr.ufuncs.sin(xr.ufuncs.deg2rad(90 - ds.DD)),
    )
    ds.U.attrs = {"units": ds.FF.attrs["units"], "long_name": ds.FF.attrs["long_name"] + ", x direction"}
    ds.V.attrs = {"units": ds.FF.attrs["units"], "long_name": ds.FF.attrs["long_name"] + ", y direction"}

    return ds


def calculate_wind_speed_in_direction(ds: xr.Dataset, direction_x: float, direction_y: float) -> xr.Dataset:
    return (ds.U * direction_x + ds.V * direction_y) / np.sqrt(direction_x ** 2 + direction_y ** 2)


def print_dataset_variables_summary(ds: xr.Dataset):
    print(f"{'Variable':<20}{'Long name':<50} {'Units':<10}")
    print("-" * 80)
    for var in ds:
        if var == "grid_mapping_1":
            continue
        attrs = ds[var].attrs
        print(f"{var:<20}{attrs['long_name']:<50} {attrs['units']:<10}")


def get_swiss_projection():
    return ccrs.epsg(21781)  # Oblique Mercator (CH1903)


def truncate_cmap(cmap, start=0.0, stop=1.0, n=256) -> mcolors.LinearSegmentedColormap:
    cmap = plt.get_cmap(cmap)
    new_colors = cmap(np.linspace(start, stop, n))
    return mcolors.LinearSegmentedColormap.from_list("trunc_cmap", new_colors)

def get_cmap_for_variable(var: str) -> mcolors.Colormap | str | None:
    match var:
        # DD
        case "v_on_z":
            return "seismic"
        case "RELHUM":
            return "YlGn"
        # P
        # QI
        # QR
        # QS
        # QC
        # QV
        # HZEROCL
        # TD
        # THETA
        # TOT_PREC
        # T
        # U
        # V
    return None


def estimate_topography_from_dataset(ds: xr.Dataset) -> xr.Dataset:
    topography = (~xr.ufuncs.isnan(ds["P"])).idxmax(dim="z_1")
    topography.attrs = {"units": "m", "long_name": "Topography"}
    return topography


class Plotter:
    def __init__(self, ds_list: list[xr.Dataset]):
        self._ds_min = xr.concat(ds_list, dim="tmp").min(dim="tmp")
        self._ds_max = xr.concat(ds_list, dim="tmp").max(dim="tmp")

    def plot_cross_section_across_dim(
        self,
        ds: xr.Dataset,
        dim: str | None,
        value: Number,
        var: str, var_contour: str | None = None, with_wind: bool = False,
        plot_kwargs: dict | None = None,
        gif_writer: GIFWriter | None = None,
        figsize: tuple[int, int] = (12, 6),
    ) -> plt.Figure:
        title_suffix = ""

        subplot_kwargs = {}
        axes_features = []

        no_third_dim = len(ds[var].dims) < 3
        if no_third_dim:
            dim = None
            value = None
            ds_cross_section = ds
            assert "x_1" in ds.dims and "y_1" in ds.dims, (f"TODO: support plotting dataset with "
                                                           f"{[dim for dim in ds.dims]} dimensions")
        else:
            ds_cross_section = ds.sel(**{dim: value})

        if plot_kwargs is None:
            plot_kwargs = {}

        if "norm" not in plot_kwargs:
            da_min = self._ds_min[var] if no_third_dim else self._ds_min[var].sel(**{dim: value})
            da_max = self._ds_max[var] if no_third_dim else self._ds_max[var].sel(**{dim: value})
            plot_kwargs["norm"] = _build_norm(da_min, da_max)

        if dim == "x_1":
            x_label = "y_1"
            y_label = "z_1"
            ds_cross_section = ds_cross_section.assign({WIND_U: ds_cross_section.U, WIND_V: ds_cross_section.v_on_z})
        elif dim == "y_1":
            x_label = "x_1"
            y_label = "z_1"
            ds_cross_section = ds_cross_section.assign({WIND_U: ds_cross_section.V, WIND_V: ds_cross_section.v_on_z})
        elif dim == "z_1" or no_third_dim:
            if not no_third_dim:
                title_suffix = f" at z={value:.0f} m"
            x_label = "x_1"
            y_label = "y_1"
            ds_cross_section = ds_cross_section.assign({WIND_U: ds_cross_section.U, WIND_V: ds_cross_section.V})

            subplot_kwargs["projection"] = get_swiss_projection()
            axes_features.append({"feature": cfeature.BORDERS, "linewidth": 1})
        else:
            raise ValueError(f"Invalid dim: {dim}")

        if with_wind:
            wind_max = (self._ds_max.FF if no_third_dim else self._ds_max.FF.sel(**{dim: value})).quantile(0.99).item()
        else:
            wind_max = None

        time = ds_cross_section["time"].values.astype("datetime64[s]").item()
        image_path = build_image_path(time, var, var_contour, with_wind, cross_section=None, dim=dim, value=value,
                                      extension=".png")

        return self._plot_cross_section(ds_cross_section,
                                   x_label, y_label,
                                   var, var_contour, with_wind, wind_max,
                                   plot_kwargs, subplot_kwargs,
                                   figsize, axes_features,
                                   title_suffix, image_path, gif_writer)


    def plot_cross_section_between_two_points(
        self,
        ds: xr.Dataset,
        point1: tuple[Number, Number],
        point2: tuple[Number, Number],
        var: str, var_contour: str | None = None, with_wind: bool = False,
        plot_kwargs: dict | None = None,
        gif_writer: GIFWriter | None = None,
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
        ds_cross_section = _build_cross_section(ds, xs, ys, points_label)

        if plot_kwargs is None:
            plot_kwargs = {}

        if "norm" not in plot_kwargs:
            plot_kwargs["norm"] = _build_norm(_build_cross_section(self._ds_min[var], xs, ys, points_label),
                                              _build_cross_section(self._ds_max[var], xs, ys, points_label))

        wind_max = None if not with_wind else _build_cross_section(self._ds_max.FF,
                                                                   xs, ys, points_label).quantile(0.99).item()

        wind = calculate_wind_speed_in_direction(ds_cross_section, xs[-1] - xs[0], ys[-1] - ys[0])
        ds_cross_section = ds_cross_section.assign({WIND_U: wind, WIND_V: ds_cross_section.v_on_z})

        time = ds_cross_section["time"].values.astype("datetime64[s]").item()
        image_path = build_image_path(time, var, var_contour, with_wind, cross_section=[point1, point2], dim="xy",
                                      value=None, extension=".png")

        return self._plot_cross_section(ds_cross_section,
                                   points_label, "z_1",
                                   var, var_contour, with_wind, wind_max,
                                   plot_kwargs=plot_kwargs, subplot_kwargs={},
                                   figsize=figsize, axes_features=[],
                                   title_suffix="", path=image_path, gif_writer=gif_writer)



    def _plot_cross_section(self,
                            ds_cross_section: xr.Dataset,
                            x_label: str, y_label: str,
                            var: str, var_contour: str | None, with_wind: bool, wind_max: float | None,
                            plot_kwargs: dict, subplot_kwargs: dict,
                            figsize: tuple[int, int], axes_features: list[dict],
                            title_suffix: str, path: str | pathlib.Path,
                            gif_writer: GIFWriter | None) -> plt.Figure:
        if "cmap" not in plot_kwargs:
            cmap = get_cmap_for_variable(var)
            if cmap is not None:
                plot_kwargs["cmap"] = cmap

        fig, ax = plt.subplots(figsize=figsize, subplot_kw=subplot_kwargs)
        ds_cross_section[var].plot(x=x_label, y=y_label, ax=ax, extend="both", **plot_kwargs)
        title = f"{var} (shaded)"

        if var_contour is not None:
            title += f" + {var_contour} (contours)"
            title += f" [{var_contour}: {ds_cross_section[var_contour].attrs['long_name']}, {ds_cross_section[var_contour].attrs['units']}]"

            da_cross_section = ds_cross_section[var_contour]
            da_cross_section_coarse = da_cross_section.interp({x_label: _get_coarse_grid(da_cross_section, x_label),
                                                               y_label: _get_coarse_grid(da_cross_section, y_label)})
            cs = da_cross_section_coarse.plot.contour(x=x_label, y=y_label, ax=ax, colors="black", linewidths=0.5, levels=10)
            ax.clabel(cs, inline=True, fontsize=10, fmt="%.0f")

        if with_wind:
            ds_cross_section_wind = ds_cross_section[[WIND_U, WIND_V]].isel(
                {x_label: _get_sparse_slice(ds_cross_section, x_label),
                 y_label: _get_sparse_slice(ds_cross_section, y_label)}
            )
            wind_plot = ds_cross_section_wind.plot.quiver(
                x=x_label,
                y=y_label,
                u=WIND_U,
                v=WIND_V,
                ax=ax,
                scale=1200 * wind_max / 40,
                add_guide=False,
                headwidth=2,
                headlength=3,
                headaxislength=3,
            )
            quiver_u = int(wind_max / 2)
            ax.quiverkey(wind_plot, 0.95, 1.05, quiver_u, f"{quiver_u} m/s", labelpos='N')

        for feature in axes_features:
            ax.add_feature(**feature)

        title += title_suffix
        title += f"\n{ds_cross_section["time"].values.astype("datetime64[s]")}"
        plt.title(title)
        plt.savefig(path)
        plt.close(fig)

        if gif_writer is not None:
            gif_writer.add_path(path)

        return fig


def _build_cross_section(ds: xr.Dataset | xr.DataArray,
                         xs: np.ndarray, ys: np.ndarray, points_label: str) -> xr.Dataset | xr.DataArray:
    dist = np.sqrt((xs - xs[0]) ** 2 + (ys - ys[0]) ** 2) / 1000
    ds_list_cross_section = ds.interp(x_1=(points_label, xs), y_1=(points_label, ys), method="linear")
    ds_list_cross_section = ds_list_cross_section.assign_coords({points_label: dist})
    ds_list_cross_section.coords[points_label].attrs = dict(units="km", long_name="Distance along section")

    return ds_list_cross_section


def _build_norm(var_min_cross_section: xr.DataArray, var_max_cross_section: xr.DataArray):
    var = var_min_cross_section.name
    vmin = var_min_cross_section.quantile(0.01).item()
    vmax = var_max_cross_section.quantile(0.99).item()

    if var == "v_on_z":
        vmin = min(vmin, -vmax)
        vmax = max(-vmin, vmax)

    return mcolors.Normalize(vmin=vmin, vmax=vmax)


def _get_sparse_slice(ds: xr.Dataset | xr.DataArray, dim: str) -> slice:
    n = max(len(ds[dim]) // 10, 20)
    step = (len(ds[dim]) + n - 1) // n
    return slice(None, None, step)


def _get_coarse_grid(ds: xr.Dataset | xr.DataArray, dim: str) -> np.ndarray:
    if len(ds[dim]) < 200:
        return ds[dim].values
    return np.linspace(ds[dim].min().item(), ds[dim].max().item(), len(ds[dim]) // 8)


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
        plt.savefig(build_image_path(time=None,
                                     var="topography",
                                     var_contour=None,
                                     with_wind=False,
                                     cross_section=selected_points,
                                     dim=None,
                                     value=None,
                                     extension=".png"))

    return selected_points, markers, line


def build_image_path(
    time: datetime.datetime | None,
    var: str, var_contour: str | None, with_wind: bool,
    cross_section: list[tuple[Number, Number]] | None,
    dim: str | None, value: Number | None,
    extension: str,
):
    filename = "" if time is None else time.strftime("%H:%M")
    images_dir = pathlib.Path(__file__).parent / "images"

    if cross_section is not None:
        cross_section_dirname = "vertical_cross_section"
        for point in cross_section:
            cross_section_dirname += f"_{point[0] / 1000:.0f}_{point[1] / 1000:.0f}"
    elif dim is not None:
        cross_section_dirname = f"cross_section_{dim}_{value:.0f}"
    else:
        cross_section_dirname = "no_z_levels"

    filename += f"_{cross_section_dirname}"
    images_dir = images_dir / cross_section_dirname

    var_dirname = f"{var}_contour_{var_contour}_wind_{with_wind}"
    filename += f"_{var_dirname}"
    images_dir = images_dir / var_dirname

    images_dir.mkdir(exist_ok=True, parents=True)

    filename += extension

    return images_dir / filename


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
