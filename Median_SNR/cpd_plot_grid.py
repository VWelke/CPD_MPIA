"""CPD grid plotting helpers.

This module is meant to replace large, notebook-only plotting cells with a
reusable API.

Design goals
------------
- Keep the plotting logic in one place, with small row-plot functions.
- Make it easy to:
  - change CSV name/pattern used for Andrews grids
  - change the number of columns (just pass more/less sources)
  - add/remove rows (via a row registry)
  - control per-row colorbar ranges
  - control per-row or per-panel x/y ranges
  - optionally auto-adjust x-limits of Zhu rows based on flux map support

Expected inputs
---------------
- sources: list[tuple[str, float, float]]
    Each entry: (disk_name, disk_radius_au, sigma_ujy).
- disk_arr: dict[str, array-like]
    Your per-disk metadata. This module only assumes the same indexing you use
    in your notebook (Mstar at [1], distance pc at [2], Lstar at [3], beam major
    axis arcsec at [4]).
- Andrew_model, Zhu_model: your existing model modules/objects.
- add_row_colorbar: optional external helper. If not provided, a built-in one is
    used.

Example usage
-------------
See the bottom of this file for example call patterns.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


# -----------------------------
# Small utilities
# -----------------------------

def auto_xlim_from_flux(
    Mp_grid_mjup: Sequence[float],
    flux_2d: np.ndarray,
    *,
    flux_min: Optional[float] = None,
    pad_frac: float = 0.06,
    min_span: float = 0.5,
) -> Tuple[float, float]:
    """Infer x-limits in log10(Mp) from the flux map support.

    The inferred range is the union of x-columns that contain finite values.
    If flux_min is given, only values >= flux_min count.

    Returns
    -------
    (x1, x2): float
        Limits in log10(Mp/Mjup).
    """

    x = np.log10(np.asarray(Mp_grid_mjup, dtype=float))
    Z = np.asarray(flux_2d)

    if Z.ndim != 2 or Z.shape[1] != x.size:
        raise ValueError(
            f"flux_2d must be (Ny, Nx=len(Mp_grid)); got {Z.shape} vs {x.size}"
        )

    mask = np.isfinite(Z)
    if flux_min is not None:
        mask &= (Z >= flux_min)

    cols = np.any(mask, axis=0)
    if not np.any(cols):
        return float(x.min()), float(x.max())

    x1 = float(x[cols].min())
    x2 = float(x[cols].max())

    span = x2 - x1
    if span < min_span:
        mid = 0.5 * (x1 + x2)
        x1, x2 = mid - 0.5 * min_span, mid + 0.5 * min_span
        span = x2 - x1

    x1 -= pad_frac * span
    x2 += pad_frac * span
    return x1, x2


def _get_clim(mappable: Any) -> Optional[Tuple[float, float]]:
    """Best-effort get clim from common matplotlib mappables."""
    if mappable is None:
        return None

    # AxesImage / QuadMesh / ScalarMappable
    if hasattr(mappable, "get_clim"):
        try:
            vmin, vmax = mappable.get_clim()
            if vmin is None or vmax is None:
                return None
            return float(vmin), float(vmax)
        except Exception:
            return None

    # ContourSet
    if hasattr(mappable, "levels"):
        try:
            levels = np.asarray(getattr(mappable, "levels"))
            if levels.size == 0:
                return None
            return float(levels.min()), float(levels.max())
        except Exception:
            return None

    return None


def _apply_norm(mappable: Any, norm: mpl.colors.Normalize) -> None:
    """Best-effort apply a Normalize to common matplotlib mappables."""
    if mappable is None:
        return

    if hasattr(mappable, "set_norm"):
        try:
            mappable.set_norm(norm)
            return
        except Exception:
            pass

    # ContourSet
    if hasattr(mappable, "collections"):
        try:
            for coll in mappable.collections:
                coll.set_norm(norm)
        except Exception:
            pass


def add_row_colorbar(
    fig: mpl.figure.Figure,
    axes_row: Sequence[mpl.axes.Axes],
    mappables: Sequence[Any],
    *,
    label: str,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    pad: float = 0.02,
    fraction: float = 0.046,
) -> Optional[mpl.colorbar.Colorbar]:
    """One shared colorbar for a row.

    If vmin/vmax are omitted, they are inferred from the mappables.
    """

    mappables = [m for m in mappables if m is not None]
    if not mappables:
        return None

    clims = [_get_clim(m) for m in mappables]
    clims = [c for c in clims if c is not None]
    if not clims:
        return None

    if vmin is None:
        vmin = min(c[0] for c in clims)
    if vmax is None:
        vmax = max(c[1] for c in clims)

    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    for m in mappables:
        _apply_norm(m, norm)

    cmap = getattr(mappables[0], "cmap", mpl.cm.viridis)
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])

    cb = fig.colorbar(sm, ax=list(axes_row), fraction=fraction, pad=pad)
    # Match notebook styling: explicitly size label and ticks from rcParams.
    cb.set_label(label, fontsize=mpl.rcParams.get("font.size", None))
    cb.ax.tick_params(labelsize=mpl.rcParams.get("font.size", 12) * 0.9)
    return cb


# -----------------------------
# Config + Row registry
# -----------------------------

XLimSpec = Union[Tuple[float, float], Callable[[str, float, Mapping[str, Any]], Tuple[float, float]]]
YLimSpec = Union[Tuple[float, float], Callable[[str, float, Any, Mapping[str, Any]], Tuple[float, float]]]


@dataclass
class PlotStyle:
    """Matplotlib rcParams overrides to apply during the plot."""

    font_size: int = 16
    font_family: str = "serif"
    axes_linewidth: float = 1.0
    xtick_direction: str = "in"
    ytick_direction: str = "in"

    def as_rc(self) -> Dict[str, Any]:
        return {
            "font.size": self.font_size,
            "font.family": self.font_family,
            "axes.linewidth": self.axes_linewidth,
            "xtick.direction": self.xtick_direction,
            "ytick.direction": self.ytick_direction,
        }


@dataclass
class ZhuGridConfig:
    lam_mm: float = 0.9
    Rout_frac: float = 0.3
    alpha_calc: float = 1e-3
    alpha_label: float = 1e-2
    plot_thick_thin: bool = False

    Mdot_logmin: float = -8
    Mdot_logmax: float = -5
    n_Mp: int = 100
    n_Mdot: int = 100

    # Mp vs alpha panel uses a fixed Mdot (can be None if your Zhu_model supports that)
    Mdot_fixed_for_alpha_panel: Optional[float] = None


@dataclass
class AndrewsGridConfig:
    """Where to load the Andrews grid CSV from."""

    # e.g. r"D:\CPD_MPIA\CPD_Emission_Models\Andrews\{disk_name}_{disk_radius}_alpha0.001_{tag}.csv"
    csv_template: str
    tag: str = "MidRes"
    alpha: float = 1e-3


@dataclass
class GridPlotConfig:
    """Top-level configuration for the grid plot."""

    style: PlotStyle
    andrews: AndrewsGridConfig
    zhu: ZhuGridConfig

    # detection sigma used in Andrews panel
    det_sigma: float = 5

    # axis limits: values are log10 bounds for Mp (x axis). Use callables for per-disk logic.
    xlim_andrews: XLimSpec = (-2.0, 1.0)
    xlim_zhu_mdot: XLimSpec = (-2.0, 1.0)
    xlim_zhu_alpha: XLimSpec = (-2.0, 1.0)

    # y-limits for Andrews panel (Rout/RHill). If callable, signature:
    #   fn(disk_name, disk_radius, Rout_frac_grid, cfg_dict)->(ymin,ymax)
    ylim_andrews: YLimSpec = (0.0, 1.0)

    # colorbar ranges (None means infer from mappables)
    clim_andrews: Optional[Tuple[float, float]] = (-5.5, -1.0)
    clim_zhu_row: Optional[Tuple[float, float]] = None

    # Optional: after plotting, couple xlim of zhu_mdot + zhu_alpha using inferred support.
    auto_xlim_zhu_rows: bool = False
    auto_xlim_flux_min: Optional[float] = None


@dataclass(frozen=True)
class RowSpec:
    key: str
    plotter: Callable[..., Tuple[Any, Dict[str, Any]]]
    side_label: str
    cbar_label: str


ROW_REGISTRY: Dict[str, RowSpec] = {}


def register_row(key: str, plotter: Callable[..., Tuple[Any, Dict[str, Any]]], *, side_label: str, cbar_label: str) -> None:
    """Register a row key -> plotter mapping.

    Adding a new row later means:
      1) write a plotter(ax, ..., cfg, disk_arr, Andrew_model, Zhu_model) -> (mappable, cache)
      2) register_row("newrow", plotter, side_label="...", cbar_label="...")
      3) include "newrow" in rows=(...,)
    """

    ROW_REGISTRY[key] = RowSpec(key=key, plotter=plotter, side_label=side_label, cbar_label=cbar_label)


def _resolve_xlim(xlim: XLimSpec, disk_name: str, disk_radius: float, cfg_dict: Mapping[str, Any]) -> Tuple[float, float]:
    if callable(xlim):
        return xlim(disk_name, disk_radius, cfg_dict)
    return xlim


# -----------------------------
# Default row plotters
# -----------------------------


def plot_row_andrews(
    ax: mpl.axes.Axes,
    *,
    disk_name: str,
    disk_radius: float,
    sigma_ujy: float,
    cfg: GridPlotConfig,
    disk_arr: Mapping[str, Sequence[float]],
    Andrew_model: Any,
    **_: Any,
) -> Tuple[Any, Dict[str, Any]]:
    """Row 1: Andrews Mp vs Rout/RHill panel."""

    csv_path = cfg.andrews.csv_template.format(
        disk_name=disk_name, disk_radius=disk_radius, tag=cfg.andrews.tag
    )

    import pandas as pd  # local import keeps module import lightweight

    df = pd.read_csv(csv_path)

    Mp_grid = df["Mp_grid"].unique()
    Rout_frac_grid = df["Rout_frac_grid"].unique()

    Flux_vals = df.pivot(index="Rout_frac_grid", columns="Mp_grid", values="Flux_vals").values
    M_cpd_chosen = df.pivot(index="Rout_frac_grid", columns="Mp_grid", values="Mcpd_vals").values

    x_min, x_max = _resolve_xlim(cfg.xlim_andrews, disk_name, disk_radius, cfg.__dict__)

    if callable(cfg.ylim_andrews):
        y_min, y_max = cfg.ylim_andrews(disk_name, disk_radius, Rout_frac_grid, cfg.__dict__)
    else:
        y_min, y_max = cfg.ylim_andrews

    vmin, vmax = cfg.clim_andrews if cfg.clim_andrews is not None else (-5.5, -1.0)

    im = Andrew_model.plot_flux_map(
        Mp_grid,
        Rout_frac_grid,
        Flux_vals,
        M_cpd_chosen,
        disk_arr=disk_arr[disk_name],
        ax=ax,
        rp=disk_radius,
        alpha=cfg.andrews.alpha,
        x_min=x_min,
        x_max=x_max,
        y_min=float(y_min),
        y_max=float(y_max),
        min=vmin,
        max=vmax,
        res_au=disk_arr[disk_name][4] * disk_arr[disk_name][2],
        sigma_ujy=sigma_ujy,
        det_sigma=cfg.det_sigma,
    )

    return im, {}


def plot_row_zhu_mdot(
    ax: mpl.axes.Axes,
    *,
    disk_name: str,
    disk_radius: float,
    sigma_ujy: float,
    cfg: GridPlotConfig,
    disk_arr: Mapping[str, Sequence[float]],
    Zhu_model: Any,
    **_: Any,
) -> Tuple[Any, Dict[str, Any]]:
    """Row: Zhu Mp vs Mdot flux map."""

    x_min, x_max = _resolve_xlim(cfg.xlim_zhu_mdot, disk_name, disk_radius, cfg.__dict__)
    Mp_grid = np.logspace(x_min, x_max, cfg.zhu.n_Mp)
    Mdot_grid = np.logspace(cfg.zhu.Mdot_logmin, cfg.zhu.Mdot_logmax, cfg.zhu.n_Mdot)

    Flux_vals = np.zeros((len(Mdot_grid), len(Mp_grid)))
    bmaj = disk_arr[disk_name][4]  # arcsec

    for i, Mdot in enumerate(Mdot_grid):
        for j, Mp in enumerate(Mp_grid):
            RHill = disk_radius * (Mp / (3 * 1047.56 * disk_arr[disk_name][1])) ** (1 / 3)
            Rout = cfg.zhu.Rout_frac * RHill

            out = Zhu_model.calculate_disk_properties_zhu(
                M_star=disk_arr[disk_name][1],
                Mp=Mp,
                Mdot=Mdot,
                alpha=cfg.zhu.alpha_calc,
                rp=disk_radius,
                d_pc=disk_arr[disk_name][2],
                lam_mm=cfg.zhu.lam_mm,
                Rout=Rout,
                Lstar=disk_arr[disk_name][3],
            )
            Flux_vals[i, j] = out["F_nu_tot"]

    im = Zhu_model.plot_zhu_Mp_Mdot_flux(
        Mp_grid,
        Mdot_grid,
        Flux_vals,
        disk_arr=disk_arr[disk_name],
        rp=disk_radius,
        alpha=cfg.zhu.alpha_label,
        target_flux_arr=[sigma_ujy],
        ax=ax,
        beam_major_arcsec=bmaj,
        plot_thick_thin=cfg.zhu.plot_thick_thin,
    )

    return im, {"Mp_grid": Mp_grid, "flux_2d": Flux_vals}


def plot_row_zhu_alpha(
    ax: mpl.axes.Axes,
    *,
    disk_name: str,
    disk_radius: float,
    sigma_ujy: float,
    cfg: GridPlotConfig,
    disk_arr: Mapping[str, Sequence[float]],
    Zhu_model: Any,
    **_: Any,
) -> Tuple[Any, Dict[str, Any]]:
    """Row: Zhu Mp vs alpha flux map."""

    x_min, x_max = _resolve_xlim(cfg.xlim_zhu_alpha, disk_name, disk_radius, cfg.__dict__)

    Mp_grid_alpha, alpha_grid, Flux_map = Zhu_model.compute_flux_map_zhu_MpAlpha(
        disk_arr[disk_name],
        Mdot=cfg.zhu.Mdot_fixed_for_alpha_panel,
        Mp_min=x_min,
        Mp_max=x_max,
        rp=disk_radius,
        lam_mm=cfg.zhu.lam_mm,
        verbose=False,
    )

    im = Zhu_model.plot_zhu_Mp_alpha_flux(
        Mp_grid_alpha,
        alpha_grid,
        Flux_map,
        disk_arr=disk_arr[disk_name],
        rp=disk_radius,
        Mdot=cfg.zhu.Mdot_fixed_for_alpha_panel,
        target_flux_arr=[sigma_ujy],
        beam_major_arcsec=disk_arr[disk_name][4],
        ax=ax,
    )

    return im, {"Mp_grid": Mp_grid_alpha, "flux_2d": Flux_map}


# Register default rows
register_row(
    "andrews",
    plot_row_andrews,
    side_label="Mp vs Rout",
    cbar_label=r"$\log_{10}(M_{\rm CPD}/M_{\rm Jup})$",
)
register_row(
    "zhu_mdot",
    plot_row_zhu_mdot,
    side_label="Mp vs Mdot",
    cbar_label=r"$\log_{10}(F_\nu\, [\mu{\rm Jy}])$",
)
register_row(
    "zhu_alpha",
    plot_row_zhu_alpha,
    side_label="Mp vs α",
    cbar_label=r"$\log_{10}(F_\nu\, [\mu{\rm Jy}])$",
)


# -----------------------------
# Main API
# -----------------------------


def plot_cpd_grid(
    sources: Sequence[Tuple[str, float, float]],
    *,
    rows: Sequence[Union[str, RowSpec]] = ("andrews", "zhu_mdot", "zhu_alpha"),
    title_fmt: str = "{disk_name} (r={disk_radius:.0f} au)",
    figsize: Tuple[float, float] = (18, 15),
    cfg: GridPlotConfig,
    disk_arr: Mapping[str, Sequence[float]],
    Andrew_model: Any,
    Zhu_model: Any,
    row_colorbar_limits: Optional[Mapping[str, Tuple[float, float]]] = None,
    colorbar_func: Optional[Callable[..., Any]] = None,
) -> Tuple[mpl.figure.Figure, np.ndarray]:
    """Plot a grid of CPD model panels.

    Parameters
    ----------
    sources:
        List of (disk_name, disk_radius_au, sigma_ujy).

    rows:
        Row keys (strings) registered via register_row, or RowSpec objects.

    row_colorbar_limits:
        Optional overrides per-row key, e.g. {"zhu_mdot": (0, 6), "zhu_alpha": (0, 6)}.

    colorbar_func:
        Colorbar helper. If None, uses built-in add_row_colorbar().

    Returns
    -------
    (fig, axes)
        axes is a (nrows, ncols) ndarray of Axes.
    """

    row_specs: List[RowSpec] = []
    for r in rows:
        if isinstance(r, RowSpec):
            row_specs.append(r)
        else:
            row_specs.append(ROW_REGISTRY[r])

    ncols = len(sources)
    nrows = len(row_specs)

    if colorbar_func is None:
        colorbar_func = add_row_colorbar

    # Apply style within a local rc_context so we don't pollute the notebook state.
    with mpl.rc_context(cfg.style.as_rc()):
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize, sharex=False, sharey=False)
        axes = np.atleast_2d(axes).reshape(nrows, ncols)

        # mappables per row for shared colorbars
        row_mappables: Dict[str, List[Any]] = {rs.key: [] for rs in row_specs}

        # per-column caches, used for optional auto-xlim coupling
        col_cache: List[Dict[str, Dict[str, Any]]] = [dict() for _ in range(ncols)]

        for col, (disk_name, disk_radius, sigma_ujy) in enumerate(sources):
            axes[0, col].set_title(
                title_fmt.format(disk_name=disk_name.replace("_", " "), disk_radius=disk_radius),
                pad=10,
                fontweight="bold",
            )

            for r_i, rs in enumerate(row_specs):
                ax = axes[r_i, col]
                ax.set_aspect("auto")

                im, cache = rs.plotter(
                    ax,
                    disk_name=disk_name,
                    disk_radius=disk_radius,
                    sigma_ujy=sigma_ujy,
                    cfg=cfg,
                    disk_arr=disk_arr,
                    Andrew_model=Andrew_model,
                    Zhu_model=Zhu_model,
                )

                row_mappables[rs.key].append(im)
                col_cache[col][rs.key] = cache

            # Optional: auto-couple x-limits for the Zhu rows.
            if cfg.auto_xlim_zhu_rows:
                # Only if both caches exist in this column
                if ("zhu_mdot" in col_cache[col]) and ("zhu_alpha" in col_cache[col]):
                    A = col_cache[col]["zhu_mdot"]
                    B = col_cache[col]["zhu_alpha"]

                    x1a, x2a = auto_xlim_from_flux(A["Mp_grid"], A["flux_2d"], flux_min=cfg.auto_xlim_flux_min)
                    x1b, x2b = auto_xlim_from_flux(B["Mp_grid"], B["flux_2d"], flux_min=cfg.auto_xlim_flux_min)
                    x1, x2 = min(x1a, x1b), max(x2a, x2b)

                    for r_i, rs in enumerate(row_specs):
                        if rs.key in ("zhu_mdot", "zhu_alpha"):
                            axes[r_i, col].set_xlim(x1, x2)

                    # Match notebook behavior: adjust subplot spacing BEFORE adding colorbars.
                    # (If you adjust after adding colorbars, matplotlib will reflow the axes
                    # and the colorbars can look squashed/misaligned.)
                    plt.subplots_adjust(left=0.12, hspace=0.35, wspace=0.30)

        # One colorbar per row
        row_colorbar_limits = dict(row_colorbar_limits or {})

        for r_i, rs in enumerate(row_specs):
            vmin_vmax = row_colorbar_limits.get(rs.key)
            if vmin_vmax is None:
                # fall back to config defaults
                if rs.key == "andrews" and cfg.clim_andrews is not None:
                    vmin_vmax = cfg.clim_andrews
                if rs.key in ("zhu_mdot", "zhu_alpha") and cfg.clim_zhu_row is not None:
                    vmin_vmax = cfg.clim_zhu_row

            if vmin_vmax is None:
                colorbar_func(fig, axes[r_i, :], row_mappables[rs.key], label=rs.cbar_label)
            else:
                colorbar_func(
                    fig,
                    axes[r_i, :],
                    row_mappables[rs.key],
                    label=rs.cbar_label,
                    vmin=vmin_vmax[0],
                    vmax=vmin_vmax[1],
                )

            # left-side row label
            axes[r_i, 0].annotate(
                rs.side_label,
                xy=(0, 0.5),
                xycoords="axes fraction",
                xytext=(-60, 0),
                textcoords="offset points",
                ha="right",
                va="center",
                rotation="vertical",
                fontsize=14,
                fontweight="bold",
            )

    return fig, axes


# -----------------------------
# Examples
# -----------------------------

if __name__ == "__main__":
    # Example patterns for calling this module.
    #
    # Note: This block is not executed in a notebook import. It’s mainly a
    # reference for how to structure calls.

    # 1) Build cfg
    cfg = GridPlotConfig(
        style=PlotStyle(font_size=16, font_family="serif"),
        andrews=AndrewsGridConfig(
            csv_template=r"D:\\CPD_MPIA\\CPD_Emission_Models\\Andrews\\{disk_name}_{disk_radius}_alpha0.001_{tag}.csv",
            tag="MidRes",
            alpha=1e-3,
        ),
        zhu=ZhuGridConfig(lam_mm=0.9),
        xlim_andrews=(-2.0, 1.0),
        xlim_zhu_mdot=(-2.0, 1.0),
        xlim_zhu_alpha=(-2.0, 1.0),
        ylim_andrews=(0.0, 1.0),
        clim_andrews=(-5.5, -1.0),
        clim_zhu_row=(0.0, 6.0),
        auto_xlim_zhu_rows=False,
    )

    # 2) Prepare sources (from your notebook):
    # sources = mid_res_gaps[7:10]
    sources = [
        ("AA_Tau", 80.0, 10.0),
        ("CQ_Tau", 165.0, 20.0),
        ("LkCa_15", 64.0, 15.0),
    ]

    # 3) Provide disk_arr, Andrew_model, Zhu_model from your environment.
    # from Dictionaries import disk_arr
    # import Andrew_model, Zhu_model

    # fig, axes = plot_cpd_grid(
    #     sources,
    #     rows=("andrews", "zhu_mdot", "zhu_alpha"),
    #     cfg=cfg,
    #     disk_arr=disk_arr,
    #     Andrew_model=Andrew_model,
    #     Zhu_model=Zhu_model,
    #     row_colorbar_limits={"zhu_mdot": (0, 6), "zhu_alpha": (0, 6)},
    # )
    # plt.show()

    # 4) Add a new row later:
    #
    # def my_new_row(ax, *, disk_name, disk_radius, sigma_ujy, cfg, disk_arr, Zhu_model, **_):
    #     # ... compute something ...
    #     im = ax.imshow(np.random.randn(30, 30), origin="lower")
    #     return im, {}
    #
    # register_row("newdiag", my_new_row, side_label="New diag", cbar_label="arb")
    #
    # fig, axes = plot_cpd_grid(
    #     sources,
    #     rows=("andrews", "zhu_mdot", "zhu_alpha", "newdiag"),
    #     cfg=cfg,
    #     disk_arr=disk_arr,
    #     Andrew_model=Andrew_model,
    #     Zhu_model=Zhu_model,
    # )
    # plt.show()
    pass
