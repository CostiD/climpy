"""
climpy.plot
===========
Publication-quality climate figures.

Design: editorial minimalism. Warm parchment continents, pale steel ocean,
hairline coastlines, horizontal colorbars, left-aligned titles.
Every visual element earns its place.
"""

from __future__ import annotations

import string
from pathlib import Path
from typing import Union

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from cartopy.util import add_cyclic_point


# ── Palette ───────────────────────────────────────────────────────────

_LAND_COLOR   = "#2E2E2E"   # charcoal
_OCEAN_COLOR  = "#E6EEF4"   # pale arctic blue
_COAST_COLOR  = "#7A7670"   # muted warm grey
_GRID_COLOR   = "#C8C8C8"

# Colorblind-safe line series (Wong 2011)
_SERIES_COLORS = [
    "#0077BB",  # blue
    "#CC3311",  # red
    "#009988",  # teal
    "#EE7733",  # orange
    "#33BBEE",  # cyan
    "#EE3377",  # magenta
    "#BBBBBB",  # grey
]


# ── Projection helpers ────────────────────────────────────────────────

def Map(central_longitude: float = 0) -> ccrs.PlateCarree:
    return ccrs.PlateCarree(central_longitude=central_longitude)

def Globe(central_longitude: float = 0, central_latitude: float = 0) -> ccrs.Orthographic:
    return ccrs.Orthographic(central_longitude=central_longitude,
                              central_latitude=central_latitude)

def Mollweide(central_longitude: float = 0) -> ccrs.Mollweide:
    return ccrs.Mollweide(central_longitude=central_longitude)

def Robinson(central_longitude: float = 0) -> ccrs.Robinson:
    return ccrs.Robinson(central_longitude=central_longitude)

def NorthPolarStereo(central_longitude: float = 0) -> ccrs.NorthPolarStereo:
    return ccrs.NorthPolarStereo(central_longitude=central_longitude)

def SouthPolarStereo(central_longitude: float = 0) -> ccrs.SouthPolarStereo:
    return ccrs.SouthPolarStereo(central_longitude=central_longitude)

Moll  = Mollweide
Rob   = Robinson
NPole = NorthPolarStereo


# ── _AxProxy ─────────────────────────────────────────────────────────

class _AxProxy:
    def __init__(self, parent: "ClimPlot", ax: mpl.axes.Axes, idx: int):
        self._p   = parent
        self.ax   = ax
        self._idx = idx

    def map(self, X, **kwargs):
        self._p.fill_with_spatial_pattern(X, ax=self.ax, **kwargs)
        return self

    def ts(self, *series, **kwargs):
        if not series:
            raise ValueError("Pass at least one DataArray.")
        self._p.fill_with_time_series(*series, ax=self.ax, **kwargs)
        return self

    def stipple(self, pvalues, lat, lon, **kwargs):
        self._p.add_stippling(self.ax, pvalues, lat, lon, **kwargs)
        return self


# ── ClimPlot ──────────────────────────────────────────────────────────

class ClimPlot:
    """Multi-panel climate figure.

    Parameters
    ----------
    nrows, ncols : int
    n : int, optional    — active subplots (default: nrows * ncols)
    w, h : float         — figure size in inches
    map_proj : tuple     — cartopy CRS per subplot, or 'ts' for plain axes
    layout : str         — 'constrained' (default) or 'tight'

    Example
    -------
    >>> fig = climpy.ClimPlot(nrows=2, ncols=2, w=7.2, h=5.5,
    ...     map_proj=(Map(), Map(), 'ts', 'ts'))
    >>> fig[0].map(eof1, title='(a) EOF 1', cbar_label='°C')
    >>> fig[1].map(eof2, title='(b) EOF 2', cbar_label='°C')
    >>> fig[2].ts(pc1, pc1_ma, labels=['PC 1', '7-yr MA'])
    >>> fig[3].ts(pc2, pc2_ma, labels=['PC 2', '7-yr MA'])
    >>> fig.label_subplots().savefig('fig1.pdf')
    """

    def __init__(
        self,
        nrows: int,
        ncols: int,
        n: int = None,
        w: float = 7.2,
        h: float = 5.0,
        map_proj: tuple = None,
        layout: str = "constrained",
    ):
        if n is None:
            n = nrows * ncols
        if n > nrows * ncols:
            raise ValueError(f"n={n} exceeds nrows x ncols={nrows*ncols}.")

        self._transf = ccrs.PlateCarree()
        self._n      = n
        self._fig    = plt.figure(figsize=(w, h), layout=layout)

        if map_proj is None:
            map_proj = ("ts",) * n

        self.axes: list[mpl.axes.Axes] = []
        for i in range(n):
            proj = map_proj[i]
            if isinstance(proj, str):
                ax = self._fig.add_subplot(nrows, ncols, i + 1)
                self._style_ts_ax(ax)
            else:
                ax = self._fig.add_subplot(nrows, ncols, i + 1, projection=proj)
            self.axes.append(ax)

    @staticmethod
    def _style_ts_ax(ax: mpl.axes.Axes) -> None:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        for spine in ("left", "bottom"):
            ax.spines[spine].set_linewidth(0.5)
            ax.spines[spine].set_color("#999999")
        ax.tick_params(width=0.5, length=3, color="#999999", labelcolor="#333333")

    def __getitem__(self, idx: int) -> _AxProxy:
        return _AxProxy(self, self.axes[idx], idx)

    @property
    def fig(self) -> mpl.figure.Figure:
        return self._fig

    # ── Spatial pattern ───────────────────────────────────────────────

    def fill_with_spatial_pattern(
        self,
        X,
        ax: mpl.axes.Axes,
        *,
        title: str = "",
        xlabel: str = "",
        ylabel: str = "",
        lon_ticks=None,
        lat_ticks=None,
        map_extent=None,
        filltype: str = "contourf",
        nlevels: int = 19,
        show_land: bool = True,
        land_color: str = _LAND_COLOR,
        show_ocean: bool = True,
        ocean_color: str = _OCEAN_COLOR,
        cmap=None,
        diverging: bool = True,
        vcenter: float = 0.0,
        vmin=None,
        vmax=None,
        cbar_step=None,
        cbar_ticks=None,
        cbar_nticks: int = 7,
        shrink: float = 0.90,
        cbar_label: str = "",
        cbar_orientation: str = "horizontal",
        cbar_extend: str = "both",
        cbar_pad: float = 0.03,
        coastlines_lw: float = 0.30,
        coastlines_color: str = _COAST_COLOR,
        grid: bool = True,
        grid_lw: float = 0.20,
        grid_alpha: float = 0.55,
        title_fontsize=None,
        contour_lines: bool = False,
        contour_levels=None,
        contour_lw: float = 0.4,
        contour_color: str = "#555555",
    ) -> "ClimPlot":
        """Draw a filled 2-D spatial pattern on a cartopy map axes.

        Land color fix
        --------------
        facecolor is passed to add_feature(), NOT to NaturalEarthFeature().
        This is required for cartopy >= 0.22 — passing facecolor to the
        constructor has no effect in recent versions.
        """
        if cmap is None:
            cmap = plt.cm.RdYlBu_r

        if vmin is None:
            vmin = float(X.min())
        if vmax is None:
            vmax = float(X.max())

        if cbar_ticks is not None:
            cb_ticks = np.asarray(cbar_ticks)
        elif cbar_step is not None:
            cb_ticks = np.arange(vmin, vmax + cbar_step * 0.01, step=cbar_step)
        else:
            cb_ticks = np.linspace(vmin, vmax, cbar_nticks)

        levels = np.linspace(vmin, vmax, nlevels)

        if diverging:
            norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)
        else:
            norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

        # ── Background (ocean z=0, filled BEFORE data) ────────────────
        if show_ocean:
            ax.add_feature(
                cfeature.OCEAN.with_scale("110m"),
                facecolor=ocean_color,   # ← correct: passed here, not in constructor
                edgecolor="none",
                zorder=0,
            )

        # ── Fill data (z=1) ───────────────────────────────────────────
        lat  = X["lat"].values
        lon  = X["lon"].values
        data, lon_cyc = add_cyclic_point(X.values, coord=lon)

        if filltype == "contourf":
            fill = ax.contourf(
                lon_cyc, lat, data,
                levels=levels, norm=norm, cmap=cmap,
                transform=self._transf, zorder=1, extend=cbar_extend,
            )
        elif filltype == "pcolormesh":
            fill = ax.pcolormesh(
                lon_cyc, lat, data,
                norm=norm, cmap=cmap,
                transform=self._transf, zorder=1, rasterized=True,
            )
        else:
            raise ValueError(f"filltype='{filltype}' not recognised.")

        if contour_lines:
            cl = contour_levels if contour_levels is not None else levels
            ax.contour(
                lon_cyc, lat, data, levels=cl,
                colors=contour_color, linewidths=contour_lw,
                transform=self._transf, zorder=2,
            )

        # ── Land drawn AFTER data so it masks data over continents (z=3) ─
        if show_land:
            ax.add_feature(
                cfeature.LAND.with_scale("110m"),
                facecolor=land_color,    # ← correct: passed here
                edgecolor="none",
                zorder=3,
            )

        # ── Coastlines (z=4) ─────────────────────────────────────────
        ax.coastlines(
            linewidth=coastlines_lw,
            color=coastlines_color,
            zorder=4,
        )

        # ── Extent ───────────────────────────────────────────────────
        if map_extent is not None:
            ax.set_extent(map_extent, crs=self._transf)
        else:
            ax.set_global()

        # ── Gridlines ────────────────────────────────────────────────
        if grid:
            gl = ax.gridlines(
                crs=self._transf,
                draw_labels=True,
                linewidth=grid_lw,
                color=_GRID_COLOR,
                alpha=grid_alpha,
                linestyle="--",
                zorder=5,
            )
            if lon_ticks is not None:
                gl.xlocator = mticker.FixedLocator(lon_ticks)
            if lat_ticks is not None:
                gl.ylocator = mticker.FixedLocator(lat_ticks)
            gl.top_labels   = False
            gl.right_labels = False
            gl.xlabel_style = {"size": mpl.rcParams.get("xtick.labelsize", 6),
                                "color": "#555555"}
            gl.ylabel_style = {"size": mpl.rcParams.get("ytick.labelsize", 6),
                                "color": "#555555"}
            gl.xformatter   = LONGITUDE_FORMATTER
            gl.yformatter   = LATITUDE_FORMATTER

        # ── Colourbar ────────────────────────────────────────────────
        cbar = self._fig.colorbar(
            fill, ax=ax,
            ticks=cb_ticks,
            shrink=shrink,
            extend=cbar_extend,
            orientation=cbar_orientation,
            pad=cbar_pad,
        )
        rounded  = np.round(cb_ticks, 2)
        ticksize = mpl.rcParams.get("xtick.labelsize", 6)
        if cbar_orientation == "horizontal":
            cbar.ax.set_xticklabels(rounded)
            cbar.ax.tick_params(labelsize=ticksize, width=0.4, length=2)
        else:
            cbar.ax.set_yticklabels(rounded)
            cbar.ax.tick_params(labelsize=ticksize, width=0.4, length=2)
        cbar.set_label(cbar_label,
                       size=mpl.rcParams.get("axes.labelsize", 7), labelpad=3)
        cbar.outline.set_linewidth(0.4)

        fs = title_fontsize or mpl.rcParams.get("axes.titlesize", 8)
        ax.set_title(title, fontsize=fs, pad=4, loc="left", color="#222222")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        return self

    # ── Time series ───────────────────────────────────────────────────

    def fill_with_time_series(
        self,
        *series,
        ax: mpl.axes.Axes,
        title: str = "",
        xlabel: str = "",
        ylabel: str = "",
        labels=None,
        colors=None,
        linewidths=None,
        alphas=None,
        linestyles=None,
        markers=None,
        markersizes=None,
        shade_first: bool = True,
        shade_alpha: float = 0.13,
        text: str = None,
        xtext_loc=None,
        ytext_loc=None,
        zero_line: bool = True,
        zero_lw: float = 0.5,
        legend: bool = True,
        legend_loc: str = "best",
        ylim=None,
        title_fontsize=None,
    ) -> "ClimPlot":
        """Plot 1-N time series.

        Convention when multiple series are passed:
          series[0] = raw / thin / semi-transparent
          series[1] = smoothed / bold / opaque  (standard climate paper style)

        shade_first : bool
            Shade positive values in blue and negative in red under the first
            series. Highlights variability sign at a glance.
        """
        n = len(series)
        if n == 0:
            raise ValueError("Pass at least one DataArray.")

        cycle    = plt.rcParams["axes.prop_cycle"].by_key().get("color", _SERIES_COLORS)
        _colors  = colors     or [cycle[i % len(cycle)] for i in range(n)]
        _lws     = linewidths or ([0.7] + [1.8] * (n - 1) if n > 1 else [1.2])
        _alphas  = alphas     or ([0.40] + [1.0] * (n - 1) if n > 1 else [1.0])
        _lstyles = linestyles or ["-"] * n
        _markers = markers    or [""] * n
        _msizes  = markersizes or [3] * n
        _labels  = labels     or [f"Series {i+1}" for i in range(n)]

        # Shade under first series
        if shade_first and n > 1:
            s0 = series[0]
            t  = s0.coords[s0.dims[0]].values
            v  = s0.values
            ax.fill_between(t, np.where(v > 0, v, 0), 0,
                            color=_colors[0], alpha=shade_alpha, linewidth=0)
            ax.fill_between(t, np.where(v < 0, v, 0), 0,
                            color=_SERIES_COLORS[1], alpha=shade_alpha, linewidth=0)

        for i, s in enumerate(series):
            s.plot.line(
                ax=ax,
                color=_colors[i],
                linewidth=_lws[i],
                alpha=_alphas[i],
                linestyle=_lstyles[i],
                marker=_markers[i],
                ms=_msizes[i],
                label=_labels[i],
                add_legend=False,
            )

        if zero_line:
            ax.axhline(0, color="#AAAAAA", linewidth=zero_lw, zorder=0)

        if text is not None and xtext_loc is not None:
            props = dict(boxstyle="round,pad=0.3", facecolor="white",
                         edgecolor="#CCCCCC", linewidth=0.5, alpha=0.85)
            ax.text(xtext_loc, ytext_loc, text, bbox=props,
                    fontsize=mpl.rcParams.get("font.size", 7))

        fs = title_fontsize or mpl.rcParams.get("axes.titlesize", 8)
        ax.set_title(title, fontsize=fs, pad=4, loc="left", color="#222222")
        ax.set_xlabel(xlabel, labelpad=3, color="#555555")
        ax.set_ylabel(ylabel, labelpad=3, color="#555555")

        if ylim is not None:
            ax.set_ylim(ylim)

        auto_labels = [f"Series {i+1}" for i in range(n)]
        if legend and _labels != auto_labels:
            ax.legend(
                loc=legend_loc, frameon=True,
                framealpha=0.88, edgecolor="#DDDDDD",
                fancybox=False,
                fontsize=mpl.rcParams.get("legend.fontsize", 6),
                handlelength=1.8, borderpad=0.5,
            )

        return self

    # ── Stippling ─────────────────────────────────────────────────────

    def add_stippling(
        self,
        ax: mpl.axes.Axes,
        pvalues: np.ndarray,
        lat: np.ndarray,
        lon: np.ndarray,
        pthresh: float = 0.05,
        density: int = 2,
        color: str = "#222222",
        size: float = 2.5,
        marker: str = "x",
        zorder: int = 6,
    ) -> "ClimPlot":
        """Stipple grid points where p < pthresh.

        Default marker changed to 'x' (size=2.5) — the cross marker is the
        standard in climate publications (e.g. Hartmann 2015) and is clearly
        visible at typical journal print sizes, unlike the original dot marker.
        """
        sig = pvalues < pthresh
        ii, jj = np.where(sig)
        if len(ii) == 0:
            return self
        lat_pts = lat[ii[::density]]
        lon_pts = lon[jj[::density]]
        ax.scatter(
            lon_pts, lat_pts, s=size, c=color, marker=marker,
            transform=self._transf, zorder=zorder,
            linewidths=0, alpha=0.65,
        )
        return self

    # ── Panel labels ──────────────────────────────────────────────────

    def label_subplots(
        self,
        labels=None,
        x: float = 0.012,
        y: float = 0.967,
        fontsize: int = None,
        fontweight: str = "bold",
        style: str = "paren",
        ha: str = "left",
        va: str = "top",
        color: str = "#111111",
        box: bool = True,
    ) -> "ClimPlot":
        """Add (a), (b), (c)... inside each panel.

        Default: top-left corner, subtle white pill background.
        style : 'paren' | 'bracket' | 'plain' | 'upper' | 'upper_paren'
        """
        if fontsize is None:
            fontsize = mpl.rcParams.get("axes.titlesize", 8)

        if labels is None:
            lo = string.ascii_lowercase
            hi = string.ascii_uppercase
            n  = len(self.axes)
            _map = {
                "paren"      : [f"({c})" for c in lo[:n]],
                "bracket"    : [f"[{c}]" for c in lo[:n]],
                "plain"      : list(lo[:n]),
                "upper"      : list(hi[:n]),
                "upper_paren": [f"({c})" for c in hi[:n]],
            }
            if style not in _map:
                raise ValueError(f"style='{style}' not in {list(_map)}.")
            labels = _map[style]

        for ax, label in zip(self.axes, labels):
            kw = dict(
                transform=ax.transAxes,
                fontsize=fontsize, fontweight=fontweight,
                ha=ha, va=va, color=color, zorder=10,
            )
            if box:
                kw["bbox"] = dict(
                    boxstyle="round,pad=0.18",
                    facecolor="white", edgecolor="none", alpha=0.72,
                )
            ax.text(x, y, label, **kw)

        return self

    # ── Save / show ───────────────────────────────────────────────────

    def savefig(
        self,
        path: Union[str, Path],
        fmt: str = None,
        dpi: int = 300,
        transparent: bool = False,
        verbose: bool = True,
    ) -> "ClimPlot":
        """Save at publication quality (.pdf / .svg / .png)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if fmt is None:
            fmt = path.suffix.lstrip(".")
        if not fmt:
            fmt  = "pdf"
            path = path.with_suffix(".pdf")
        self._fig.savefig(
            str(path), format=fmt, dpi=dpi,
            transparent=transparent, bbox_inches="tight",
        )
        if verbose:
            print(f"Saved: {path}")
        return self

    def show(self) -> "ClimPlot":
        plt.show()
        return self
