"""
climpy.plot
===========
Publication-quality climate plotting (maps + time series).

Quick start
-----------
>>> import climpy
>>> climpy.use_style('nature')          # optional but recommended
>>>
>>> fig = climpy.ClimPlot(nrows=2, ncols=2, w=7.2, h=5,
...                       map_proj=(Map(), Map(), 'ts', 'ts'))
>>> fig[0].map(eof1, title='EOF 1', cbar_label='°C', vmin=-0.6, vmax=0.6)
>>> fig[1].map(eof2, title='EOF 2', cbar_label='°C', vmin=-0.6, vmax=0.6)
>>> fig[2].ts(pc1, pc1_smooth, labels=['PC 1', '7-yr MA'])
>>> fig[3].ts(pc2, pc2_smooth, labels=['PC 2', '7-yr MA'])
>>> fig.label_subplots()
>>> fig.savefig('figures/fig1.pdf')

The short-hand ``fig[i].map(...)`` / ``fig[i].ts(...)`` API delegates to the
full ``ClimPlot.fill_with_spatial_pattern`` / ``fill_with_time_series`` methods,
which accept every possible keyword argument for fine-grained control.
"""

from __future__ import annotations

import string
from pathlib import Path
from typing import Optional, Union

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from cartopy.util import add_cyclic_point


# ── Projection helpers ────────────────────────────────────────────────

def Map(central_longitude: float = 0) -> ccrs.PlateCarree:
    """Plate Carrée projection (standard rectangular map)."""
    return ccrs.PlateCarree(central_longitude=central_longitude)


def Globe(central_longitude: float = 0, central_latitude: float = 0) -> ccrs.Orthographic:
    """Orthographic projection (3-D globe view)."""
    return ccrs.Orthographic(central_longitude=central_longitude,
                             central_latitude=central_latitude)


def Mollweide(central_longitude: float = 0) -> ccrs.Mollweide:
    """Mollweide equal-area projection."""
    return ccrs.Mollweide(central_longitude=central_longitude)


def Robinson(central_longitude: float = 0) -> ccrs.Robinson:
    """Robinson projection."""
    return ccrs.Robinson(central_longitude=central_longitude)


def NorthPolarStereo(central_longitude: float = 0) -> ccrs.NorthPolarStereo:
    """North polar stereographic projection."""
    return ccrs.NorthPolarStereo(central_longitude=central_longitude)


def SouthPolarStereo(central_longitude: float = 0) -> ccrs.SouthPolarStereo:
    """South polar stereographic projection."""
    return ccrs.SouthPolarStereo(central_longitude=central_longitude)


# Keep old short names for compatibility
Moll = Mollweide
Rob  = Robinson
NPole = NorthPolarStereo


# ── _AxProxy ─────────────────────────────────────────────────────────

class _AxProxy:
    """Thin proxy returned by ClimPlot[i] that exposes short-hand methods."""

    def __init__(self, parent: "ClimPlot", ax: mpl.axes.Axes, idx: int):
        self._p = parent
        self.ax = ax
        self._idx = idx

    # Short aliases
    def map(self, X, **kwargs):
        """Short-hand for fill_with_spatial_pattern on this axis."""
        self._p.fill_with_spatial_pattern(X, ax=self.ax, **kwargs)
        return self

    def ts(self, *series, **kwargs):
        """Short-hand for fill_with_time_series on this axis."""
        if len(series) == 0:
            raise ValueError("At least one DataArray must be passed.")
        self._p.fill_with_time_series(*series, ax=self.ax, **kwargs)
        return self

    def stipple(self, pvalues, lat, lon, **kwargs):
        """Short-hand for add_stippling on this axis."""
        self._p.add_stippling(self.ax, pvalues, lat, lon, **kwargs)
        return self


# ── ClimPlot ──────────────────────────────────────────────────────────

class ClimPlot:
    """Multi-panel figure for climate analysis.

    Parameters
    ----------
    nrows, ncols : int
        Grid dimensions.
    n : int, optional
        Number of active subplots (≤ nrows * ncols). Default: nrows * ncols.
    w, h : float
        Figure width and height in inches.
        Use ``climpy.style.NATURE_2COL`` etc. for journal-compliant widths.
    map_proj : tuple
        One entry per subplot. Use a cartopy projection object (e.g. ``Map()``)
        for maps, or the string ``'ts'`` (or ``'rectilinear'``) for
        ordinary axes (time series, histograms, etc.).
    layout : str
        Matplotlib layout engine: ``'constrained'`` (default, recommended) or
        ``'tight'``.

    Examples
    --------
    >>> # 2×2 grid: two maps on top, two time-series on bottom
    >>> fig = ClimPlot(nrows=2, ncols=2, w=7.2, h=5,
    ...                map_proj=(Map(), Map(), 'ts', 'ts'))
    >>> fig[0].map(eof1, title='(a) EOF 1')
    >>> fig[1].map(eof2, title='(b) EOF 2')
    >>> fig[2].ts(pc1)
    >>> fig[3].ts(pc2)
    >>> fig.savefig('fig1.pdf')
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
            raise ValueError(
                f"n={n} exceeds nrows×ncols={nrows*ncols}."
            )

        self._transf = ccrs.PlateCarree()
        self._n = n

        self._fig = plt.figure(figsize=(w, h), layout=layout)

        if map_proj is None:
            map_proj = ("ts",) * n

        self.axes: list[mpl.axes.Axes] = []
        for i in range(n):
            proj = map_proj[i]
            if isinstance(proj, str):
                # 'ts', 'rectilinear', etc. → plain axes
                ax = self._fig.add_subplot(nrows, ncols, i + 1)
            else:
                ax = self._fig.add_subplot(nrows, ncols, i + 1, projection=proj)
            self.axes.append(ax)

    # ── Indexing shortcut ─────────────────────────────────────────────

    def __getitem__(self, idx: int) -> _AxProxy:
        """fig[i] returns a proxy with .map() and .ts() short-hand methods."""
        return _AxProxy(self, self.axes[idx], idx)

    # ── Spatial pattern (map) ─────────────────────────────────────────

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
        nlevels: int = 17,
        show_land: bool = True,
        land_color: str = "white",
        show_ocean: bool = False,
        ocean_color: str = "lightblue",
        cmap=None,
        diverging: bool = True,
        vcenter: float = 0.0,
        vmin=None,
        vmax=None,
        cbar_step=None,
        cbar_ticks=None,
        cbar_nticks: int = 6,
        shrink: float = 0.7,
        cbar_label: str = "",
        cbar_orientation: str = "vertical",
        cbar_extend: str = "both",
        coastlines_lw: float = 0.4,
        grid: bool = False,
        grid_lw: float = 0.3,
        grid_alpha: float = 0.5,
        title_fontsize: int = None,
        contour_lines: bool = False,
        contour_levels=None,
        contour_lw: float = 0.5,
        contour_color: str = "k",
    ) -> "ClimPlot":
        """Draw a filled 2-D spatial pattern (map).

        Parameters
        ----------
        X : xr.DataArray
            2-D field with 'lat' and 'lon' coordinates.
        ax : matplotlib.axes.Axes
            Target axes (must be a GeoAxes with a cartopy projection).
        title : str
            Subplot title.
        lon_ticks, lat_ticks : list of float
            Tick positions for the longitude and latitude axes.
        map_extent : [min_lon, max_lon, min_lat, max_lat], optional
            If None, the full global extent is shown.
        filltype : 'contourf' | 'pcolormesh'
            Type of fill. 'contourf' is smoother; 'pcolormesh' shows raw pixels.
        nlevels : int
            Number of colour levels for 'contourf' (default 17).
        diverging : bool
            If True, use a two-slope normalisation centred on ``vcenter``.
        vmin, vmax : float
            Colour scale limits. Computed automatically if None.
        cbar_step : float, optional
            Spacing between colourbar ticks. Overrides cbar_nticks.
        cbar_nticks : int
            Approximate number of colourbar ticks (used when cbar_step is None).
        shrink : float
            Colourbar length relative to the axes height.
        cbar_label : str
            Colourbar label (supports LaTeX).
        cbar_orientation : 'vertical' | 'horizontal'
        contour_lines : bool
            If True, overlay thin contour lines on the filled plot.
        contour_levels : array-like, optional
            Specific contour levels. If None, the same levels as contourf are used.
        """
        if cmap is None:
            cmap = plt.cm.RdBu_r

        # --- colour limits ---
        if vmin is None:
            vmin = float(X.min())
        if vmax is None:
            vmax = float(X.max())

        if cbar_step is not None:
            cbar_ticks_computed = np.arange(vmin, vmax + cbar_step * 0.01,
                                             step=cbar_step)
        elif cbar_ticks is not None:
            cbar_ticks_computed = np.asarray(cbar_ticks)
        else:
            cbar_ticks_computed = np.linspace(vmin, vmax, cbar_nticks)

        levels = np.linspace(vmin, vmax, nlevels, endpoint=True)

        # --- normalisation ---
        if diverging:
            norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)
        else:
            norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

        # --- fill ---
        lat = X["lat"].values
        lon = X["lon"].values
        data, lon_cyc = add_cyclic_point(X.values, coord=lon)

        if filltype == "contourf":
            fill = ax.contourf(
                lon_cyc, lat, data, levels=levels,
                norm=norm, cmap=cmap,
                transform=self._transf, zorder=2, extend=cbar_extend,
            )
        elif filltype == "pcolormesh":
            fill = ax.pcolormesh(
                lon_cyc, lat, data,
                norm=norm, cmap=cmap,
                transform=self._transf, zorder=2,
            )
        else:
            raise ValueError(
                f"filltype='{filltype}' not recognised. "
                "Use 'contourf' or 'pcolormesh'."
            )

        # --- optional contour lines overlay ---
        if contour_lines:
            cl = contour_levels if contour_levels is not None else levels
            ax.contour(
                lon_cyc, lat, data, levels=cl,
                colors=contour_color, linewidths=contour_lw,
                transform=self._transf, zorder=3,
            )

        # --- colourbar ---
        cbar = self._fig.colorbar(
            fill, ax=ax,
            ticks=cbar_ticks_computed,
            shrink=shrink,
            extend=cbar_extend,
            label=cbar_label,
            orientation=cbar_orientation,
        )
        tick_labels = np.round(cbar_ticks_computed, decimals=2)
        if cbar_orientation == "horizontal":
            cbar.ax.set_xticklabels(tick_labels)
        else:
            cbar.ax.set_yticklabels(tick_labels)

        # --- extent, land, features ---
        if map_extent is not None:
            ax.set_extent(map_extent, crs=self._transf)
        else:
            ax.set_global()

        ax.coastlines(linewidth=coastlines_lw, zorder=4)

        if show_land:
            ax.add_feature(cfeature.LAND, zorder=1, facecolor=land_color)
        if show_ocean:
            ax.add_feature(cfeature.OCEAN, zorder=1, facecolor=ocean_color)

        # --- gridlines ---
        if grid:
            gl = ax.gridlines(
                crs=self._transf, draw_labels=False,
                linewidth=grid_lw, color="gray",
                alpha=grid_alpha, linestyle="--",
            )

        # --- ticks ---
        if lon_ticks is not None and lat_ticks is not None:
            ax.set_xticks(lon_ticks, crs=self._transf)
            ax.set_yticks(lat_ticks, crs=self._transf)
            ax.xaxis.set_major_formatter(
                LongitudeFormatter(zero_direction_label=True)
            )
            ax.yaxis.set_major_formatter(LatitudeFormatter())

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if title_fontsize:
            ax.set_title(title, fontsize=title_fontsize)
        else:
            ax.set_title(title)

        return self

    # ── Time series ───────────────────────────────────────────────────

    def fill_with_time_series(
        self,
        *series,
        ax: mpl.axes.Axes,
        title: str = "",
        xlabel: str = "Year",
        ylabel: str = "",
        labels=None,
        colors=None,
        markers=None,
        markersizes=None,
        linewidths=None,
        alphas=None,
        linestyles=None,
        text: str = None,
        xtext_loc=None,
        ytext_loc=None,
        zero_line: bool = True,
        zero_lw: float = 0.6,
        legend: bool = True,
        legend_loc: str = "best",
        shading_series=None,
        shading_alpha: float = 0.15,
    ) -> "ClimPlot":
        """Plot 1–N time series on the same axes.

        Parameters
        ----------
        *series : xr.DataArray
            Any number of 1-D time series to overlay.
        ax : matplotlib.axes.Axes
        labels : list of str, optional
            Legend labels, one per series.
        colors : list of str, optional
            Line colours. Defaults to the current colour cycle.
        markers : list of str, optional
            Marker style per series (default '').
        text : str, optional
            Annotation text, placed at (xtext_loc, ytext_loc).
        zero_line : bool
            If True (default), draw a thin horizontal line at y=0.
        shading_series : xr.DataArray, optional
            If given, shade ±1 std or a second series as a band around series[0].

        Example
        -------
        >>> fig[2].ts(pc1, pc1_smooth,
        ...           labels=['PC 1', '7-yr MA'],
        ...           colors=['steelblue', 'firebrick'],
        ...           alphas=[0.5, 1.0])
        """
        n = len(series)
        if n == 0:
            raise ValueError("Pass at least one DataArray.")

        # --- defaults ---
        prop_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        _colors    = colors    or [prop_cycle[i % len(prop_cycle)] for i in range(n)]
        _markers   = markers   or [""] * n
        _msizes    = markersizes or [3] * n
        _lws       = linewidths  or [1.0] * n
        _alphas    = alphas      or [1.0] * n
        _lstyles   = linestyles  or ["-"] * n
        _labels    = labels      or [f"Series {i+1}" for i in range(n)]

        for i, s in enumerate(series):
            s.plot.line(
                ax=ax,
                color=_colors[i],
                marker=_markers[i],
                ms=_msizes[i],
                linewidth=_lws[i],
                alpha=_alphas[i],
                linestyle=_lstyles[i],
                label=_labels[i],
            )

        # optional shading band
        if shading_series is not None:
            ax.fill_between(
                series[0].coords[series[0].dims[0]].values,
                series[0].values - shading_series.values,
                series[0].values + shading_series.values,
                color=_colors[0], alpha=shading_alpha, linewidth=0,
            )

        if zero_line:
            ax.axhline(0, color="k", linewidth=zero_lw, zorder=0)

        if text is not None and xtext_loc is not None:
            props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)
            ax.text(xtext_loc, ytext_loc, text, bbox=props,
                    fontsize=mpl.rcParams.get("font.size", 7))

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)

        if legend:
            ax.legend(loc=legend_loc, frameon=True, framealpha=0.8)

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
        color: str = "k",
        size: float = 0.5,
        marker: str = ".",
        zorder: int = 6,
    ) -> "ClimPlot":
        """Add stippling to mark statistically significant regions.

        Parameters
        ----------
        ax : GeoAxes target.
        pvalues : 2-D numpy array (lat × lon) of p-values.
        lat, lon : 1-D arrays of coordinates.
        pthresh : significance threshold (default 0.05).
        density : plot every ``density``-th significant point (default 2).
        size : scatter dot size (default 0.5 pt²).

        Example
        -------
        >>> corr, pval = climpy.correlation_pvalue(pc, field)
        >>> fig.add_stippling(fig.axes[0], pval.values, lat, lon)
        """
        sig = pvalues < pthresh
        ii, jj = np.where(sig)
        lat_pts = lat[ii[::density]]
        lon_pts = lon[jj[::density]]
        ax.scatter(
            lon_pts, lat_pts,
            s=size, c=color, marker=marker,
            transform=self._transf, zorder=zorder, linewidths=0,
        )
        return self

    # ── Subplot labels ────────────────────────────────────────────────

    def label_subplots(
        self,
        labels=None,
        x: float = -0.06,
        y: float = 1.02,
        fontsize: int = None,
        fontweight: str = "bold",
        style: str = "paren",
        ha: str = "right",
        va: str = "bottom",
    ) -> "ClimPlot":
        """Add (a), (b), (c)... labels to each subplot.

        Parameters
        ----------
        labels : list of str, optional
            Custom labels. If None, auto-generated from ``style``.
        x, y : float
            Position in axes-fraction coordinates.
            x < 0 places the label to the left of the y-axis.
        fontsize : int, optional
            Defaults to axes.titlesize from rcParams.
        fontweight : str
            Default 'bold'.
        style : 'paren' | 'bracket' | 'plain' | 'upper' | 'upper_paren'
            Label style:
            'paren'       → (a), (b), (c)
            'bracket'     → [a], [b], [c]
            'plain'       → a, b, c
            'upper'       → A, B, C
            'upper_paren' → (A), (B), (C)

        Returns
        -------
        self (for method chaining)
        """
        if fontsize is None:
            fontsize = mpl.rcParams.get("axes.titlesize", 8)

        if labels is None:
            alpha = string.ascii_lowercase
            ALPHA = string.ascii_uppercase
            n = len(self.axes)
            _map = {
                "paren"      : [f"({c})" for c in alpha[:n]],
                "bracket"    : [f"[{c}]" for c in alpha[:n]],
                "plain"      : list(alpha[:n]),
                "upper"      : list(ALPHA[:n]),
                "upper_paren": [f"({c})" for c in ALPHA[:n]],
            }
            if style not in _map:
                raise ValueError(
                    f"style='{style}' not recognised. "
                    f"Choose from: {list(_map)}"
                )
            labels = _map[style]

        for ax, label in zip(self.axes, labels):
            ax.text(
                x, y, label,
                transform=ax.transAxes,
                fontsize=fontsize,
                fontweight=fontweight,
                ha=ha, va=va,
            )
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
        """Save figure to file.

        The format is inferred from the file extension if ``fmt`` is None.
        For publication, use ``.pdf`` or ``.svg`` (vector) or ``.png`` (300 dpi).

        Parameters
        ----------
        path : str or Path, e.g. 'figures/fig1.pdf'
        dpi  : int — resolution for raster formats (default 300).
        transparent : bool — transparent background (default False).
        verbose : bool — print confirmation (default True).

        Returns
        -------
        self (for method chaining)

        Example
        -------
        >>> fig.label_subplots().savefig('figures/fig1.pdf')
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if fmt is None:
            fmt = path.suffix.lstrip(".")
        if not fmt:
            fmt = "pdf"
            path = path.with_suffix(".pdf")
        self._fig.savefig(
            str(path), format=fmt, dpi=dpi,
            transparent=transparent, bbox_inches="tight",
        )
        if verbose:
            print(f"✓ Figure saved → {path}")
        return self

    def show(self) -> "ClimPlot":
        """Display the figure."""
        plt.show()
        return self

    @property
    def fig(self) -> mpl.figure.Figure:
        """The underlying matplotlib Figure."""
        return self._fig
