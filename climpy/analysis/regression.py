"""
climpy.analysis.regression
===========================
Linear regression of a time series onto a spatial field.

Functions
---------
linear_regression   — regression slopes (fast, vectorised)
regression_pvalue   — slopes + p-values (for stippling)
"""

from __future__ import annotations

import numpy as np
import xarray as xr
from scipy.stats import linregress as _scipy_linregress


def linear_regression(
    s: xr.DataArray,
    F: xr.DataArray,
    dim: str = "time",
) -> xr.DataArray:
    """Regression slope of a field onto a normalised time series.

    Returns the slope β in units of [F] per standard deviation of s.
    Computed as Cov(s, F) / Var(s), fully vectorised.

    Parameters
    ----------
    s   : xr.DataArray — 1-D time series (dimension ``dim``).
    F   : xr.DataArray — 3-D field (dim × lat × lon).
    dim : str — shared time dimension (default 'time').

    Returns
    -------
    xr.DataArray (lat × lon) — regression slopes.

    Example
    -------
    >>> slope = linear_regression(pc1, precip_anom, dim='year')
    """
    s_m = s - s.mean(dim)
    F_m = F - F.mean(dim)
    cov  = (s_m * F_m).sum(dim)
    var_s = (s_m ** 2).sum(dim)
    return cov / var_s


def regression_pvalue(
    s: xr.DataArray,
    F: xr.DataArray,
    dim: str = "time",
    min_valid: int = 10,
) -> tuple[xr.DataArray, xr.DataArray]:
    """Regression slope + two-tailed p-value at every grid point.

    Parameters
    ----------
    s         : xr.DataArray — 1-D time series.
    F         : xr.DataArray — 3-D field (dim × lat × lon).
    dim       : str — time dimension (default 'time').
    min_valid : int — minimum valid pairs needed (default 10).

    Returns
    -------
    (slope, pval) : tuple of xr.DataArray, both (lat × lon).

    Example
    -------
    >>> slope, pval = regression_pvalue(pc1, precip, dim='year')
    >>> fig.add_stippling(ax, pval.values, lat, lon)
    """
    s_vals = s.values
    lat = F["lat"].values
    lon = F["lon"].values

    slope_map = np.full((len(lat), len(lon)), np.nan)
    pval_map  = np.full((len(lat), len(lon)), np.nan)

    for i in range(len(lat)):
        for j in range(len(lon)):
            f_vals = F.values[:, i, j]
            mask = ~(np.isnan(s_vals) | np.isnan(f_vals))
            if mask.sum() >= min_valid:
                result = _scipy_linregress(s_vals[mask], f_vals[mask])
                slope_map[i, j] = result.slope
                pval_map[i, j]  = result.pvalue

    coords = {"lat": lat, "lon": lon}
    dims   = ["lat", "lon"]
    slope_da = xr.DataArray(slope_map, coords=coords, dims=dims,
                            attrs={"long_name": "Regression slope"})
    pval_da  = xr.DataArray(pval_map,  coords=coords, dims=dims,
                            attrs={"long_name": "p-value (two-tailed)"})
    return slope_da, pval_da
