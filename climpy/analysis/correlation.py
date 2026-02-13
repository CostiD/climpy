"""
climpy.analysis.correlation
============================
Pearson correlation between a time series and a spatial field,
with optional p-value maps for significance stippling.

Functions
---------
pearson_correlation   — fast vectorised correlation (wraps xr.corr)
correlation_pvalue    — correlation + p-values via scipy (for stippling)
"""

from __future__ import annotations

from typing import Optional
import numpy as np
import xarray as xr
from scipy.stats import pearsonr


def pearson_correlation(
    s: xr.DataArray,
    F: xr.DataArray,
    dim: str = "time",
) -> xr.DataArray:
    """Pearson correlation between a 1-D series and every grid point of a field.

    Uses xarray's native ``xr.corr``, which is fully vectorised and NaN-aware.

    Parameters
    ----------
    s   : xr.DataArray — 1-D time series (dimension ``dim``).
    F   : xr.DataArray — 3-D field (dim × lat × lon).
    dim : str — shared time dimension (default 'time').

    Returns
    -------
    xr.DataArray (lat × lon) — Pearson r at each grid point.

    Example
    -------
    >>> r = pearson_correlation(pc1, sst_anom, dim='year')
    """
    return xr.corr(s, F, dim=dim)


def correlation_pvalue(
    s: xr.DataArray,
    F: xr.DataArray,
    dim: str = "time",
    min_valid: int = 10,
) -> tuple[xr.DataArray, xr.DataArray]:
    """Pearson correlation + two-tailed p-value at every grid point.

    This is slower than ``pearson_correlation`` but returns the p-value map
    needed for significance stippling.

    Parameters
    ----------
    s         : xr.DataArray — 1-D time series.
    F         : xr.DataArray — 3-D field (dim × lat × lon).
    dim       : str — time dimension (default 'time').
    min_valid : int — minimum number of valid pairs required to compute r.
                Points with fewer valid pairs get NaN (default 10).

    Returns
    -------
    (corr, pval) : tuple of xr.DataArray, both (lat × lon).

    Example
    -------
    >>> corr, pval = correlation_pvalue(pc1, precip_anom)
    >>> fig.add_stippling(ax, pval.values, lat, lon, pthresh=0.05)
    """
    s_vals = s.values
    spatial_dims = [d for d in F.dims if d != dim]
    lat = F["lat"].values
    lon = F["lon"].values

    corr_map = np.full((len(lat), len(lon)), np.nan)
    pval_map = np.full((len(lat), len(lon)), np.nan)

    for i in range(len(lat)):
        for j in range(len(lon)):
            f_vals = F.values[:, i, j]
            mask = ~(np.isnan(s_vals) | np.isnan(f_vals))
            if mask.sum() >= min_valid:
                r, p = pearsonr(s_vals[mask], f_vals[mask])
                corr_map[i, j] = r
                pval_map[i, j] = p

    coords = {"lat": lat, "lon": lon}
    dims   = ["lat", "lon"]
    corr_da = xr.DataArray(corr_map, coords=coords, dims=dims,
                           attrs={"long_name": "Pearson correlation"})
    pval_da = xr.DataArray(pval_map, coords=coords, dims=dims,
                           attrs={"long_name": "p-value (two-tailed)"})
    return corr_da, pval_da
