"""
climpy.analysis.correlation
============================
Pearson correlation between a time series and a spatial field,
with optional p-value maps for significance stippling.

Functions
---------
pearson_correlation   — fast vectorised correlation (wraps xr.corr)
correlation_pvalue    — correlation + p-values via scipy (for stippling)

Changes vs. original
--------------------
- BUG FIX: correlation_pvalue now transposes F to ensure time is always
  on axis 0, regardless of the input dimension order.
- NEW: optional Benjamini-Hochberg FDR correction (fdr=True) to control
  the false discovery rate across the spatial grid (Wilks 2006, JClim).
"""

from __future__ import annotations

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


def _fdr_correction(pvals: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """Benjamini-Hochberg FDR correction for a flat array of p-values.

    Only non-NaN values are tested; NaN positions remain NaN.

    Reference: Benjamini & Hochberg (1995); Wilks (2006, JClim).
    """
    flat = pvals.ravel()
    valid = ~np.isnan(flat)
    p_valid = flat[valid]
    m = len(p_valid)

    if m == 0:
        return pvals.copy()

    order = np.argsort(p_valid)
    ranks = np.empty(m, dtype=float)
    ranks[order] = np.arange(1, m + 1)

    p_adj = np.minimum(p_valid * m / ranks, 1.0)
    # Enforce monotonicity (cumulative min from the right)
    p_sorted = p_adj[order]
    p_sorted = np.minimum.accumulate(p_sorted[::-1])[::-1]
    p_adj[order] = p_sorted

    result = flat.copy()
    result[valid] = p_adj
    return result.reshape(pvals.shape)


def correlation_pvalue(
    s: xr.DataArray,
    F: xr.DataArray,
    dim: str = "time",
    min_valid: int = 10,
    fdr: bool = False,
    fdr_alpha: float = 0.05,
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
    fdr       : bool — apply Benjamini-Hochberg FDR correction to p-values
                (recommended for large grids; Wilks 2006). Default False.
    fdr_alpha : float — FDR significance level (default 0.05).

    Returns
    -------
    (corr, pval) : tuple of xr.DataArray, both (lat × lon).
                   If fdr=True, pval contains FDR-adjusted p-values.

    Notes
    -----
    For climate fields with many grid points, individual p < 0.05 tests
    yield ~5% false positives even under the null hypothesis. FDR correction
    (fdr=True) controls this; the stippling threshold (p < 0.05) is applied
    to the adjusted p-values.

    Example
    -------
    >>> corr, pval = correlation_pvalue(pc1, precip_anom, fdr=True)
    >>> fig[0].map(corr).stipple(pval.values, lat, lon, pthresh=0.05)
    """
    s_vals = s.values

    # BUG FIX: transpose so time is always axis 0.
    # Without this, F.values[:, i, j] silently reads the wrong dimension
    # if F was loaded with dims in a different order (e.g. lat x time x lon).
    F_ordered = F.transpose(dim, "lat", "lon")

    lat = F["lat"].values
    lon = F["lon"].values

    corr_map = np.full((len(lat), len(lon)), np.nan)
    pval_map = np.full((len(lat), len(lon)), np.nan)

    for i in range(len(lat)):
        for j in range(len(lon)):
            f_vals = F_ordered.values[:, i, j]
            mask = ~(np.isnan(s_vals) | np.isnan(f_vals))
            if mask.sum() >= min_valid:
                r, p = pearsonr(s_vals[mask], f_vals[mask])
                corr_map[i, j] = r
                pval_map[i, j] = p

    if fdr:
        pval_map = _fdr_correction(pval_map, alpha=fdr_alpha)

    coords = {"lat": lat, "lon": lon}
    dims   = ["lat", "lon"]
    label  = "p-value (two-tailed)" + (" [FDR-adjusted]" if fdr else "")
    corr_da = xr.DataArray(corr_map, coords=coords, dims=dims,
                           attrs={"long_name": "Pearson correlation"})
    pval_da = xr.DataArray(pval_map, coords=coords, dims=dims,
                           attrs={"long_name": label})
    return corr_da, pval_da
