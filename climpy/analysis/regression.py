"""
climpy.analysis.regression
===========================
Linear regression of a time series onto a spatial field.

Functions
---------
linear_regression   — regression slopes (fast, vectorised)
regression_pvalue   — slopes + p-values (for stippling)

Changes vs. original
--------------------
- BUG FIX: regression_pvalue now transposes F to ensure time is always
  on axis 0, regardless of the input dimension order.
- BUG FIX: linear_regression now normalises s by its std so that the
  returned slope is truly [F units per 1 std of s], as stated in the docstring.
- NEW: autocorr_correction=True in regression_pvalue uses effective sample
  size (Quenouille 1952 / Chelton 1983) to correct for AR(1) autocorrelation,
  which inflates degrees of freedom in climate time series.
- NEW: optional Benjamini-Hochberg FDR correction (fdr=True).
"""

from __future__ import annotations

import numpy as np
import xarray as xr
from scipy.stats import t as t_dist
from scipy.stats import linregress as _scipy_linregress

from climpy.analysis.correlation import _fdr_correction


def linear_regression(
    s: xr.DataArray,
    F: xr.DataArray,
    dim: str = "time",
    normalise: bool = True,
) -> xr.DataArray:
    """Regression slope of a field onto a time series.

    Returns the slope β. If ``normalise=True`` (default), s is divided by
    its own standard deviation so β is in units of [F] per 1 std of s.

    Parameters
    ----------
    s         : xr.DataArray — 1-D time series (dimension ``dim``).
    F         : xr.DataArray — 3-D field (dim × lat × lon).
    dim       : str — shared time dimension (default 'time').
    normalise : bool — divide s by its std before regression (default True).
                Set False to get raw OLS slope in [F units / s units].

    Returns
    -------
    xr.DataArray (lat × lon) — regression slopes.

    Notes
    -----
    Fully vectorised via broadcasting — fast for large grids.

    Example
    -------
    >>> slope = linear_regression(pc1, precip_anom, dim='year')
    """
    s_m = s - s.mean(dim)
    if normalise:
        std_s = s_m.std(dim)
        if float(std_s) > 0:
            s_m = s_m / std_s

    F_m  = F - F.mean(dim)
    cov  = (s_m * F_m).sum(dim)
    var_s = (s_m ** 2).sum(dim)
    return cov / var_s


def _effective_n(x: np.ndarray, y: np.ndarray) -> int:
    """Estimate effective sample size correcting for AR(1) autocorrelation.

    Uses the Quenouille (1952) approximation as in Chelton (1983):
        N_eff = N * (1 - r1x * r1y) / (1 + r1x * r1y)
    where r1x, r1y are lag-1 autocorrelations of x and y respectively.

    Returns at least 4 to avoid degenerate t-tests.
    """
    n = len(x)
    if n < 4:
        return n
    r1x = float(np.corrcoef(x[:-1], x[1:])[0, 1]) if n > 2 else 0.0
    r1y = float(np.corrcoef(y[:-1], y[1:])[0, 1]) if n > 2 else 0.0
    denom = 1.0 + r1x * r1y
    if denom <= 0:
        return n
    neff = n * (1.0 - r1x * r1y) / denom
    return max(4, int(neff))


def regression_pvalue(
    s: xr.DataArray,
    F: xr.DataArray,
    dim: str = "time",
    min_valid: int = 10,
    normalise: bool = True,
    autocorr_correction: bool = True,
    fdr: bool = False,
    fdr_alpha: float = 0.05,
) -> tuple[xr.DataArray, xr.DataArray]:
    """Regression slope + two-tailed p-value at every grid point.

    Parameters
    ----------
    s                    : xr.DataArray — 1-D time series.
    F                    : xr.DataArray — 3-D field (dim × lat × lon).
    dim                  : str — time dimension (default 'time').
    min_valid            : int — minimum valid pairs needed (default 10).
    normalise            : bool — normalise s by its std (default True).
    autocorr_correction  : bool — correct for AR(1) autocorrelation using
                           effective N (Quenouille 1952). Default True.
                           IMPORTANT: climate time series typically have
                           positive AR(1) autocorrelation, which inflates
                           degrees of freedom and makes p-values too small
                           (excess false positives) without this correction.
    fdr                  : bool — apply Benjamini-Hochberg FDR correction
                           (Wilks 2006). Default False.
    fdr_alpha            : float — FDR target level (default 0.05).

    Returns
    -------
    (slope, pval) : tuple of xr.DataArray, both (lat × lon).

    Example
    -------
    >>> slope, pval = regression_pvalue(pc1, precip, dim='year',
    ...                                 autocorr_correction=True, fdr=True)
    >>> fig[0].map(slope).stipple(pval.values, lat, lon)
    """
    s_vals = s.values
    if normalise:
        std_s = np.nanstd(s_vals)
        if std_s > 0:
            s_vals = s_vals / std_s

    # BUG FIX: transpose so time is always axis 0.
    F_ordered = F.transpose(dim, "lat", "lon")

    lat = F["lat"].values
    lon = F["lon"].values

    slope_map = np.full((len(lat), len(lon)), np.nan)
    pval_map  = np.full((len(lat), len(lon)), np.nan)

    for i in range(len(lat)):
        for j in range(len(lon)):
            f_vals = F_ordered.values[:, i, j]
            mask = ~(np.isnan(s_vals) | np.isnan(f_vals))
            n = int(mask.sum())
            if n < min_valid:
                continue

            sv = s_vals[mask]
            fv = f_vals[mask]

            # Slope via OLS
            result = _scipy_linregress(sv, fv)
            slope_map[i, j] = result.slope

            # P-value with or without autocorrelation correction
            if autocorr_correction:
                neff = _effective_n(sv, fv)
                if neff < 3:
                    continue
                # Recompute t-stat with effective df
                r = result.rvalue
                if abs(r) >= 1.0:
                    pval_map[i, j] = 0.0
                    continue
                t_stat = r * np.sqrt((neff - 2) / (1 - r**2))
                pval_map[i, j] = 2.0 * t_dist.sf(abs(t_stat), df=neff - 2)
            else:
                pval_map[i, j] = result.pvalue

    if fdr:
        pval_map = _fdr_correction(pval_map, alpha=fdr_alpha)

    coords = {"lat": lat, "lon": lon}
    dims   = ["lat", "lon"]
    label  = "p-value (two-tailed)" + (" [FDR-adjusted]" if fdr else "")
    slope_da = xr.DataArray(slope_map, coords=coords, dims=dims,
                             attrs={"long_name": "Regression slope"})
    pval_da  = xr.DataArray(pval_map,  coords=coords, dims=dims,
                             attrs={"long_name": label})
    return slope_da, pval_da
