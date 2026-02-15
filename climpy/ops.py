"""
climpy.ops
==========
General-purpose climate operations on xarray objects.

Functions
---------
moving_average          — centred n-year moving average
lag_pair                — build a lagged (series, field) pair
cosine_weights          — area weights proportional to cos(lat)
getneofs                — number of EOFs explaining ≥ P% variance
"""

from __future__ import annotations

from typing import Union
import numpy as np
import xarray as xr


# ── Smoothing ─────────────────────────────────────────────────────────

def moving_average(
    da: xr.DataArray,
    n: int,
    dim: str = "time",
    center: bool = True,
    min_periods: int = None,
) -> xr.DataArray:
    """Centred (or trailing) n-point moving average.

    Parameters
    ----------
    da : xr.DataArray
    n  : int — window size (years or time steps).
    dim : str — dimension along which to smooth (default 'time').
    center : bool — if True (default), window is centred.
    min_periods : int, optional — minimum number of non-NaN points required.

    Returns
    -------
    xr.DataArray, same shape as input (NaN at edges).

    Examples
    --------
    >>> pc_smooth = moving_average(pc, n=7)
    """
    if min_periods is None:
        min_periods = n
    return da.rolling({dim: n}, center=center, min_periods=min_periods).mean()


# Alias for backward compatibility
MA = moving_average


# ── Lag utilities ─────────────────────────────────────────────────────

def lag_pair(
    series: xr.DataArray,
    field: xr.DataArray,
    k: int,
    dim: str = "year",
) -> tuple[xr.DataArray, xr.DataArray]:
    """Build a lagged pair: series leads field by k years.

    Removes the first k time steps from `field` and the last k from `series`
    so that series[t] is paired with field[t + k].

    Parameters
    ----------
    series : 1-D xr.DataArray (e.g. a PC or index).
    field  : 3-D xr.DataArray (time × lat × lon).
    k      : int — lag in number of time steps (must be > 0).
    dim    : str — time dimension name (default 'year').

    Returns
    -------
    (series_lagged, field_lagged)

    Example
    -------
    >>> s_lag, f_lag = lag_pair(pc, field, k=10)
    >>> corr = pearson_correlation(s_lag, f_lag)
    """
    if k <= 0:
        raise ValueError(f"k must be a positive integer, got {k}.")
    return series.isel({dim: slice(None, -k)}), field.isel({dim: slice(k, None)})


# ── Weights ───────────────────────────────────────────────────────────

def cosine_weights(da: xr.DataArray) -> xr.DataArray:
    """Return area weights proportional to cos(lat), suitable for EOF analysis.

    Shape: (lat, 1) so that broadcasting over (lat, lon) works automatically.

    Parameters
    ----------
    da : xr.DataArray with a 'lat' coordinate.

    Returns
    -------
    xr.DataArray of shape (lat, 1).
    """
    coslat = np.cos(np.deg2rad(da.coords["lat"].values)).clip(0.0, 1.0)
    wgts = np.sqrt(coslat)[..., np.newaxis]
    return wgts


# ── EOF helpers ───────────────────────────────────────────────────────

def getneofs(
    solver,
    percent: float = 70.0,
    max_eofs: int = 100,
) -> int:
    """Return the minimum number of EOFs needed to explain >= percent% variance."""
    # Numărul maxim de EOFs posibil = min(n_time, n_space) - 1
    try:
        n_possible = solver._solver._data.shape[0] - 1
    except AttributeError:
        n_possible = max_eofs
    n = min(max_eofs, n_possible)

    var_fracs = solver.varianceFraction(neigs=n).values * 100.0
    cumvar = np.cumsum(var_fracs)
    idx = int(np.searchsorted(cumvar, percent))
    return min(idx + 1, n)


# ── Arithmetic helpers ────────────────────────────────────────────────

def lincomb(
    X: xr.DataArray,
    Y: xr.DataArray,
    a: float = 1.0,
    b: float = 1.0,
) -> xr.DataArray:
    """Linear combination: a·X + b·Y.

    Useful for constructing EOF composites such as AMO and Tripole patterns.

    Parameters
    ----------
    X, Y : xr.DataArray — same shape or broadcastable.
    a, b : float — coefficients.

    Example
    -------
    >>> eof_amo    = lincomb(eof1, eof2, a=1, b=1)
    >>> eof_tripol = lincomb(eof1, eof2, a=1, b=-1)
    """
    return a * X + b * Y
