"""
climpy.analysis.eof
===================
Wrapper around ``eofs.xarray.Eof`` with sensible defaults.

Example
-------
>>> from climpy.analysis import EOF
>>> solver = EOF(sst_anomalies, n_eofs=10)
>>> print(solver.summary())
>>> eofs = solver.eofs()          # xr.DataArray (mode × lat × lon)
>>> pcs  = solver.pcs()           # xr.DataArray (time × mode)
>>> frac = solver.variance_fraction()
"""

from __future__ import annotations

from typing import Optional
import numpy as np
import pandas as pd
import xarray as xr
from eofs.xarray import Eof as _Eof

from climpy.ops import cosine_weights, getneofs


class EOF:
    """EOF (Principal Component) analysis of a climate field.

    Wraps ``eofs.xarray.Eof`` with:
    - automatic cosine-latitude weighting
    - automatic selection of the number of modes (variance threshold)
    - clean xarray output

    Parameters
    ----------
    da : xr.DataArray
        Monthly or annual data with dimensions (time/year, lat, lon).
        Should be anomalies with the mean already removed.
    n_eofs : int, optional
        Number of EOFs to retain. If None, chosen automatically so that
        the cumulative explained variance ≥ ``min_variance``.
    min_variance : float
        Minimum cumulative explained variance (%) used when n_eofs is None.
        Default 70.
    lat_weights : bool
        If True (default), apply cosine-latitude area weighting.
    time_dim : str
        Name of the time dimension (default 'time').

    Attributes
    ----------
    n_eofs : int
        Number of retained EOFs.
    solver : eofs.xarray.Eof
        The underlying solver (gives access to all eofs methods).
    """

    def __init__(
    self,
    da: xr.DataArray,
    n_eofs=None,
    min_variance: float = 70.0,
    lat_weights: bool = True,
    time_dim: str = None,
):
    self._da = da

    # Detectează automat dimensiunea temporală
    if time_dim is None:
        if "time" in da.dims:
            time_dim = "time"
        elif "year" in da.dims:
            time_dim = "year"
        else:
            raise ValueError("Nu găsesc o dimensiune temporală ('time' sau 'year').")
    self._time_dim = time_dim

    # eofs necesită coordonata să se numească 'time' cu dtype datetime64
    # Dacă avem 'year' (int), facem o copie temporară cu 'time' datetime64
    if time_dim != "time" or not np.issubdtype(da[time_dim].dtype, np.datetime64):
        years = da[time_dim].values.astype(int)
        fake_time = xr.DataArray(
            da.values,
            dims=["time"] + [d for d in da.dims if d != time_dim],
            coords={
                "time": pd.date_range(
                    start=f"{years[0]}-07-01",
                    periods=len(years),
                    freq="YS"
                ),
                **{k: v for k, v in da.coords.items()
                   if k != time_dim},
            },
        )
        da_for_eof = fake_time
    else:
        da_for_eof = da

    # Weights
    if lat_weights:
        wgts = cosine_weights(da_for_eof)
    else:
        wgts = None

    self.solver = _Eof(da_for_eof, weights=wgts)

    # Număr de EOFs
    if n_eofs is None:
        self.n_eofs = getneofs(self.solver, percent=min_variance)
    else:
        self.n_eofs = n_eofs

    # ── Main outputs ──────────────────────────────────────────────────

    def eofs(
        self,
        n: int = None,
        scaling: int = 2,
    ) -> xr.DataArray:
        """Return spatial EOF patterns.

        Parameters
        ----------
        n : int, optional
            Number of EOFs (default: self.n_eofs).
        scaling : {0, 1, 2}
            0 — orthonormal (unit variance)
            1 — eigenvalue scaled (large = important)
            2 — variance fraction scaled (units of data) ← **default**

        Returns
        -------
        xr.DataArray (mode × lat × lon)
        """
        n = n or self.n_eofs
        return self.solver.eofs(neofs=n, eofscaling=scaling)

    def pcs(
        self,
        n: int = None,
        scaling: int = 1,
    ) -> xr.DataArray:
        """Return principal component time series.

        Parameters
        ----------
        n : int, optional
        scaling : {0, 1}
            0 — raw (unit variance in EOF space)
            1 — standard deviation scaled ← **default**

        Returns
        -------
        xr.DataArray (time × mode)
        """
        n = n or self.n_eofs
        return self.solver.pcs(npcs=n, pcscaling=scaling)

    def variance_fraction(self, n: int = None) -> xr.DataArray:
        """Explained variance fraction (%) for each EOF.

        Returns
        -------
        xr.DataArray (mode,) in percent.
        """
        n = n or self.n_eofs
        return self.solver.varianceFraction(neigs=n) * 100.0

    def total_variance(self) -> float:
        """Total anomaly variance of the input field."""
        return float(self.solver.totalAnomalyVariance())

    # ── Summary ───────────────────────────────────────────────────────

    def summary(self) -> str:
        """Print a table of EOF modes, explained variance, and cumulative variance."""
        fracs = self.variance_fraction().values
        cumvar = np.cumsum(fracs)
        lines = [
            f"{'Mode':>6}  {'Var (%)':>9}  {'Cum. var (%)':>13}",
            "-" * 34,
        ]
        for i, (f, c) in enumerate(zip(fracs, cumvar)):
            lines.append(f"{i+1:>6}  {f:>9.2f}  {c:>13.2f}")
        lines.append("-" * 34)
        lines.append(f"Total anomaly variance: {self.total_variance():.4f}")
        result = "\n".join(lines)
        print(result)
        return result
