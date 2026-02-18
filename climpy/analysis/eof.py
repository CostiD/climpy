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

Changes vs. original
--------------------
- Minor fix: fake datetime coordinate for year-based data now starts on
  January 1 (not July 1), making date_range consistent across all periods.
"""
from __future__ import annotations

from typing import Optional
import numpy as np
import pandas as pd
import xarray as xr
from eofs.xarray import Eof as _Eof

from climpy.ops import cosine_weights, getneofs


class EOF:
    """EOF (Principal Component) analysis of a climate field."""

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
                raise ValueError(
                    "Nu găsesc o dimensiune temporală ('time' sau 'year')."
                )
        self._time_dim = time_dim

        # eofs necesită 'time' cu dtype datetime64
        if time_dim != "time" or not np.issubdtype(
            da[time_dim].dtype, np.datetime64
        ):
            years = da[time_dim].values.astype(int)
            fake_time = xr.DataArray(
                da.values,
                dims=["time"] + [d for d in da.dims if d != time_dim],
                coords={
                    "time": pd.date_range(
                        # FIX: start from Jan 1 (not Jul 1) so all periods
                        # are exactly 12 months and date_range is consistent.
                        start=f"{years[0]}-01-01",
                        periods=len(years),
                        freq="YS",
                    ),
                    **{
                        k: v
                        for k, v in da.coords.items()
                        if k != time_dim
                    },
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

    def eofs(self, n: int = None, scaling: int = 2) -> xr.DataArray:
        """Returnează pattern-urile EOF (mode × lat × lon).

        scaling=2: EOFs scalate cu sqrt(eigenvalue) → unități fizice ale datelor.
        """
        n = n or self.n_eofs
        return self.solver.eofs(neofs=n, eofscaling=scaling)

    def pcs(self, n: int = None, scaling: int = 1) -> xr.DataArray:
        """Returnează seriile de timp PC (time × mode).

        scaling=1: PCs normalizate la varianță unitară (std ≈ 1).
        """
        n = n or self.n_eofs
        return self.solver.pcs(npcs=n, pcscaling=scaling)

    def variance_fraction(self, n: int = None) -> xr.DataArray:
        """Varianța explicată (%) pentru fiecare EOF."""
        n = n or self.n_eofs
        return self.solver.varianceFraction(neigs=n) * 100.0

    def total_variance(self) -> float:
        """Varianța totală a câmpului de intrare."""
        return float(self.solver.totalAnomalyVariance())

    def summary(self) -> str:
        """Printează tabel cu moduri, varianță, varianță cumulată."""
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
