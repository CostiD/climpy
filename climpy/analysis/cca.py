"""
climpy.analysis.cca
====================
Batch-PCA prefiltered Canonical Correlation Analysis (BP-CCA).

Example
-------
>>> from climpy.analysis import CCA
>>> cca = CCA(sst_anom, slp_anom, n_eofs=(10, 15))
>>> cca.summary()
>>> Xpat = cca.x_patterns()     # (mode × lat × lon)
>>> Ypat = cca.y_patterns()     # (mode × lat × lon)
>>> Xts  = cca.x_timeseries()   # (time × mode)
>>> Yts  = cca.y_timeseries()   # (time × mode)
>>> corrs = cca.canonical_correlations()
"""

from __future__ import annotations

from typing import Optional
import numpy as np
import xarray as xr
from eofs.xarray import Eof as _Eof

from climpy.ops import cosine_weights, getneofs


class CCA:
    """Batch-PCA prefiltered Canonical Correlation Analysis.

    The algorithm:
    1. Pre-filter X and Y with EOF analysis to reduce dimensionality.
    2. Compute the covariance matrix of the retained PCs.
    3. Apply SVD to find the canonical correlation patterns (CCPs).
    4. Project back onto the original coordinate space.

    Parameters
    ----------
    X, Y : xr.DataArray
        Climate fields (anomalies / de-trended) with dimensions
        (time, lat, lon).  The time dimension must be the same length.
    n_eofs : (int, int), optional
        Number of EOFs to retain for X and Y respectively.
        If None, chosen automatically to explain ≥ ``min_variance``% of variance.
    min_variance : float
        Minimum cumulative explained variance (%) for automatic EOF selection.
    lat_weights : bool
        Apply cosine-latitude weighting (default True).

    Key attributes (after fitting)
    --------------------------------
    n_modes : int      — number of canonical modes (= min(kX, kY))
    cancorrs : ndarray — canonical correlations for each mode
    """

    def __init__(
        self,
        X: xr.DataArray,
        Y: xr.DataArray,
        n_eofs: Optional[tuple[int, int]] = None,
        min_variance: float = 70.0,
        lat_weights: bool = True,
    ):
        self._X_orig = X
        self._Y_orig = Y

        # --- align time axes ---
        X, Y = self._align_time(X, Y)
        self._X = X
        self._Y = Y

        # --- EOF pre-filtering ---
        wX = cosine_weights(X) if lat_weights else None
        wY = cosine_weights(Y) if lat_weights else None
        solverX = _Eof(X, weights=wX)
        solverY = _Eof(Y, weights=wY)

        if n_eofs is None:
            kX = getneofs(solverX, percent=min_variance)
            kY = getneofs(solverY, percent=min_variance)
        else:
            kX, kY = n_eofs

        self._kX = kX
        self._kY = kY
        self.n_modes = min(kX, kY)
        modes = np.arange(self.n_modes)

        # --- extract PCs and EOFs ---
        pcsX  = solverX.pcs(npcs=kX, pcscaling=1).values          # (T × kX)
        pcsY  = solverY.pcs(npcs=kY, pcscaling=1).values          # (T × kY)
        eofsX = solverX.eofs(neofs=kX, eofscaling=2)              # (kX × lat × lon)
        eofsY = solverY.eofs(neofs=kY, eofscaling=2)

        # stack to (sX × kX) — drop NaN grid points
        eXstacked = eofsX.transpose("lat", "lon", "mode").stack(s=["lat", "lon"])
        eYstacked = eofsY.transpose("lat", "lon", "mode").stack(s=["lat", "lon"])
        self._eXstacked = eXstacked.dropna(dim="s")
        self._eYstacked = eYstacked.dropna(dim="s")
        eX = self._eXstacked.values.T          # (sX' × kX)
        eY = self._eYstacked.values.T          # (sY' × kY)

        # --- total variance (sum of eigenvalues) ---
        self._totvarX = float(solverX.totalAnomalyVariance())
        self._totvarY = float(solverY.totalAnomalyVariance())

        # --- CCA via SVD ---
        Sxy = pcsX.T @ pcsY / (len(pcsX) - 1)           # covariance matrix
        fX, cancorrs, fYT = np.linalg.svd(Sxy, full_matrices=False)
        fY = fYT.T

        # Truncate to n_modes
        fX = fX[:, :self.n_modes]
        fY = fY[:, :self.n_modes]
        self.cancorrs = cancorrs[:self.n_modes]

        # --- back-project to physical space ---
        self._Fx = eX @ fX                               # (sX' × n_modes)
        self._Fy = eY @ fY                               # (sY' × n_modes)
        self._tsX = pcsX @ fX                            # (T × n_modes)
        self._tsY = pcsY @ fY                            # (T × n_modes)

        self._modes = modes

    # ── Static helper ─────────────────────────────────────────────────

    @staticmethod
    def _align_time(X: xr.DataArray, Y: xr.DataArray):
        """Select the common time interval of X and Y."""
        t_dim = "time" if "time" in X.dims else "year"
        tmin = max(X[t_dim].values[0], Y[t_dim].values[0])
        tmax = min(X[t_dim].values[-1], Y[t_dim].values[-1])
        return (X.sel({t_dim: slice(tmin, tmax)}),
                Y.sel({t_dim: slice(tmin, tmax)}))

    # ── Output helpers ────────────────────────────────────────────────

    def _unstack_pattern(
        self,
        F_stacked: np.ndarray,      # (s' × n_modes)
        eof_xarr_stacked,           # stacked DataArray holding coordinates
        field: xr.DataArray,        # original field (for shape / coords)
        dim_name: str,              # 's' stacked dimension name
        var_name: str,
    ) -> xr.DataArray:
        """Reindex and unstack a CCP from stacked to (mode, lat, lon)."""
        full_s = field.stack({dim_name: ["lat", "lon"]})[dim_name]
        da_s = xr.DataArray(
            F_stacked,
            coords=[eof_xarr_stacked[dim_name].values, self._modes],
            dims=[dim_name, "mode"],
        )
        da_s = da_s.reindex({dim_name: full_s.values})
        vals = da_s.values.reshape(
            field.sizes["lat"], field.sizes["lon"], self.n_modes
        )
        return xr.DataArray(
            vals,
            coords=[field["lat"].values, field["lon"].values, self._modes],
            dims=["lat", "lon", "mode"],
            name=var_name,
        ).transpose("mode", "lat", "lon")

    # ── Public API ────────────────────────────────────────────────────

    def x_patterns(self) -> xr.DataArray:
        """X canonical correlation patterns.

        Returns
        -------
        xr.DataArray (mode × lat × lon) named 'X_CCP'.
        """
        return self._unstack_pattern(
            self._Fx, self._eXstacked, self._X, "sX", "X_CCP"
        )

    def y_patterns(self) -> xr.DataArray:
        """Y canonical correlation patterns.

        Returns
        -------
        xr.DataArray (mode × lat × lon) named 'Y_CCP'.
        """
        return self._unstack_pattern(
            self._Fy, self._eYstacked, self._Y, "sY", "Y_CCP"
        )

    def x_timeseries(self) -> xr.DataArray:
        """Canonical time series associated with X.

        Returns
        -------
        xr.DataArray (time × mode) named 'X_ts'.
        """
        t_dim = "time" if "time" in self._X.dims else "year"
        return xr.DataArray(
            self._tsX,
            coords=[self._X[t_dim].values, self._modes],
            dims=[t_dim, "mode"],
            name="X_ts",
        )

    def y_timeseries(self) -> xr.DataArray:
        """Canonical time series associated with Y.

        Returns
        -------
        xr.DataArray (time × mode) named 'Y_ts'.
        """
        t_dim = "time" if "time" in self._Y.dims else "year"
        return xr.DataArray(
            self._tsY,
            coords=[self._Y[t_dim].values, self._modes],
            dims=[t_dim, "mode"],
            name="Y_ts",
        )

    def canonical_correlations(self) -> np.ndarray:
        """Canonical correlations for each mode.

        Returns
        -------
        numpy.ndarray, shape (n_modes,)
        """
        return self.cancorrs

    def variance_fraction(self) -> tuple[np.ndarray, np.ndarray]:
        """Explained variance fractions (%) for X and Y CCPs.

        Returns
        -------
        (varX, varY) : tuple of numpy.ndarray, shape (n_modes,)
        """
        varX = np.sum(self._Fx ** 2, axis=0) / self._totvarX
        varY = np.sum(self._Fy ** 2, axis=0) / self._totvarY
        return varX * 100.0, varY * 100.0

    def summary(self) -> str:
        """Print a table of modes, canonical correlations, and variance fractions."""
        varX, varY = self.variance_fraction()
        lines = [
            f"  CCA Summary  (kX={self._kX}, kY={self._kY}, n_modes={self.n_modes})",
            f"{'Mode':>6}  {'r_canon':>8}  {'Var X (%)':>10}  {'Var Y (%)':>10}",
            "-" * 42,
        ]
        for i in range(self.n_modes):
            lines.append(
                f"{i+1:>6}  {self.cancorrs[i]:>8.3f}  "
                f"{varX[i]:>10.2f}  {varY[i]:>10.2f}"
            )
        result = "\n".join(lines)
        print(result)
        return result
