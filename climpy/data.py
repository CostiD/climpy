"""
climpy.data
===========
Climate data loading and preprocessing.

All functions are non-interactive: parameters are passed explicitly.
This makes scripts reproducible, batch-runnable, and testable.

Dataset reference — raw vs already-anomalies
---------------------------------------------
| Dataset          | Type              | Note                           |
|------------------|-------------------|--------------------------------|
| SST ERSSTv5      | raw monthly means | preprocess computes anomalies  |
| SLP NCEP-R1      | raw monthly means | preprocess computes anomalies  |
| GPCP v2.3 Precip | raw monthly means | preprocess computes anomalies  |
| GISTEMP SAT      | already anomalies | use already_anomalies=True     |
| Berkeley Earth   | already anomalies | use load_berkeley_earth()      |

Typical workflow (raw data)
----------------------------
>>> import climpy.data as cd
>>> ds   = cd.load_nc("data/sst.nc", var="sst")
>>> anom = cd.preprocess("data/sst.nc", var="sst",
...            lat=(0,80), lon=(-80,20),
...            time=("1950-01","2020-12"),
...            ref_period=("1981-01","2010-12"),
...            subtract_gm=True)['annual_no_gm']

Typical workflow (already-anomaly data — GISTEMP)
--------------------------------------------------
>>> result = cd.preprocess("data/sat.nc", var="air",
...              time=("1979-01","2020-12"),
...              already_anomalies=True)          # ← no double anomalising
>>> sat_djfm = cd.seasonal_means(result['anom'], months=[12,1,2,3])

Typical workflow (Berkeley Earth)
----------------------------------
>>> da = cd.load_berkeley_earth("data/best.nc",
...          time=("1979-01","2020-12"))
>>> sat_djfm = cd.seasonal_means(da, months=[12,1,2,3])
"""

from __future__ import annotations

from typing import Optional, Union
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


# ── Loading ───────────────────────────────────────────────────────────────────

def load_nc(
    path: Union[str, Path],
    var: Optional[str] = None,
    squeeze: Optional[Union[str, list]] = None,
    decode_times: bool = True,
) -> xr.Dataset:
    """Load a NetCDF file into an xarray.Dataset."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    ds = xr.open_dataset(path, decode_times=decode_times)

    if squeeze is not None:
        if isinstance(squeeze, str):
            squeeze = [squeeze]
        for dim in squeeze:
            if dim in ds.dims:
                ds = ds.squeeze(dim, drop=True)

    if var is not None:
        if var not in ds:
            raise KeyError(
                f"Variable '{var}' not found. "
                f"Available: {list(ds.data_vars)}"
            )
        ds = ds[[var]]

    return ds


def fix_time(
    ds: xr.Dataset,
    start: str,
    freq: str = "MS",
) -> xr.Dataset:
    """Replace a numeric/cftime time coordinate with a proper DatetimeIndex."""
    n = ds.sizes["time"]
    ds["time"] = pd.date_range(start=start, periods=n, freq=freq)
    return ds


def load_berkeley_earth(
    path: Union[str, Path],
    time: Optional[tuple] = None,
    lat: Optional[tuple] = None,
    lon: Optional[tuple] = None,
    lon_convention: str = "[-180,180]",
) -> xr.DataArray:
    """Load Berkeley Earth Land+Ocean gridded temperature (LatLong1 format).

    Berkeley Earth stores time as a decimal year (e.g. 1981.125 = Feb 1981)
    and uses 'latitude'/'longitude' coordinate names. This function handles
    all of that transparently and returns a standard climpy DataArray with
    dims (time, lat, lon) and a proper DatetimeIndex.

    The returned DataArray contains **temperature anomalies** relative to
    1951-1980 — i.e. it is already an anomaly field, ready to pass directly
    to ``seasonal_means`` or ``annual_means`` without calling ``anomalies()``.

    Parameters
    ----------
    path           : Path to the Berkeley Earth NetCDF file.
    time           : Optional (start, end) tuple of strings e.g. ('1979-01', '2020-12').
    lat            : Optional (lat_min, lat_max) tuple.
    lon            : Optional (lon_min, lon_max) tuple.
    lon_convention : '[-180,180]' (default) or '[0,360]'.

    Returns
    -------
    xr.DataArray — (time, lat, lon), anomalies in °C rel. 1951-1980.

    Download URL
    ------------
    https://berkeley-earth-temperature.s3.amazonaws.com/Global/Gridded/Land_and_Ocean_LatLong1.nc

    Example
    -------
    >>> da = load_berkeley_earth('data/best.nc', time=('1979-01','2020-12'))
    >>> sat_djfm = climpy.seasonal_means(da, months=[12,1,2,3])
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Berkeley Earth file not found: {path}")

    # Berkeley Earth time is decimal year: 1981.125 → Feb 1981
    # decode_times=False prevents xarray from choking on "year A.D." units
    ds = xr.open_dataset(path, decode_times=False)

    # Convert decimal-year time → proper DatetimeIndex
    dec_years = ds["time"].values.astype(float)
    years      = dec_years.astype(int)
    # fraction of year → month index (0-based)
    months_f   = (dec_years - years) * 12.0
    months_i   = np.round(months_f).astype(int) % 12  # 0 = Jan
    dates = pd.to_datetime(
        [f"{y}-{m+1:02d}-01" for y, m in zip(years, months_i)]
    )
    ds["time"] = dates

    # Rename Berkeley Earth coords to climpy standard
    rename = {}
    if "latitude"  in ds.coords: rename["latitude"]  = "lat"
    if "longitude" in ds.coords: rename["longitude"] = "lon"
    if rename:
        ds = ds.rename(rename)

    da = ds["temperature"]

    # Sort lat/lon
    if "lat" in da.coords:
        da = da.isel(lat=np.argsort(da["lat"].values))
    if "lon" in da.coords:
        da = da.isel(lon=np.argsort(da["lon"].values))

    # Longitude convention
    lon_vals = da["lon"].values
    if lon_convention == "[-180,180]" and lon_vals.max() > 180:
        da = da.assign_coords(lon=(((lon_vals + 180) % 360) - 180))
        da = da.sortby("lon")
    elif lon_convention == "[0,360]" and lon_vals.min() < 0:
        da = da.assign_coords(lon=(lon_vals % 360))
        da = da.sortby("lon")

    # Temporal subset
    if time is not None:
        da = da.sel(time=slice(str(time[0]), str(time[1])))

    # Spatial subset
    sel = {}
    if lat is not None:
        sel["lat"] = slice(float(lat[0]), float(lat[1]))
    if lon is not None:
        sel["lon"] = slice(float(lon[0]), float(lon[1]))
    if sel:
        da = da.sel(sel)

    da.attrs["description"] = (
        "Berkeley Earth Land+Ocean temperature anomaly "
        "(°C, relative to 1951-1980 baseline)"
    )
    da.attrs["source"] = "Berkeley Earth, Land_and_Ocean_LatLong1.nc"
    return da


# ── Coordinate normalisation ──────────────────────────────────────────────────

def standardise_coords(
    ds: xr.Dataset,
    time: Optional[str] = None,
    lat: Optional[str] = None,
    lon: Optional[str] = None,
    var: Optional[str] = None,
) -> xr.Dataset:
    """Rename coordinates and variables to the standard names used by climpy."""
    rename = {}
    if time and time != "time":
        rename[time] = "time"
    if lat and lat != "lat":
        rename[lat] = "lat"
    if lon and lon != "lon":
        rename[lon] = "lon"
    if var and var != "Xdata":
        rename[var] = "Xdata"

    for name in list(ds.data_vars):
        if name == "__xarray_dataarray_variable__":
            rename[name] = "Xdata"
            break

    if rename:
        ds = ds.rename(rename)

    if "lat" in ds.coords:
        ds = ds.isel(lat=np.argsort(ds["lat"].values))
    if "lon" in ds.coords:
        ds = ds.isel(lon=np.argsort(ds["lon"].values))

    return ds


def set_lon_convention(
    ds: xr.Dataset,
    convention: str = "[-180,180]",
) -> xr.Dataset:
    """Convert longitude convention."""
    if "lon" not in ds.coords:
        raise KeyError("Dataset has no 'lon' coordinate. Run standardise_coords first.")

    lon = ds["lon"].values

    if convention == "[-180,180]":
        if lon.max() <= 180:
            return ds
        ds = ds.assign_coords(lon=(((lon + 180) % 360) - 180))
        ds = ds.sortby("lon")

    elif convention == "[0,360]":
        if lon.min() >= 0:
            return ds
        ds = ds.assign_coords(lon=(lon % 360))
        ds = ds.sortby("lon")

    else:
        raise ValueError(f"convention must be '[-180,180]' or '[0,360]', got '{convention}'")

    return ds


def mask_fill_values(
    ds: xr.Dataset,
    fill_value: float = -1000.0,
    abs_threshold: float = 1e29,
) -> xr.Dataset:
    """Replace unrealistic fill values with NaN."""
    ds = ds.where(ds != fill_value)
    ds = ds.where(np.abs(ds) < abs_threshold)
    return ds


# ── Spatial and temporal subsetting ──────────────────────────────────────────

def subset(
    ds: Union[xr.Dataset, xr.DataArray],
    time: Optional[tuple] = None,
    lat: Optional[tuple] = None,
    lon: Optional[tuple] = None,
) -> Union[xr.Dataset, xr.DataArray]:
    """Select a spatio-temporal subregion."""
    sel = {}
    if time is not None:
        sel["time"] = slice(str(time[0]), str(time[1]))
    if lat is not None:
        sel["lat"] = slice(float(lat[0]), float(lat[1]))
    if lon is not None:
        sel["lon"] = slice(float(lon[0]), float(lon[1]))
    if sel:
        ds = ds.sel(sel)
    return ds


def reduce_resolution(
    da: xr.DataArray,
    lat_step: int = 1,
    lon_step: int = 1,
) -> xr.DataArray:
    """Sub-sample spatial resolution by taking every nth grid point."""
    return da.isel(lat=slice(None, None, lat_step),
                   lon=slice(None, None, lon_step))


# ── Anomalies ─────────────────────────────────────────────────────────────────

def anomalies(
    da: xr.DataArray,
    ref_period: Optional[tuple] = None,
) -> xr.DataArray:
    """Compute monthly anomalies by subtracting the monthly climatology.

    anom[i,t] = da[i,t] - mean(da[i, months in ref_period matching month t])

    Parameters
    ----------
    da         : xr.DataArray with a 'time' dimension.
    ref_period : (start, end) strings for climatology, e.g. ('1981-01','2010-12').
                 If None, uses the full time range.

    Returns
    -------
    xr.DataArray — monthly anomalies.

    Note
    ----
    Do NOT call this on datasets that are already anomalies (e.g. GISTEMP,
    Berkeley Earth). Use ``preprocess(..., already_anomalies=True)`` or
    ``load_berkeley_earth()`` instead.
    """
    if ref_period is not None:
        ref = da.sel(time=slice(str(ref_period[0]), str(ref_period[1])))
    else:
        ref = da

    clim = ref.groupby("time.month").mean("time")
    anom = da.groupby("time.month") - clim
    anom.attrs = {**da.attrs, "description": "Monthly anomalies"}
    return anom


# ── Temporal averaging ────────────────────────────────────────────────────────

def annual_means(
    da: xr.DataArray,
    time_dim: str = "time",
) -> xr.DataArray:
    """Compute annual (calendar-year) means.

    Only years with all 12 months present receive a valid value.
    """
    ann = da.resample({time_dim: "YS"}).mean(time_dim)
    counts = da.resample({time_dim: "YS"}).count(time_dim)
    ann = ann.where(counts >= 12)
    ann = ann.dropna(time_dim, how="all")

    years = ann[time_dim].dt.year.values
    ann = ann.assign_coords({time_dim: years}).rename({time_dim: "year"})
    ann.attrs = {**da.attrs, "description": "Annual means"}
    return ann


def seasonal_means(
    da: xr.DataArray,
    months: list,
    time_dim: str = "time",
) -> xr.DataArray:
    """Compute multi-month seasonal means."""
    months_set = set(months)
    n_months   = len(months)

    time_pd    = pd.DatetimeIndex(da[time_dim].values)
    month_mask = np.isin(time_pd.month, list(months_set))
    da_sel     = da.isel({time_dim: month_mask})

    if da_sel.sizes[time_dim] == 0:
        raise ValueError(f"Nu există date pentru lunile {months}.")

    time_sel      = pd.DatetimeIndex(da_sel[time_dim].values)
    is_cross_year = 12 in months_set and any(m < 6 for m in months_set)

    if is_cross_year:
        season_years = time_sel.year + (time_sel.month == 12).astype(int)
    else:
        season_years = time_sel.year

    unique_years = np.unique(season_years)
    slices       = []
    valid_years  = []

    for yr in unique_years:
        mask  = season_years == yr
        count = int(mask.sum())
        chunk = da_sel.isel({time_dim: mask}).mean(time_dim)
        if count < n_months:
            chunk = chunk * np.nan
        slices.append(chunk)
        valid_years.append(yr)

    seas         = xr.concat(slices, dim="year")
    seas["year"] = np.array(valid_years, dtype=int)
    seas.attrs   = {**da.attrs, "description": f"Seasonal means (months={months})"}
    seas         = seas.dropna("year", how="all")
    return seas


# ── Global mean removal ───────────────────────────────────────────────────────

def global_mean(
    da: xr.DataArray,
    lat_weights: bool = True,
) -> xr.DataArray:
    """Compute the spatial mean time series.

    lat_weights=True (default): area-weighted mean with cos(lat).
        Physically correct — cells at higher latitudes cover less area.
    lat_weights=False: simple unweighted mean (for backward compatibility).
    """
    if lat_weights:
        coslat  = np.cos(np.deg2rad(da["lat"])).clip(0.0, 1.0)
        weights = coslat * xr.ones_like(da["lon"])
        gm      = da.weighted(weights).mean(("lat", "lon"))
    else:
        gm = da.mean(("lat", "lon"))

    gm.attrs = {**da.attrs, "description": "Global mean"}
    return gm


def subtract_global_mean(
    da: xr.DataArray,
    gm: xr.DataArray,
    dim: str = "year",
) -> xr.DataArray:
    """Subtract a global-mean time series from every grid point."""
    result = da - gm
    result.attrs = {
        **da.attrs,
        "description": da.attrs.get("description", "") + " (global mean removed)",
    }
    return result


# ── NaN handling ──────────────────────────────────────────────────────────────

def drop_sparse_gridpoints(
    da: xr.DataArray,
    dim: str = "time",
    max_nan_fraction: float = 0.25,
) -> xr.DataArray:
    """Set entirely NaN any time series with too many missing values."""
    nan_frac = da.isnull().mean(dim)
    return da.where(nan_frac <= max_nan_fraction)


# ── Main preprocessing wrapper ────────────────────────────────────────────────

def preprocess(
    path: Union[str, Path],
    *,
    var: Optional[str] = None,
    squeeze: Optional[Union[str, list]] = None,
    time_coord: str = "time",
    lat_coord: str = "lat",
    lon_coord: str = "lon",
    lon_convention: str = "[-180,180]",
    fill_value: float = -1000.0,
    time: Optional[tuple] = None,
    lat: Optional[tuple] = None,
    lon: Optional[tuple] = None,
    ref_period: Optional[tuple] = None,
    already_anomalies: bool = False,
    season_months: Optional[list] = None,
    compute_annual: bool = True,
    subtract_gm: bool = False,
    gm_period: Optional[tuple] = None,
    max_nan_fraction: float = 0.25,
    lat_step: int = 1,
    lon_step: int = 1,
    fix_time_start: Optional[str] = None,
    fix_time_freq: str = "MS",
) -> dict:
    """One-stop preprocessing function.

    Parameters
    ----------
    already_anomalies : bool, default False
        Set True when the input dataset is already an anomaly field
        (e.g. GISTEMP SAT, Berkeley Earth temperature).
        When True, the ``anomalies()`` step is skipped entirely — the
        data is used as-is for all downstream steps.
        When False (default), monthly anomalies are computed from the
        raw data using ``ref_period`` as the climatology baseline.

    Returns dict with keys:
        'ds'             : standardised xr.Dataset
        'da'             : raw DataArray (spatially subsetted, NOT anomalised)
        'anom'           : anomaly DataArray
                           (= 'da' if already_anomalies=True, else computed)
        'annual'         : annual means (year coordinate)
        'seasonal'       : seasonal means, if season_months given
        'gm'             : global mean time series, if subtract_gm=True
        'annual_no_gm'   : annual means with GW removed, if subtract_gm=True
        'seasonal_no_gm' : seasonal means with GW removed, if subtract_gm=True

    Dataset-specific usage
    ----------------------
    SST ERSSTv5  → preprocess('sst.nc', var='sst', ref_period=..., subtract_gm=True)
    SLP NCEP     → preprocess('slp.nc', var='slp', ref_period=...)
    GPCP Precip  → preprocess('precip.nc', var='precip', ref_period=...)
    GISTEMP SAT  → preprocess('sat.nc', var='air', already_anomalies=True)
    Berkeley E.  → use load_berkeley_earth() directly (handles decimal time)

    Note on global-mean removal
    ---------------------------
    The global mean is computed from the FULL spatial domain (before lat/lon
    subsetting) to ensure it captures the global warming signal, not a
    regional mean that would leave a residual trend in the data.
    """
    # 1. Load
    ds = load_nc(path, var=var, squeeze=squeeze)

    # 2. Fix time axis if needed
    if fix_time_start is not None:
        ds = fix_time(ds, start=fix_time_start, freq=fix_time_freq)

    # 3. Standardise coords
    ds = standardise_coords(ds,
                             time=time_coord, lat=lat_coord, lon=lon_coord, var=var)

    # 4. Fix longitudes
    ds = set_lon_convention(ds, convention=lon_convention)

    # 5. Mask fill values
    ds = mask_fill_values(ds, fill_value=fill_value)

    # 6. Reduce resolution
    if lat_step > 1 or lon_step > 1:
        ds["Xdata"] = reduce_resolution(ds["Xdata"], lat_step, lon_step)

    da_full = ds["Xdata"]

    # 7. Temporal subset only (lat/lon subset comes AFTER global-mean removal)
    da_full = subset(da_full, time=time)

    # 8. Anomalies (or pass-through if data is already anomalies)
    if already_anomalies:
        # Data is already an anomaly field — do NOT call anomalies() again.
        # This applies to: GISTEMP SAT, Berkeley Earth, and any other
        # pre-processed anomaly dataset.
        anom_full = da_full
        anom_full.attrs = {
            **da_full.attrs,
            "description": "Monthly anomalies (pre-computed in source dataset)",
        }
    else:
        # Raw data → compute anomalies from climatology
        anom_full = anomalies(da_full, ref_period=ref_period)

    anom_full = drop_sparse_gridpoints(anom_full, max_nan_fraction=max_nan_fraction)

    # 9. Global-mean removal — must happen before spatial subset
    gm = None
    if subtract_gm:
        ann_full = annual_means(anom_full)
        ann_full = drop_sparse_gridpoints(ann_full, dim="year",
                                          max_nan_fraction=max_nan_fraction)
        ann_for_gm = ann_full
        if gm_period is not None:
            ann_for_gm = ann_for_gm.sel(
                year=slice(int(gm_period[0][:4]), int(gm_period[1][:4]))
            )
        gm = global_mean(ann_for_gm)   # lat-weighted (default)

    # 10. Apply spatial subset
    da   = subset(da_full,   lat=lat, lon=lon)
    anom = subset(anom_full, lat=lat, lon=lon)

    out = {"ds": ds, "da": da, "anom": anom}
    if gm is not None:
        out["gm"] = gm

    # 11. Annual means on subsetted domain
    if compute_annual:
        ann = annual_means(anom)
        ann = drop_sparse_gridpoints(ann, dim="year",
                                     max_nan_fraction=max_nan_fraction)
        out["annual"] = ann
        if subtract_gm:
            out["annual_no_gm"] = subtract_global_mean(ann, gm, dim="year")

    # 12. Seasonal means on subsetted domain
    if season_months is not None:
        seas = seasonal_means(anom, months=season_months)
        seas = drop_sparse_gridpoints(seas, dim="year",
                                      max_nan_fraction=max_nan_fraction)
        out["seasonal"] = seas
        if subtract_gm:
            seas_full = seasonal_means(anom_full, months=season_months)
            seas_full = drop_sparse_gridpoints(seas_full, dim="year",
                                               max_nan_fraction=max_nan_fraction)
            seas_for_gm = seas_full
            if gm_period is not None:
                seas_for_gm = seas_for_gm.sel(
                    year=slice(int(gm_period[0][:4]), int(gm_period[1][:4]))
                )
            gm_s = global_mean(seas_for_gm)
            out["seasonal_no_gm"] = subtract_global_mean(seas, gm_s, dim="year")

    return out
