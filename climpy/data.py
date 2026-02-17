"""
climpy.data
===========
Climate data loading and preprocessing.

All functions are non-interactive: parameters are passed explicitly.
This makes scripts reproducible, batch-runnable, and testable.

Typical workflow
----------------
>>> import climpy.data as cd
>>> ds = cd.load_nc("data/sst.nc", var="sst", squeeze="lev")
>>> ds = cd.standardise_coords(ds, time="T", lat="latitude", lon="longitude")
>>> ds = cd.set_lon_convention(ds, convention="[-180,180]")
>>> ds = cd.subset(ds, time=("1950-01", "2020-12"), lat=(0, 80), lon=(-80, 20))
>>> anom = cd.anomalies(ds["sst"], ref_period=("1981-01", "2010-12"))
>>> ann  = cd.annual_means(anom)
>>> seas = cd.seasonal_means(anom, months=[12, 1, 2, 3])   # DJFM
>>> gw   = cd.global_mean(ann)
>>> detrended = cd.subtract_global_mean(ann, gw)
"""

from __future__ import annotations

from typing import Optional, Union
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


# ── Loading ───────────────────────────────────────────────────────────

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


# ── Coordinate normalisation ──────────────────────────────────────────

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


# ── Spatial and temporal subsetting ──────────────────────────────────

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


# ── Anomalies ─────────────────────────────────────────────────────────

def anomalies(
    da: xr.DataArray,
    ref_period: Optional[tuple] = None,
) -> xr.DataArray:
    """Compute monthly anomalies by subtracting the monthly climatology.

    anom[i,t] = SST_raw[i,t] - mean(SST_raw[i, months in ref_period matching month t])
    """
    if ref_period is not None:
        ref = da.sel(time=slice(str(ref_period[0]), str(ref_period[1])))
    else:
        ref = da

    clim = ref.groupby("time.month").mean("time")
    anom = da.groupby("time.month") - clim
    anom.attrs = {**da.attrs, "description": "Monthly anomalies"}
    return anom


# ── Temporal averaging ────────────────────────────────────────────────

def annual_means(
    da: xr.DataArray,
    time_dim: str = "time",
) -> xr.DataArray:
    """Compute annual (calendar-year) means.

    annual[i,j] = mean(anom[i, months in year j])
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

    seas          = xr.concat(slices, dim="year")
    seas["year"]  = np.array(valid_years, dtype=int)
    seas.attrs    = {**da.attrs, "description": f"Seasonal means (months={months})"}
    seas          = seas.dropna("year", how="all")
    return seas


# ── Global mean removal ───────────────────────────────────────────────

def global_mean(
    da: xr.DataArray,
    lat_weights: bool = False,
) -> xr.DataArray:
    """Compute the spatial mean time series.

    lat_weights=False (default): simple unweighted mean — matches original code.
    lat_weights=True: area-weighted mean with cos(lat).

    global_mean[j] = mean over all spatial points i of da[i,j]
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
    """Subtract a global-mean time series from every grid point.

    annual_no_gm[i,j] = annual[i,j] - global_mean[j]
    """
    result = da - gm
    result.attrs = {
        **da.attrs,
        "description": da.attrs.get("description", "") + " (global mean removed)",
    }
    return result


# ── NaN handling ──────────────────────────────────────────────────────

def drop_sparse_gridpoints(
    da: xr.DataArray,
    dim: str = "time",
    max_nan_fraction: float = 0.25,
) -> xr.DataArray:
    """Set entirely NaN any time series with too many missing values."""
    nan_frac = da.isnull().mean(dim)
    return da.where(nan_frac <= max_nan_fraction)


# ── Convenience wrapper ───────────────────────────────────────────────

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

    Returns dict with keys:
        'ds'             : standardised xr.Dataset
        'da'             : raw DataArray (already spatially subsetted)
        'anom'           : monthly anomalies
        'annual'         : annual means (year coordinate)
        'seasonal'       : seasonal means, if season_months given
        'gm'             : global mean time series, if subtract_gm=True
        'annual_no_gm'   : annual means with GW removed, if subtract_gm=True
        'seasonal_no_gm' : seasonal means with GW removed, if subtract_gm=True

    Note on global-mean removal
    ---------------------------
    The global mean is computed from the FULL spatial domain (before lat/lon
    subsetting), exactly as in the original code.  This ensures that the
    removed signal is a true global-ocean warming trend, not a regional mean.
    Applying the spatial subset first would leave a residual regional trend
    in the data and distort the leading EOFs.
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

    # 8. Anomalies on full spatial domain
    anom_full = anomalies(da_full, ref_period=ref_period)
    anom_full = drop_sparse_gridpoints(anom_full, max_nan_fraction=max_nan_fraction)

    # 9. Global-mean removal — must happen before spatial subset
    #    so that gm is a true global (or full-domain) ocean mean, not a regional one.
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
        gm = global_mean(ann_for_gm)   # unweighted — matches original code

    # 10. Now apply spatial subset
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
            # Subtract the global mean (computed from full domain above)
            # from the regional annual means — same as original:
            #   Xmammg = Xma_zone - Xmg_global
            out["annual_no_gm"] = subtract_global_mean(ann, gm, dim="year")

    # 12. Seasonal means on subsetted domain
    if season_months is not None:
        seas = seasonal_means(anom, months=season_months)
        seas = drop_sparse_gridpoints(seas, dim="year",
                                      max_nan_fraction=max_nan_fraction)
        out["seasonal"] = seas
        if subtract_gm:
            # For seasonal: compute seasonal gm from full domain, then subtract
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
