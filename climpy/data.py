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
    """Load a NetCDF file into an xarray.Dataset.

    Parameters
    ----------
    path : str or Path
        Path to the .nc file.
    var : str, optional
        If given, select only this variable from the dataset.
    squeeze : str or list of str, optional
        Coordinate name(s) of size-1 dimensions to remove (e.g. 'lev').
    decode_times : bool
        Passed to xarray.open_dataset (default True).

    Returns
    -------
    xr.Dataset
    """
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
    """Replace a numeric/cftime time coordinate with a proper DatetimeIndex.

    Useful when files have decode_times=False or non-standard calendars.

    Parameters
    ----------
    start : str
        First date, e.g. '1870-01-01'.
    freq  : str
        Pandas frequency string. 'MS' = month start, 'AS' = year start.

    Returns
    -------
    xr.Dataset with corrected 'time' coordinate.
    """
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
    """Rename coordinates and variables to the standard names used by climpy.

    Standard names:  time | lat | lon  for coordinates
                     Xdata             for the main climate variable

    Parameters
    ----------
    time, lat, lon, var : str, optional
        Original names in the dataset. Only pass names that differ from
        the standard names.

    Examples
    --------
    >>> ds = standardise_coords(ds, time='T', lat='latitude', var='sst')
    """
    rename = {}
    if time and time != "time":
        rename[time] = "time"
    if lat and lat != "lat":
        rename[lat] = "lat"
    if lon and lon != "lon":
        rename[lon] = "lon"
    if var and var != "Xdata":
        rename[var] = "Xdata"

    # Handle unnamed DataArray variable
    for name in list(ds.data_vars):
        if name == "__xarray_dataarray_variable__":
            rename[name] = "Xdata"
            break

    if rename:
        ds = ds.rename(rename)

    # Sort lat and lon ascendingly (needed for slicing to work)
    if "lat" in ds.coords:
        ds = ds.isel(lat=np.argsort(ds["lat"].values))
    if "lon" in ds.coords:
        ds = ds.isel(lon=np.argsort(ds["lon"].values))

    return ds


def set_lon_convention(
    ds: xr.Dataset,
    convention: str = "[-180,180]",
) -> xr.Dataset:
    """Convert longitude convention.

    Parameters
    ----------
    convention : '[-180,180]' or '[0,360]'

    Returns
    -------
    xr.Dataset with re-ordered longitudes.
    """
    if "lon" not in ds.coords:
        raise KeyError("Dataset has no 'lon' coordinate. Run standardise_coords first.")

    lon = ds["lon"].values

    if convention == "[-180,180]":
        if lon.max() <= 180:
            return ds  # already correct
        # Convert 180–360 → -180–0
        ds = ds.assign_coords(lon=(((lon + 180) % 360) - 180))
        ds = ds.sortby("lon")

    elif convention == "[0,360]":
        if lon.min() >= 0:
            return ds  # already correct
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
    """Replace unrealistic fill values with NaN.

    Parameters
    ----------
    fill_value : float
        Exact value to replace (default -1000.0, common in old NCEI files).
    abs_threshold : float
        Values whose absolute value exceeds this are also masked (default 1e29).
    """
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
    """Select a spatio-temporal subregion.

    Parameters
    ----------
    time : (start, end), optional
        E.g. ('1950-01', '2019-12').
    lat  : (south, north), optional
        E.g. (0, 80).
    lon  : (west, east), optional
        E.g. (-75, 20).

    Returns
    -------
    Subsetted Dataset or DataArray.
    """
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
    """Sub-sample spatial resolution by taking every nth grid point.

    Parameters
    ----------
    lat_step : int
        Keep every lat_step-th latitude point (default 1 = no change).
    lon_step : int
        Keep every lon_step-th longitude point.
    """
    return da.isel(lat=slice(None, None, lat_step),
                   lon=slice(None, None, lon_step))


# ── Anomalies ─────────────────────────────────────────────────────────

def anomalies(
    da: xr.DataArray,
    ref_period: Optional[tuple] = None,
) -> xr.DataArray:
    """Compute monthly anomalies by subtracting the monthly climatology.

    Parameters
    ----------
    da : xr.DataArray
        Monthly data with a 'time' dimension.
    ref_period : (start, end), optional
        Reference period for the climatology, e.g. ('1981-01', '2010-12').
        If None, the full record is used.

    Returns
    -------
    xr.DataArray of anomalies.
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

    The output has a 'year' coordinate expressed as integer years.
    Only years with all 12 months present receive a valid value
    (partial years at the record edges are NaN).

    Parameters
    ----------
    da : xr.DataArray with monthly 'time' coordinate.

    Returns
    -------
    xr.DataArray indexed by integer year.
    """
    ann = da.resample({time_dim: "AS"}).mean(time_dim)
    # Mark partial years at edges as NaN
    counts = da.resample({time_dim: "AS"}).count(time_dim)
    ann = ann.where(counts >= 12)
    ann = ann.dropna(time_dim, how="all")

    # Convert DatetimeIndex to integer years for convenience
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
    import pandas as pd

    months_set = set(months)
    n_months   = len(months)

    # Filtrare cu pandas — evită problemele cu xarray.dt.isin
    time_pd   = pd.DatetimeIndex(da[time_dim].values)
    month_mask = np.isin(time_pd.month, list(months_set))
    da_sel    = da.isel({time_dim: month_mask})

    if da_sel.sizes[time_dim] == 0:
        raise ValueError(f"Nu există date pentru lunile {months}.")

    # Etichete de an sezonier
    time_sel = pd.DatetimeIndex(da_sel[time_dim].values)
    is_cross_year = 12 in months_set and any(m < 6 for m in months_set)

    if is_cross_year:
        season_years = time_sel.year + (time_sel.month == 12).astype(int)
    else:
        season_years = time_sel.year

    # Grupare manuală — evită orice groupby din xarray
    unique_years = np.unique(season_years)
    slices = []
    valid_years = []

    for yr in unique_years:
        mask  = season_years == yr
        count = int(mask.sum())
        chunk = da_sel.isel({time_dim: mask}).mean(time_dim)
        if count < n_months:
            chunk = chunk * np.nan   # sezon incomplet → NaN
        slices.append(chunk)
        valid_years.append(yr)

    seas = xr.concat(slices, dim="year")
    seas["year"] = np.array(valid_years, dtype=int)
    seas.attrs   = {**da.attrs,
                    "description": f"Seasonal means (months={months})"}
    return seas


# ── Global mean removal ───────────────────────────────────────────────

def global_mean(
    da: xr.DataArray,
    lat_weights: bool = True,
) -> xr.DataArray:
    """Compute the area-weighted global (or domain) mean time series.

    Parameters
    ----------
    da : xr.DataArray
        Field with 'lat' and 'lon' dimensions.
    lat_weights : bool
        If True (default), weight each grid cell by cos(lat).

    Returns
    -------
    xr.DataArray with only the time/year dimension.
    """
    if lat_weights:
        coslat = np.cos(np.deg2rad(da["lat"])).clip(0.0, 1.0)
        weights = coslat * xr.ones_like(da["lon"])
        gm = da.weighted(weights).mean(("lat", "lon"))
    else:
        gm = da.mean(("lat", "lon"))

    gm.attrs = {**da.attrs, "description": "Area-weighted global mean"}
    return gm


def subtract_global_mean(
    da: xr.DataArray,
    gm: xr.DataArray,
    dim: str = "year",
) -> xr.DataArray:
    """Subtract a global-mean time series from every grid point.

    This removes the uniform warming/cooling signal
    (global-warming removal step).

    Parameters
    ----------
    da  : xr.DataArray, field (year/time × lat × lon).
    gm  : xr.DataArray, 1D global mean (year/time).
    dim : str, the time dimension name (default 'year').
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
    """Set entirely NaN any time series with too many missing values.

    The NaN-filled series are kept (they are not dropped) so that the
    spatial shape of the field is preserved. Use this before EOF analysis.

    Parameters
    ----------
    dim : str
        The time dimension (e.g. 'time' or 'year').
    max_nan_fraction : float
        Grid points with a fraction of NaNs above this threshold are masked.
        Default 0.25 (25 %).

    Returns
    -------
    xr.DataArray with the same shape as input.
    """
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
    season_months: Optional[list[int]] = None,
    compute_annual: bool = True,
    subtract_gm: bool = False,
    max_nan_fraction: float = 0.25,
    lat_step: int = 1,
    lon_step: int = 1,
    fix_time_start: Optional[str] = None,
    fix_time_freq: str = "MS",
) -> dict:
    """One-stop preprocessing function.

    Loads a NetCDF file and returns a dictionary with pre-computed
    standard products (anomalies, annual means, seasonal means, etc.).

    Parameters
    ----------
    path : path to .nc file
    var  : variable name (optional)
    squeeze : size-1 dimension to drop (e.g. 'lev')
    time_coord, lat_coord, lon_coord : original coordinate names if non-standard
    lon_convention : '[-180,180]' or '[0,360]'
    fill_value : value to mask as NaN (default -1000)
    time, lat, lon : (start, end) tuples for subsetting
    ref_period : (start, end) for anomaly climatology
    season_months : e.g. [12,1,2,3] for DJFM
    compute_annual : if True, compute annual means
    subtract_gm : if True, also return GW-removed annual means
    max_nan_fraction : threshold for masking sparse series
    lat_step, lon_step : spatial downsampling
    fix_time_start : if not None, rebuild time axis starting here
    fix_time_freq : frequency for fix_time

    Returns
    -------
    dict with keys:
        'ds'      : standardised xr.Dataset
        'da'      : raw DataArray (Xdata)
        'anom'    : monthly anomalies
        'annual'  : annual means (year coordinate)
        'seasonal': seasonal means (year coordinate), if season_months given
        'gm'      : global mean time series, if subtract_gm=True
        'annual_no_gm'  : annual means with GW removed, if subtract_gm=True
        'seasonal_no_gm': seasonal means with GW removed, if subtract_gm=True

    Example
    -------
    >>> from climpy.data import preprocess
    >>> result = preprocess(
    ...     "data/ersstv5.nc",
    ...     var="sst",
    ...     lon_convention="[-180,180]",
    ...     time=("1950-01", "2019-12"),
    ...     lat=(0, 80), lon=(-80, 20),
    ...     ref_period=("1981-01", "2010-12"),
    ...     season_months=[12, 1, 2, 3],
    ...     subtract_gm=True,
    ... )
    >>> anom  = result['anom']
    >>> djfm  = result['seasonal']
    """
    # 1. Load
    ds = load_nc(path, var=var, squeeze=squeeze)

    # 2. Fix time axis if needed
    if fix_time_start is not None:
        ds = fix_time(ds, start=fix_time_start, freq=fix_time_freq)

    # 3. Standardise coords
    ds = standardise_coords(ds,
                             time=time_coord, lat=lat_coord, lon=lon_coord)

    # 4. Fix longitudes
    ds = set_lon_convention(ds, convention=lon_convention)

    # 5. Mask fill values
    ds = mask_fill_values(ds, fill_value=fill_value)

    # 6. Reduce resolution
    if lat_step > 1 or lon_step > 1:
        ds["Xdata"] = reduce_resolution(ds["Xdata"], lat_step, lon_step)

    da = ds["Xdata"]

    # 7. Subset
    da = subset(da, time=time, lat=lat, lon=lon)

    # 8. Anomalies
    anom = anomalies(da, ref_period=ref_period)
    anom = drop_sparse_gridpoints(anom, max_nan_fraction=max_nan_fraction)

    out = {"ds": ds, "da": da, "anom": anom}

    # 9. Annual means
    if compute_annual:
        ann = annual_means(anom)
        ann = drop_sparse_gridpoints(ann, dim="year",
                                     max_nan_fraction=max_nan_fraction)
        out["annual"] = ann

    # 10. Seasonal means
    if season_months is not None:
        seas = seasonal_means(anom, months=season_months)
        seas = drop_sparse_gridpoints(seas, dim="year",
                                      max_nan_fraction=max_nan_fraction)
        out["seasonal"] = seas

    # 11. Global-mean removal
    if subtract_gm and "annual" in out:
        gm = global_mean(out["annual"])
        out["gm"] = gm
        out["annual_no_gm"] = subtract_global_mean(out["annual"], gm, dim="year")

        if "seasonal" in out:
            gm_s = global_mean(out["seasonal"])
            out["seasonal_no_gm"] = subtract_global_mean(
                out["seasonal"], gm_s, dim="year"
            )

    return out
