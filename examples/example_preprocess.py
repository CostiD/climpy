"""
example_preprocess.py
=====================
Demonstrates climpy.preprocess: load, clean, compute anomalies,
annual means, seasonal means and global-warming removal.

All parameters are set at the top of the script — no interactive input().
"""

from pathlib import Path
import climpy

# ── 1. Configuration (edit these) ────────────────────────────────────

DATA_FILE    = Path("data/sat/giss_temp_both_1200.nc")
OUT_DIR      = Path("preprocessed_data/sat")
DATASET_NAME = "gisstemp"   # used in output filename

# Variable / coordinate names in the raw file (only fill if non-standard)
VAR_NAME    = None       # None → auto-detect
TIME_COORD  = "time"
LAT_COORD   = "lat"
LON_COORD   = "lon"

# Fix time axis from scratch (set to None if the file already has proper dates)
FIX_TIME_START = "1880-01-01"   # or None
FIX_TIME_FREQ  = "MS"

# Region of interest
TIME_RANGE = ("1880-01-01", "2019-12-31")
LAT_RANGE  = (-90, 90)
LON_RANGE  = (-180, 180)

# Climatology reference period for anomalies
REF_PERIOD = ("1951-01-01", "1980-12-31")

# Season (change months to e.g. [12,1,2,3] for DJFM, [6,7,8,9] for JJAS)
SEASON_MONTHS = [6, 7, 8, 9]    # JJAS
SEASON_NAME   = "JJAS"

# Longitude convention
LON_CONVENTION = "[-180,180]"

# Remove global warming signal from annual/seasonal means?
SUBTRACT_GW = True


# ── 2. Run preprocessing ──────────────────────────────────────────────

print(f"Processing: {DATA_FILE}")

result = climpy.preprocess(
    DATA_FILE,
    var             = VAR_NAME,
    time_coord      = TIME_COORD,
    lat_coord       = LAT_COORD,
    lon_coord       = LON_COORD,
    lon_convention  = LON_CONVENTION,
    time            = TIME_RANGE,
    lat             = LAT_RANGE,
    lon             = LON_RANGE,
    ref_period      = REF_PERIOD,
    season_months   = SEASON_MONTHS,
    subtract_gm     = SUBTRACT_GW,
    fix_time_start  = FIX_TIME_START,
    fix_time_freq   = FIX_TIME_FREQ,
)

anom   = result["anom"]             # monthly anomalies
annual = result["annual"]           # annual means
seas   = result["seasonal"]         # seasonal (JJAS) means

print(f"Monthly anomalies:  {anom.dims}  {dict(anom.sizes)}")
print(f"Annual means:       {annual.dims}  {dict(annual.sizes)}")
print(f"Seasonal means:     {seas.dims}  {dict(seas.sizes)}")

if SUBTRACT_GW:
    annual_no_gw = result["annual_no_gm"]
    seas_no_gw   = result["seasonal_no_gm"]
    print(f"Annual (-GW):       {annual_no_gw.dims}")
    print(f"Seasonal (-GW):     {seas_no_gw.dims}")


# ── 3. Save to NetCDF ─────────────────────────────────────────────────

OUT_DIR.mkdir(parents=True, exist_ok=True)

yr_start = int(anom["time"].dt.year.values[0])
yr_end   = int(anom["time"].dt.year.values[-1])

lat0, lat1 = int(LAT_RANGE[0]), int(LAT_RANGE[1])
lon0, lon1 = int(LON_RANGE[0]), int(LON_RANGE[1])

base = f"{DATASET_NAME}_{yr_start}-{yr_end}_{lon0}E-{lon1}E_{lat0}N-{lat1}N"

anom.to_netcdf(OUT_DIR / f"{base}_anom.nc")
annual.to_netcdf(OUT_DIR / f"{base}_annual.nc")
seas.to_netcdf(OUT_DIR / f"{base}_{SEASON_NAME}.nc")

if SUBTRACT_GW:
    annual_no_gw.to_netcdf(OUT_DIR / f"{base}_annual_minus_GW.nc")
    seas_no_gw.to_netcdf(  OUT_DIR / f"{base}_{SEASON_NAME}_minus_GW.nc")
    result["gm"].to_netcdf(OUT_DIR / f"{DATASET_NAME}_{yr_start}-{yr_end}_global_mean.nc")

print("Done. Files saved to:", OUT_DIR)
