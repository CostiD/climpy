"""
generate_test_data.py
=====================
Generează date climatice sintetice dar realiste, ca fișiere NetCDF mici (~1–3 MB).
Nu ai nevoie să descarci nimic.  Rulează o singură dată, înainte de a folosi climpy.

Date produse în folderul  data/test/ :
  sst_test.nc      — anomalii SST, Atlanticul de Nord, 5° rezoluție, 1950–2019
  slp_test.nc      — anomalii SLP, global, 5° rezoluție, 1950–2019
  precip_test.nc   — precipitații, global, 5° rezoluție, 1950–2019

Semnalele incluse:
  SST  → pattern tip AMO (monopol) + Tripol + zgomot alb
  SLP  → pattern tip NAO + oscilație multidecadală + zgomot
  Prec → răspuns la AMO + zgomot

Utilizare
---------
    python generate_test_data.py
Sau din notebook:
    %run generate_test_data.py
"""

from pathlib import Path
import numpy as np
import xarray as xr
import pandas as pd

# ── Configurare ───────────────────────────────────────────────────────

OUT_DIR    = Path("data/test")
SEED       = 42
N_YEARS    = 70          # 1950–2019
START_YEAR = 1950

OUT_DIR.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(SEED)

# ── Grile ─────────────────────────────────────────────────────────────

# SST: Atlanticul de Nord (5° rezoluție)
lat_sst = np.arange(0, 82, 5,   dtype=float)   # 0°N – 80°N
lon_sst = np.arange(-80, 25, 5, dtype=float)   # 80°W – 20°E
nlat_s, nlon_s = len(lat_sst), len(lon_sst)

# SLP + Precip: global (5° rezoluție)
lat_gl  = np.arange(-90, 92, 5, dtype=float)
lon_gl  = np.arange(-180, 182, 5, dtype=float)
nlat_g, nlon_g = len(lat_gl), len(lon_gl)

# Timp: monthly, 1950–2019
time_monthly = pd.date_range(f"{START_YEAR}-01-01",
                              periods=N_YEARS * 12, freq="MS")
T = len(time_monthly)


# ── Pattern spatial helper ────────────────────────────────────────────

def gaussian_patch(lat2d, lon2d, clat, clon, sigma_lat=15, sigma_lon=25,
                   amplitude=1.0):
    """Patch gaussian pe grila (lat2d, lon2d)."""
    d = ((lat2d - clat) / sigma_lat) ** 2 + ((lon2d - clon) / sigma_lon) ** 2
    return amplitude * np.exp(-d)


# ── 1. SST ─────────────────────────────────────────────────────────────

print("Generez SST...", end=" ")

lon2d_s, lat2d_s = np.meshgrid(lon_sst, lat_sst)

# Pattern AMO: monopol pozitiv în tot bazinul N Atlantic
amo_pattern = (
    gaussian_patch(lat2d_s, lon2d_s, clat=40, clon=-30, amplitude=0.8)
    + gaussian_patch(lat2d_s, lon2d_s, clat=20, clon=-50, amplitude=0.5)
    + gaussian_patch(lat2d_s, lon2d_s, clat=60, clon=-20, amplitude=0.4)
)

# Pattern Tripol: pol cald în nord, rece în mijloc, cald în sud
tripol_pattern = (
    gaussian_patch(lat2d_s, lon2d_s, clat=55, clon=-35, amplitude= 0.7)
  + gaussian_patch(lat2d_s, lon2d_s, clat=35, clon=-45, amplitude=-0.6)
  + gaussian_patch(lat2d_s, lon2d_s, clat=15, clon=-40, amplitude= 0.5)
)

# Seria de timp: oscilație multidecadală lentă (~60 ani) + variabilitate interanuală
years = np.arange(N_YEARS)
monthly_idx = np.arange(T)
year_frac = monthly_idx / 12.0

amo_ts = (
    0.4 * np.sin(2 * np.pi * year_frac / 60.0)   # ciclu 60 ani
    + 0.15 * np.sin(2 * np.pi * year_frac / 20.0)
    + 0.1 * rng.standard_normal(T)
)
tripol_ts = (
    0.3 * np.sin(2 * np.pi * year_frac / 8.0 + 1.2)
    + 0.2 * np.sin(2 * np.pi * year_frac / 3.5 + 0.5)
    + 0.12 * rng.standard_normal(T)
)

# Câmpul SST: combinație liniară + zgomot spațial-temporal
sst = np.zeros((T, nlat_s, nlon_s))
for t in range(T):
    sst[t] = (amo_ts[t]    * amo_pattern
            + tripol_ts[t] * tripol_pattern
            + 0.25 * rng.standard_normal((nlat_s, nlon_s)))

# Maschează pământul (aprox.) — NaN pentru latitudini mari la unele longitudini
land_mask = np.zeros((nlat_s, nlon_s), dtype=bool)
# DUPĂ — corect:
land_mask[(lat2d_s > 70) & (lon2d_s > -10)] = True   # Scandinavia / N. Europa
land_mask[(lat2d_s < 10) & (lon2d_s > 10)]  = True   # Africa de Vest
sst[:, land_mask] = np.nan

sst_da = xr.DataArray(
    sst.astype("float32"),
    coords={"time": time_monthly, "lat": lat_sst, "lon": lon_sst},
    dims=["time", "lat", "lon"],
    name="Xdata",
    attrs={"long_name": "SST anomaly (synthetic)", "units": "degC",
           "description": "Synthetic North Atlantic SST anomalies for testing climpy"},
)
sst_da.to_netcdf(OUT_DIR / "sst_test.nc")
print(f"✓  {OUT_DIR/'sst_test.nc'}  "
      f"({sst_da.nbytes/1e6:.1f} MB uncompressed, ~{sst_da.nbytes/4e6:.1f} MB on disk)")


# ── 2. SLP ─────────────────────────────────────────────────────────────

print("Generez SLP...", end=" ")

lon2d_g, lat2d_g = np.meshgrid(lon_gl, lat_gl)

# Pattern NAO: dipol Azore (H) + Islanda (L)
nao_pattern = (
    gaussian_patch(lat2d_g, lon2d_g, clat=38, clon=-28, sigma_lat=12,
                   sigma_lon=25,  amplitude= 4.0)    # Azore High
  + gaussian_patch(lat2d_g, lon2d_g, clat=65, clon=-20, sigma_lat=12,
                   sigma_lon=25,  amplitude=-5.0)    # Icelandic Low
)

# Pattern ENSO teleconnection
enso_slp = gaussian_patch(lat2d_g, lon2d_g, clat=0, clon=-130,
                           sigma_lat=20, sigma_lon=40, amplitude=-3.0)

nao_ts = (
    0.5 * rng.standard_normal(T)
    + 0.2 * np.sin(2 * np.pi * year_frac / 2.3)
)
enso_ts = (
    0.3 * np.sin(2 * np.pi * year_frac / 3.5 + 0.8)
    + 0.15 * rng.standard_normal(T)
)

slp = np.zeros((T, nlat_g, nlon_g))
for t in range(T):
    slp[t] = (nao_ts[t]  * nao_pattern
            + enso_ts[t] * enso_slp
            + 1.5 * rng.standard_normal((nlat_g, nlon_g)))

slp_da = xr.DataArray(
    slp.astype("float32"),
    coords={"time": time_monthly, "lat": lat_gl, "lon": lon_gl},
    dims=["time", "lat", "lon"],
    name="Xdata",
    attrs={"long_name": "SLP anomaly (synthetic)", "units": "hPa",
           "description": "Synthetic global SLP anomalies for testing climpy"},
)
slp_da.to_netcdf(OUT_DIR / "slp_test.nc")
print(f"✓  {OUT_DIR/'slp_test.nc'}  "
      f"({slp_da.nbytes/1e6:.1f} MB uncompressed)")


# ── 3. Precipitații ────────────────────────────────────────────────────

print("Generez precipitații...", end=" ")

# Climatologie de bază (pattern zonal — mai multă ploaie la tropice)
clim_precip = (
    3.0 * np.exp(-((lat2d_g - 5) / 15) ** 2)     # ITCZ
  + 2.0 * np.exp(-((lat2d_g - 50) / 12) ** 2)    # midlatitude storm track
  + 0.5 * np.exp(-((lat2d_g + 60) / 10) ** 2)    # SH storm track
)

# Răspuns la AMO: mai multă ploaie în Sahel când AMO e pozitiv
sahel_pattern = gaussian_patch(lat2d_g, lon2d_g, clat=15, clon=5,
                                sigma_lat=5, sigma_lon=20, amplitude=0.5)
# Răspuns la NAO: mai puțină ploaie în Mediterana când NAO e pozitiv
med_pattern   = gaussian_patch(lat2d_g, lon2d_g, clat=38, clon=15,
                                sigma_lat=8, sigma_lon=25, amplitude=-0.3)

# Proiectăm AMO pe grilă globală (simplificat)
amo_ts_monthly = amo_ts

precip = np.zeros((T, nlat_g, nlon_g))
for t in range(T):
    precip[t] = (clim_precip
                + amo_ts_monthly[t]  * (sahel_pattern + med_pattern)
                + nao_ts[t] * med_pattern
                + 0.8 * rng.standard_normal((nlat_g, nlon_g)))
    precip[t] = np.clip(precip[t], 0, None)   # precipitațiile nu pot fi negative

# Conversie la kg/m2/s (ca în fișierele reale GPCC)
precip_si = precip / 86400.0   # mm/day → kg m-2 s-1

precip_da = xr.DataArray(
    precip_si.astype("float32"),
    coords={"time": time_monthly, "lat": lat_gl, "lon": lon_gl},
    dims=["time", "lat", "lon"],
    name="Xdata",
    attrs={"long_name": "Precipitation (synthetic)", "units": "kg m-2 s-1",
           "description": "Synthetic global precipitation for testing climpy"},
)
precip_da.to_netcdf(OUT_DIR / "precip_test.nc")
print(f"✓  {OUT_DIR/'precip_test.nc'}  "
      f"({precip_da.nbytes/1e6:.1f} MB uncompressed)")


# ── Rezumat ────────────────────────────────────────────────────────────

import os
total = sum(os.path.getsize(OUT_DIR / f)
            for f in ["sst_test.nc", "slp_test.nc", "precip_test.nc"])
print(f"\nGata! 3 fișiere NetCDF → {OUT_DIR}/")
print(f"Dimensiune totală pe disc: {total/1e6:.1f} MB")
print()
print("Fișiere generate:")
print(f"  sst_test.nc    — SST anomalii, 5°, N Atlantic, "
      f"{lat_sst[0]:.0f}°N–{lat_sst[-1]:.0f}°N, {lon_sst[0]:.0f}°E–{lon_sst[-1]:.0f}°E")
print(f"  slp_test.nc    — SLP anomalii, 5°, global")
print(f"  precip_test.nc — Precipitații, 5°, global  [kg m-2 s-1]")
print()
print("Rulează acum demo_notebook.ipynb sau example_eof_test.py")
