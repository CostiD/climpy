"""
example_corr_regression.py
===========================
Correlation and linear regression maps:
  Tripole/AMO index → precipitation (DJFM and JJAS).

The common data-loading block that was duplicated between the old
CORR.py and LINREG.py is now written once.

Figures produced:
  fig_corr_djfm_jjas.pdf    — 2×2 Pearson r maps
  fig_linreg_djfm_jjas.pdf  — 2×2 regression slope maps
"""

from pathlib import Path
import matplotlib.pyplot as plt
import climpy

# ── Configuration ─────────────────────────────────────────────────────

PC_TRIPOL_FILE  = Path("results/eofs/pc_tripol_ersstv5.nc")
PC_AMO_FILE     = Path("results/eofs/pc_amo_ersstv5.nc")
PRECIP_DJFM     = Path("preprocessed_data/precip/prate_DJFM.nc")
PRECIP_JJAS     = Path("preprocessed_data/precip/prate_JJAS.nc")
FIG_CORR_OUT    = Path("figures/fig_corr_tripol_amo.pdf")
FIG_LINREG_OUT  = Path("figures/fig_linreg_tripol_amo.pdf")

TIME_RANGE  = ("1880-01-01", "2015-01-01")
MA_WINDOW   = 7         # years for moving average
PTHRESH     = 0.05      # significance level for stippling
CMAP        = plt.cm.BrBG


# ── Load data ─────────────────────────────────────────────────────────

def load_and_rename(path, old_var, new_var):
    ds = climpy.load_nc(path)
    ds = climpy.standardise_coords(ds, time="year" if "year" in ds.dims else "time")
    if old_var in ds:
        ds = ds.rename({old_var: new_var})
    return ds[new_var].sel(time=slice(*TIME_RANGE)) if "time" in ds.dims \
        else ds[new_var]

tripol = load_and_rename(PC_TRIPOL_FILE, "pcs", "tripol")
amo    = -load_and_rename(PC_AMO_FILE,   "pcs", "amo")   # sign convention

Y_djfm = climpy.load_nc(PRECIP_DJFM)
Y_djfm = climpy.standardise_coords(Y_djfm)["Xdata"] * 86400   # → mm/day
Y_djfm = climpy.subset(Y_djfm, time=TIME_RANGE)

Y_jjas = climpy.load_nc(PRECIP_JJAS)
Y_jjas = climpy.standardise_coords(Y_jjas)["Xdata"] * 86400
Y_jjas = climpy.subset(Y_jjas, time=TIME_RANGE)


# ── 7-year moving average ─────────────────────────────────────────────

tripol_ma = climpy.moving_average(tripol, n=MA_WINDOW)
amo_ma    = climpy.moving_average(amo,    n=MA_WINDOW)


# ── Correlations ──────────────────────────────────────────────────────

# Use fast xr.corr; for p-values use correlation_pvalue
corr_tripol_djfm = climpy.pearson_correlation(tripol_ma, Y_djfm, dim="time")
corr_amo_djfm    = climpy.pearson_correlation(amo_ma,    Y_djfm, dim="time")
corr_tripol_jjas = climpy.pearson_correlation(tripol_ma, Y_jjas, dim="time")
corr_amo_jjas    = climpy.pearson_correlation(amo_ma,    Y_jjas, dim="time")

# p-values for stippling (slower — comment out if not needed)
_, pval_tripol_djfm = climpy.correlation_pvalue(tripol_ma, Y_djfm, dim="time")
_, pval_amo_djfm    = climpy.correlation_pvalue(amo_ma,    Y_djfm, dim="time")
_, pval_tripol_jjas = climpy.correlation_pvalue(tripol_ma, Y_jjas, dim="time")
_, pval_amo_jjas    = climpy.correlation_pvalue(amo_ma,    Y_jjas, dim="time")


# ── Regression slopes ─────────────────────────────────────────────────

slope_tripol_djfm = climpy.linear_regression(tripol_ma, Y_djfm, dim="time")
slope_amo_djfm    = climpy.linear_regression(amo_ma,    Y_djfm, dim="time")
slope_tripol_jjas = climpy.linear_regression(tripol_ma, Y_jjas, dim="time")
slope_amo_jjas    = climpy.linear_regression(amo_ma,    Y_jjas, dim="time")


# ── Helper: common map keyword arguments ─────────────────────────────

LON_TICKS = [-180, -120, -60, 0, 60, 120, 180]
LAT_TICKS = [-90, -60, -30, 0, 30, 60, 90]
MAP_KWARGS = dict(
    lon_ticks=LON_TICKS, lat_ticks=LAT_TICKS,
    filltype="contourf", cmap=CMAP,
    cbar_orientation="horizontal",
)


# ── Figure 1: Correlation maps ────────────────────────────────────────

climpy.use_style("nature")

fig_c = climpy.ClimPlot(
    nrows=2, ncols=2, w=climpy.NATURE_2COL, h=6,
    map_proj=(climpy.Map(-65), climpy.Map(-65),
              climpy.Map(-65), climpy.Map(-65)),
)

fig_c[0].map(corr_tripol_djfm,
             title="Tripole × Precip. DJFM",
             vmin=-0.5, vmax=0.5,
             cbar_label="r",
             **MAP_KWARGS)
fig_c.add_stippling(fig_c.axes[0],
                    pval_tripol_djfm.values,
                    corr_tripol_djfm["lat"].values,
                    corr_tripol_djfm["lon"].values,
                    pthresh=PTHRESH)

fig_c[1].map(corr_amo_djfm,
             title="AMO × Precip. DJFM",
             vmin=-0.5, vmax=0.5,
             cbar_label="r",
             **MAP_KWARGS)
fig_c.add_stippling(fig_c.axes[1],
                    pval_amo_djfm.values,
                    corr_amo_djfm["lat"].values,
                    corr_amo_djfm["lon"].values,
                    pthresh=PTHRESH)

fig_c[2].map(corr_tripol_jjas,
             title="Tripole × Precip. JJAS",
             vmin=-0.5, vmax=0.5,
             cbar_label="r",
             **MAP_KWARGS)
fig_c.add_stippling(fig_c.axes[2],
                    pval_tripol_jjas.values,
                    corr_tripol_jjas["lat"].values,
                    corr_tripol_jjas["lon"].values,
                    pthresh=PTHRESH)

fig_c[3].map(corr_amo_jjas,
             title="AMO × Precip. JJAS",
             vmin=-0.5, vmax=0.5,
             cbar_label="r",
             **MAP_KWARGS)
fig_c.add_stippling(fig_c.axes[3],
                    pval_amo_jjas.values,
                    corr_amo_jjas["lat"].values,
                    corr_amo_jjas["lon"].values,
                    pthresh=PTHRESH)

fig_c.label_subplots()
fig_c.savefig(FIG_CORR_OUT)


# ── Figure 2: Regression maps ─────────────────────────────────────────

fig_r = climpy.ClimPlot(
    nrows=2, ncols=2, w=climpy.NATURE_2COL, h=6,
    map_proj=(climpy.Map(-65), climpy.Map(-65),
              climpy.Map(-65), climpy.Map(-65)),
)

fig_r[0].map(slope_tripol_djfm,
             title="Tripole → Precip. DJFM",
             vmin=-0.35, vmax=0.35,
             cbar_label="mm day⁻¹ σ⁻¹",
             **MAP_KWARGS)

fig_r[1].map(slope_amo_djfm,
             title="AMO → Precip. DJFM",
             vmin=-0.35, vmax=0.35,
             cbar_label="mm day⁻¹ σ⁻¹",
             **MAP_KWARGS)

fig_r[2].map(slope_tripol_jjas,
             title="Tripole → Precip. JJAS",
             vmin=-0.35, vmax=0.35,
             cbar_label="mm day⁻¹ σ⁻¹",
             **MAP_KWARGS)

fig_r[3].map(slope_amo_jjas,
             title="AMO → Precip. JJAS",
             vmin=-0.35, vmax=0.35,
             cbar_label="mm day⁻¹ σ⁻¹",
             **MAP_KWARGS)

fig_r.label_subplots()
fig_r.savefig(FIG_LINREG_OUT)

plt.show()
