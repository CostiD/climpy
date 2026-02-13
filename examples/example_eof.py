"""
example_eof.py
==============
EOF analysis of North Atlantic SST anomalies.
Produces a 2×2 publication-quality figure:
  (a) EOF-AMO spatial pattern
  (b) EOF-Tripole spatial pattern
  (c) PC-AMO time series + 7-yr MA
  (d) PC-Tripole time series + 7-yr MA
"""

from pathlib import Path
import matplotlib.pyplot as plt
import climpy

# ── Configuration ─────────────────────────────────────────────────────

SST_FILE = Path("preprocessed_data/sst/ersstv5_annual_minus_GW.nc")
FIG_OUT  = Path("figures/fig_eof_amo_tripole.pdf")

# ── Load data ─────────────────────────────────────────────────────────

sst = climpy.load_nc(SST_FILE)
sst = climpy.standardise_coords(sst)   # ensures time/lat/lon/Xdata naming
X   = sst["Xdata"]

# ── EOF analysis ──────────────────────────────────────────────────────

solver = climpy.EOF(X, min_variance=70)
solver.summary()   # prints table of modes + explained variance

eofs = solver.eofs()     # (mode × lat × lon)
pcs  = solver.pcs()      # (time/year × mode)
frac = solver.variance_fraction()

# Construct AMO and Tripole as linear combinations of EOF1 and EOF2
eof_amo    = climpy.lincomb(eofs.isel(mode=0), eofs.isel(mode=1), a=1, b= 1)
eof_tripol = climpy.lincomb(eofs.isel(mode=0), eofs.isel(mode=1), a=1, b=-1)
pc_amo     = climpy.lincomb(pcs.isel(mode=0),  pcs.isel(mode=1),  a=1, b= 1)
pc_tripol  = climpy.lincomb(pcs.isel(mode=0),  pcs.isel(mode=1),  a=1, b=-1)

pc_amo_ma7    = climpy.moving_average(pc_amo,    n=7)
pc_tripol_ma7 = climpy.moving_average(pc_tripol, n=7)

# ── Figure ────────────────────────────────────────────────────────────

climpy.use_style("nature")

# 2 rows × 2 cols: top row = maps, bottom row = time series
fig = climpy.ClimPlot(
    nrows=2, ncols=2, w=climpy.NATURE_2COL, h=5.5,
    map_proj=(
        climpy.Globe(central_longitude=-20, central_latitude=40),
        climpy.Globe(central_longitude=-20, central_latitude=40),
        "ts",
        "ts",
    ),
)

# (a) EOF-AMO
fig[0].map(
    eof_amo,
    title=f"EOF-AMO ({frac.values[0]:.1f} %)",
    filltype="contourf",
    vmin=-0.55, vmax=0.05,
    cbar_step=0.1,
    cbar_label="SST anomaly (°C)",
    cmap=plt.cm.RdBu_r,
    land_color="lightgrey",
)

# (b) EOF-Tripole
fig[1].map(
    eof_tripol,
    title=f"EOF-Tripole ({frac.values[1]:.1f} %)",
    filltype="contourf",
    vmin=-0.6, vmax=0.6,
    cbar_step=0.2,
    cbar_label="SST anomaly (°C)",
    cmap=plt.cm.RdBu_r,
    land_color="lightgrey",
)

# (c) PC-AMO
fig[2].ts(
    pc_amo, pc_amo_ma7,
    title="PC-AMO",
    ylabel="Amplitude",
    labels=["AMO", "7-yr MA"],
    colors=["steelblue", "firebrick"],
    alphas=[0.4, 1.0],
    linewidths=[0.8, 1.5],
    markers=["", ""],
)

# (d) PC-Tripole
fig[3].ts(
    pc_tripol, pc_tripol_ma7,
    title="PC-Tripole",
    ylabel="Amplitude",
    labels=["Tripole", "7-yr MA"],
    colors=["steelblue", "firebrick"],
    alphas=[0.4, 1.0],
    linewidths=[0.8, 1.5],
)

fig.label_subplots()
fig.savefig(FIG_OUT)
fig.show()
