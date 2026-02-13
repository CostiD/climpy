"""
example_cca.py
==============
BP-CCA between North Atlantic SST and global SLP anomalies.
Produces a 2×3 figure (modes 1 and 2):
  (a) SST CCP-1   (b) SST CCP-2
  (c) SLP CCP-1   (d) SLP CCP-2
  (e) Time series pair 1   (f) Time series pair 2
"""

from pathlib import Path
import matplotlib.pyplot as plt
import climpy

# ── Configuration ─────────────────────────────────────────────────────

SST_FILE = Path("preprocessed_data/sst/ersstv5_annual_minus_GW.nc")
SLP_FILE = Path("preprocessed_data/slp/hadslpv2r_DJFM.nc")
FIG_OUT  = Path("figures/fig_cca_sst_slp.pdf")

MODES_TO_PLOT = [0, 1]    # 0-based indices of modes to show

# ── Load data ─────────────────────────────────────────────────────────

sst = climpy.load_nc(SST_FILE)
sst = climpy.standardise_coords(sst)["Xdata"]

slp = climpy.load_nc(SLP_FILE)
slp = climpy.standardise_coords(slp)["Xdata"]

# ── CCA ───────────────────────────────────────────────────────────────

cca = climpy.CCA(sst, slp, min_variance=70)
cca.summary()   # prints mode table

Xpat  = cca.x_patterns()                # (mode × lat × lon)
Ypat  = cca.y_patterns()
Xts   = cca.x_timeseries()              # (time × mode)
Yts   = cca.y_timeseries()
corrs = cca.canonical_correlations()
varX, varY = cca.variance_fraction()

# ── Figure ────────────────────────────────────────────────────────────

climpy.use_style("nature")

n_modes = len(MODES_TO_PLOT)
# Layout: 3 rows × n_modes columns
fig = climpy.ClimPlot(
    nrows=3, ncols=n_modes,
    n=3 * n_modes,
    w=climpy.NATURE_2COL, h=7.5,
    map_proj=(
        *[climpy.Map() for _ in MODES_TO_PLOT],    # row 1: SST maps
        *[climpy.Map() for _ in MODES_TO_PLOT],    # row 2: SLP maps
        *["ts"         for _ in MODES_TO_PLOT],    # row 3: time series
    ),
)

for col, mode in enumerate(MODES_TO_PLOT):
    m = mode + 1   # 1-based label

    # Row 1: SST patterns
    fig[col].map(
        Xpat.isel(mode=mode),
        title=f"SST CCP-{m} ({varX[mode]:.1f} %)",
        lon_ticks=[-75, -60, -30, 0, 20],
        lat_ticks=[0, 20, 40, 60, 80],
        map_extent=[-75, 20, 0, 80],
        filltype="contourf",
        cmap=plt.cm.RdBu_r,
        shrink=0.8,
        cbar_label="SST anomaly (°C)",
    )

    # Row 2: SLP patterns
    fig[n_modes + col].map(
        Ypat.isel(mode=mode),
        title=f"SLP CCP-{m} ({varY[mode]:.1f} %)",
        lon_ticks=[-180, -120, -60, 0, 60, 120, 180],
        lat_ticks=[-90, -60, -30, 0, 30, 60, 90],
        filltype="contourf",
        cmap=plt.cm.RdBu_r,
        shrink=0.8,
        cbar_label="SLP anomaly (hPa)",
    )

    # Row 3: Time series pairs
    fig[2 * n_modes + col].ts(
        Xts.isel(mode=mode),
        Yts.isel(mode=mode),
        title=f"Pair {m}  (r = {corrs[mode]:.2f})",
        ylabel="Amplitude",
        labels=["SST", "SLP"],
        colors=["steelblue", "firebrick"],
        alphas=[0.8, 0.8],
    )

fig.label_subplots()
fig.savefig(FIG_OUT)
fig.show()
