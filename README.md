# climpy

A clean, reusable Python toolkit for climate data preprocessing, analysis, and
publication-quality plotting (Nature / AGU / AMS journal style).

---

## Installation

```bash
# Clone / copy the package, then install in editable mode:
pip install -e /path/to/climpy_package

# Dependencies installed automatically:
# numpy, xarray, pandas, matplotlib, cartopy, scipy, eofs
```

---

## Package structure

```
climpy/
├── __init__.py        ← all public functions exported here
├── style.py           ← Nature / AGU / AMS rcParams presets
├── data.py            ← loading, preprocessing, anomalies, seasonal means
├── ops.py             ← moving average, lag, cosine weights, lincomb
├── plot.py            ← ClimPlot + projection helpers
└── analysis/
    ├── eof.py         ← EOF (PCA) wrapper
    ├── cca.py         ← BP-CCA
    ├── correlation.py ← Pearson r + p-values
    └── regression.py  ← regression slopes + p-values

examples/
├── example_preprocess.py
├── example_eof.py
├── example_cca.py
└── example_corr_regression.py
```

---

## Quick start

```python
import climpy
```

### 1  Preprocessing

```python
result = climpy.preprocess(
    "data/ersstv5.nc",
    var            = "sst",
    lon_convention = "[-180,180]",
    time           = ("1950-01", "2019-12"),
    lat            = (0, 80),
    lon            = (-80, 20),
    ref_period     = ("1981-01", "2010-12"),    # climatology for anomalies
    season_months  = [12, 1, 2, 3],            # DJFM
    subtract_gm    = True,                     # remove global warming
)

anom            = result["anom"]              # monthly anomalies
annual          = result["annual"]            # annual means
djfm            = result["seasonal"]          # DJFM means
annual_no_gw    = result["annual_no_gm"]      # annual - global mean
djfm_no_gw      = result["seasonal_no_gm"]
```

You can also use the lower-level functions individually:

```python
ds   = climpy.load_nc("data/sst.nc", var="sst", squeeze="lev")
ds   = climpy.standardise_coords(ds, time="T", lat="latitude")
ds   = climpy.set_lon_convention(ds, "[-180,180]")
da   = climpy.subset(ds["Xdata"], time=("1950","2019"), lat=(0,80))
anom = climpy.anomalies(da, ref_period=("1981","2010"))
ann  = climpy.annual_means(anom)
seas = climpy.seasonal_means(anom, months=[12, 1, 2, 3])
gm   = climpy.global_mean(ann)
detr = climpy.subtract_global_mean(ann, gm)
```

---

### 2  EOF analysis

```python
solver = climpy.EOF(djfm_no_gw, min_variance=70)
solver.summary()               # prints table of modes + explained variance

eofs = solver.eofs()           # xr.DataArray (mode × lat × lon)
pcs  = solver.pcs()            # xr.DataArray (time/year × mode)
frac = solver.variance_fraction()   # % per mode

# Linear combinations of EOFs (e.g. AMO and Tripole)
eof_amo    = climpy.lincomb(eofs.isel(mode=0), eofs.isel(mode=1), a=1, b= 1)
eof_tripol = climpy.lincomb(eofs.isel(mode=0), eofs.isel(mode=1), a=1, b=-1)
pc_smooth  = climpy.moving_average(pcs.isel(mode=0), n=7)
```

---

### 3  CCA

```python
cca = climpy.CCA(sst_anom, slp_anom, min_variance=70)
cca.summary()

Xpat  = cca.x_patterns()           # (mode × lat × lon)
Ypat  = cca.y_patterns()
Xts   = cca.x_timeseries()         # (time × mode)
Yts   = cca.y_timeseries()
corrs = cca.canonical_correlations()
varX, varY = cca.variance_fraction()
```

---

### 4  Correlation & regression

```python
# Fast vectorised correlation (no p-values)
r = climpy.pearson_correlation(pc1, precip_anom, dim="year")

# With p-values for significance stippling
r, pval = climpy.correlation_pvalue(pc1, precip_anom, dim="year")

# Regression slope (units of field per σ of series)
slope = climpy.linear_regression(pc1, precip_anom, dim="year")
slope, pval = climpy.regression_pvalue(pc1, precip_anom, dim="year")
```

---

### 5  Publication-quality figures

#### Apply a journal style

```python
climpy.use_style("nature")    # or "agu", "ams"

# As a context manager (does not change global settings):
with climpy.style_context("nature"):
    fig = climpy.ClimPlot(...)
```

#### Create a figure

```python
import matplotlib.pyplot as plt

# Recommended widths (inches):
# climpy.NATURE_1COL = 3.5  (89 mm)
# climpy.NATURE_2COL = 7.2  (183 mm)

fig = climpy.ClimPlot(
    nrows=2, ncols=2,
    w=climpy.NATURE_2COL, h=5.5,
    map_proj=(
        climpy.Map(),          # top-left:  map
        climpy.Map(),          # top-right: map
        "ts",                  # bottom-left:  time series
        "ts",                  # bottom-right: time series
    ),
)

# Short-hand API:  fig[i].map(...)  or  fig[i].ts(...)
fig[0].map(
    eof1,
    title="EOF 1",
    vmin=-0.6, vmax=0.6,
    cbar_label="SST anomaly (°C)",
    cmap=plt.cm.RdBu_r,
)
fig[1].map(eof2, title="EOF 2", vmin=-0.6, vmax=0.6, cbar_label="SST anomaly (°C)")

fig[2].ts(pc1, pc1_smooth,
          labels=["PC 1", "7-yr MA"],
          colors=["steelblue", "firebrick"],
          alphas=[0.4, 1.0])

fig[3].ts(pc2, pc2_smooth,
          labels=["PC 2", "7-yr MA"],
          colors=["steelblue", "firebrick"])

# Significance stippling
fig.add_stippling(fig.axes[0], pval.values, lat, lon, pthresh=0.05)

# Subplot labels: (a), (b), (c)...
fig.label_subplots()               # default: (a), (b), (c)...
fig.label_subplots(style="upper")  # → (A), (B), (C)...

# Save
fig.savefig("figures/fig1.pdf")    # PDF vector (best for journals)
fig.savefig("figures/fig1.png")    # PNG 300 DPI
```

---

### 6  Available projections

| Function | Cartopy projection |
|---|---|
| `climpy.Map(clon)` | PlateCarree |
| `climpy.Globe(clon, clat)` | Orthographic |
| `climpy.Mollweide(clon)` | Mollweide |
| `climpy.Robinson(clon)` | Robinson |
| `climpy.NorthPolarStereo(clon)` | NorthPolarStereo |
| `climpy.SouthPolarStereo(clon)` | SouthPolarStereo |

---

## API reference summary

### `climpy.data`
| Function | Description |
|---|---|
| `preprocess(path, ...)` | One-stop loader: anomalies, annual/seasonal means, GW removal |
| `load_nc(path, var, squeeze)` | Open a .nc file |
| `standardise_coords(ds, time, lat, lon, var)` | Rename to standard names |
| `set_lon_convention(ds, convention)` | Convert to [-180,180] or [0,360] |
| `subset(da, time, lat, lon)` | Select region/period |
| `anomalies(da, ref_period)` | Compute monthly anomalies |
| `annual_means(da)` | Calendar-year means |
| `seasonal_means(da, months)` | Multi-month seasonal means |
| `global_mean(da)` | Area-weighted global mean |
| `subtract_global_mean(da, gm)` | Remove GW signal |
| `drop_sparse_gridpoints(da, max_nan_fraction)` | Mask sparse time series |

### `climpy.analysis`
| Function / Class | Description |
|---|---|
| `EOF(da, n_eofs, min_variance)` | EOF analysis |
| `CCA(X, Y, n_eofs, min_variance)` | BP-CCA |
| `pearson_correlation(s, F, dim)` | Fast vectorised Pearson r |
| `correlation_pvalue(s, F, dim)` | r + two-tailed p-value |
| `linear_regression(s, F, dim)` | Regression slopes (vectorised) |
| `regression_pvalue(s, F, dim)` | Slopes + p-values |

### `climpy.plot`
| | Description |
|---|---|
| `ClimPlot(nrows, ncols, w, h, map_proj)` | Multi-panel figure |
| `fig[i].map(X, ...)` | Fill subplot i with a spatial pattern |
| `fig[i].ts(*series, ...)` | Plot time series on subplot i |
| `fig.add_stippling(ax, pval, lat, lon)` | Significance stippling |
| `fig.label_subplots(style='paren')` | Add (a), (b), (c)... |
| `fig.savefig(path)` | Save at 300 DPI / vector |

### `climpy.ops`
| Function | Description |
|---|---|
| `moving_average(da, n)` | Centred n-point moving average |
| `lag_pair(series, field, k)` | Build lagged (series, field) pair |
| `lincomb(X, Y, a, b)` | a·X + b·Y |
| `cosine_weights(da)` | √cos(lat) area weights |
| `getneofs(solver, percent)` | Min EOFs for ≥ P% variance |

---

## Changelog

### 0.2.0
- Complete rewrite; non-interactive preprocessing
- `ClimPlot` with `label_subplots()`, `savefig()`, `add_stippling()`
- Nature / AGU / AMS style presets
- `xr.corr()` backend for fast vectorised correlations
- All known bugs fixed (DivergingNorm, thresh bug, rename_variable, marker names)
