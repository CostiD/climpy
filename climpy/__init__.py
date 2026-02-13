"""
climpy
======
A clean Python toolkit for climate data preprocessing, analysis,
and publication-quality plotting.

Quick start
-----------
>>> import climpy

# 1. Preprocess
>>> result = climpy.preprocess(
...     "data/sst.nc", var="sst",
...     lat=(0, 80), lon=(-80, 20),
...     time=("1950-01", "2019-12"),
...     ref_period=("1981-01", "2010-12"),
...     season_months=[12, 1, 2, 3],   # DJFM
...     subtract_gm=True,
... )
>>> anom = result['anom']
>>> djfm = result['seasonal_no_gm']

# 2. EOF analysis
>>> solver = climpy.EOF(djfm)
>>> solver.summary()
>>> eofs = solver.eofs()
>>> pcs  = solver.pcs()

# 3. Plot (Nature style)
>>> climpy.use_style('nature')
>>> fig = climpy.ClimPlot(nrows=1, ncols=2, w=7.2, h=3,
...                       map_proj=(climpy.Map(), climpy.Map()))
>>> fig[0].map(eofs.isel(mode=0), title='EOF 1', cbar_label='°C',
...            vmin=-0.6, vmax=0.6)
>>> fig[1].map(eofs.isel(mode=1), title='EOF 2', cbar_label='°C',
...            vmin=-0.6, vmax=0.6)
>>> fig.label_subplots()
>>> fig.savefig('figures/fig1.pdf')
"""

# ── Plotting ──────────────────────────────────────────────────────────
from .plot import (
    ClimPlot,
    Map, Globe, Mollweide, Robinson, NorthPolarStereo, SouthPolarStereo,
    # short aliases
    Moll, Rob, NPole,
)

# ── Style ─────────────────────────────────────────────────────────────
from .style import (
    use_style,
    style_context,
    NATURE_1COL, NATURE_15COL, NATURE_2COL,
    AGU_1COL, AGU_15COL, AGU_2COL,
    AMS_1COL, AMS_2COL,
)

# ── Data preprocessing ────────────────────────────────────────────────
from .data import (
    preprocess,
    load_nc,
    fix_time,
    standardise_coords,
    set_lon_convention,
    mask_fill_values,
    subset,
    reduce_resolution,
    anomalies,
    annual_means,
    seasonal_means,
    global_mean,
    subtract_global_mean,
    drop_sparse_gridpoints,
)

# ── Operations ────────────────────────────────────────────────────────
from .ops import (
    moving_average,
    MA,                         # alias
    lag_pair,
    cosine_weights,
    getneofs,
    lincomb,
)

# ── Analysis ──────────────────────────────────────────────────────────
from .analysis import (
    EOF,
    CCA,
    pearson_correlation,
    correlation_pvalue,
    linear_regression,
    regression_pvalue,
)

__version__ = "0.2.0"

__all__ = [
    # Plotting
    "ClimPlot",
    "Map", "Globe", "Mollweide", "Robinson",
    "NorthPolarStereo", "SouthPolarStereo",
    "Moll", "Rob", "NPole",
    # Style
    "use_style", "style_context",
    "NATURE_1COL", "NATURE_15COL", "NATURE_2COL",
    "AGU_1COL", "AGU_15COL", "AGU_2COL",
    "AMS_1COL", "AMS_2COL",
    # Data
    "preprocess",
    "load_nc", "fix_time", "standardise_coords",
    "set_lon_convention", "mask_fill_values",
    "subset", "reduce_resolution",
    "anomalies", "annual_means", "seasonal_means",
    "global_mean", "subtract_global_mean",
    "drop_sparse_gridpoints",
    # Ops
    "moving_average", "MA", "lag_pair",
    "cosine_weights", "getneofs", "lincomb",
    # Analysis
    "EOF", "CCA",
    "pearson_correlation", "correlation_pvalue",
    "linear_regression", "regression_pvalue",
]
