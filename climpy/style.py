"""
climpy.style
============
Matplotlib style presets for publication-quality climate figures.

Supports Nature, AGU/JGR, and AMS journal specifications.

Usage
-----
>>> import climpy
>>> climpy.use_style('nature')        # apply globally
>>> climpy.use_style('ams')
>>> climpy.use_style('default')       # restore matplotlib defaults

Or as a context manager:
>>> import climpy.style as cs
>>> with cs.style_context('nature'):
...     fig = ClimPlot(...)
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
from contextlib import contextmanager

# ── Width constants (inches) ──────────────────────────────────────────
# Nature, Science
NATURE_1COL   = 3.50   # 89  mm
NATURE_15COL  = 4.72   # 120 mm
NATURE_2COL   = 7.20   # 183 mm

# AGU / JGR
AGU_1COL      = 3.74   # 95  mm
AGU_15COL     = 5.71   # 145 mm
AGU_2COL      = 7.48   # 190 mm

# AMS (e.g. J. Climate)
AMS_1COL      = 3.27   # 83  mm
AMS_2COL      = 6.77   # 172 mm


# ── rcParams dictionaries ─────────────────────────────────────────────

_BASE = {
    # Font
    "font.family"           : "sans-serif",
    "font.sans-serif"       : ["Helvetica", "Arial", "DejaVu Sans"],
    "mathtext.fontset"      : "stixsans",
    # Lines & patches
    "lines.linewidth"       : 1.0,
    "patch.linewidth"       : 0.5,
    # Axes
    "axes.linewidth"        : 0.5,
    "axes.spines.top"       : True,
    "axes.spines.right"     : True,
    # Ticks
    "xtick.major.width"     : 0.5,
    "xtick.minor.width"     : 0.4,
    "ytick.major.width"     : 0.5,
    "ytick.minor.width"     : 0.4,
    "xtick.major.size"      : 3.0,
    "ytick.major.size"      : 3.0,
    "xtick.direction"       : "out",
    "ytick.direction"       : "out",
    # Legend
    "legend.framealpha"     : 0.8,
    "legend.edgecolor"      : "0.8",
    "legend.handlelength"   : 1.5,
    # Figure / saving
    "figure.dpi"            : 150,
    "savefig.dpi"           : 300,
    "savefig.bbox"          : "tight",
    "savefig.pad_inches"    : 0.05,
    # Color cycle (colorblind-friendly, Wong 2011)
    "axes.prop_cycle"       : mpl.cycler(color=[
        "#0077BB",   # blue
        "#EE7733",   # orange
        "#009988",   # teal
        "#CC3311",   # red
        "#33BBEE",   # cyan
        "#EE3377",   # magenta
        "#BBBBBB",   # gray
    ]),
}

NATURE_RC = {
    **_BASE,
    "font.size"             : 7,
    "axes.titlesize"        : 8,
    "axes.labelsize"        : 7,
    "xtick.labelsize"       : 6,
    "ytick.labelsize"       : 6,
    "legend.fontsize"       : 6,
    "legend.title_fontsize" : 6,
}

AGU_RC = {
    **_BASE,
    "font.size"             : 9,
    "axes.titlesize"        : 10,
    "axes.labelsize"        : 9,
    "xtick.labelsize"       : 8,
    "ytick.labelsize"       : 8,
    "legend.fontsize"       : 8,
}

AMS_RC = {
    **_BASE,
    "font.size"             : 8,
    "axes.titlesize"        : 9,
    "axes.labelsize"        : 8,
    "xtick.labelsize"       : 7,
    "ytick.labelsize"       : 7,
    "legend.fontsize"       : 7,
}

_STYLES = {
    "nature"  : NATURE_RC,
    "science" : NATURE_RC,
    "agu"     : AGU_RC,
    "jgr"     : AGU_RC,
    "ams"     : AMS_RC,
    "jclimate": AMS_RC,
}


def use_style(name: str = "nature") -> None:
    """Apply a publication rcParams preset globally.

    Parameters
    ----------
    name : {'nature', 'science', 'agu', 'jgr', 'ams', 'jclimate', 'default'}
    """
    if name == "default":
        mpl.rcdefaults()
        return
    key = name.lower()
    if key not in _STYLES:
        raise ValueError(
            f"Style '{name}' not recognised. "
            f"Choose from: {list(_STYLES)} or 'default'."
        )
    plt.rcParams.update(_STYLES[key])


@contextmanager
def style_context(name: str = "nature"):
    """Context manager: temporarily apply a style, then restore previous settings.

    Example
    -------
    >>> with climpy.style.style_context('nature'):
    ...     fig = ClimPlot(...)
    ...     fig.savefig('fig1.pdf')
    """
    prev = dict(mpl.rcParams)
    try:
        use_style(name)
        yield
    finally:
        mpl.rcParams.update(prev)
