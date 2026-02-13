"""climpy.analysis — Statistical analysis tools for climate data."""

from .eof import EOF
from .cca import CCA
from .correlation import pearson_correlation, correlation_pvalue
from .regression import linear_regression, regression_pvalue

__all__ = [
    "EOF",
    "CCA",
    "pearson_correlation",
    "correlation_pvalue",
    "linear_regression",
    "regression_pvalue",
]
