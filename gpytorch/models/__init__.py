#!/usr/bin/env python3

import warnings

from . import deep_gps, exact_prediction_strategies, gplvm, pyro
from .approximate_gp import ApproximateGP
from .approximate_qep import ApproximateQEP
from .exact_gp import ExactGP
from .exact_qep import ExactQEP
from .gp import GP
from .qep import QEP
from .model_list import AbstractModelList, IndependentModelList
from .pyro import PyroGP

# Alternative name for ApproximateGP
VariationalGP = ApproximateGP


# Deprecated for 0.4 release
class AbstractVariationalGP(ApproximateGP):
    # Remove after 1.0
    def __init__(self, *args, **kwargs):
        warnings.warn("AbstractVariationalGP has been renamed to ApproximateGP.", DeprecationWarning)
        super().__init__(*args, **kwargs)


# Deprecated for 0.4 release
class PyroVariationalGP(ApproximateGP):
    # Remove after 1.0
    def __init__(self, *args, **kwargs):
        warnings.warn("PyroVariationalGP has been renamed to PyroGP.", DeprecationWarning)
        super().__init__(*args, **kwargs)


__all__ = [
    "AbstractModelList",
    "ApproximateGP",
    "ApproximateQEP",
    "ExactGP",
    "ExactQEP",
    "GP",
    "QEP",
    "IndependentModelList",
    "PyroGP",
    "VariationalGP",
    "deep_gps",
    "gplvm",
    "exact_prediction_strategies",
    "pyro",
]
