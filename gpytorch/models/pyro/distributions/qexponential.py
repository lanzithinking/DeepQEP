#!/usr/bin/env python3

import pyro

from ...distributions import Distribution, QExponential as GQExponential


class QExponential(GQExponential, Distribution):
    pass
