#!/usr/bin/env python3

from __future__ import annotations

from typing import Optional

import torch

from ..module import Module
from ..constraints import Interval, Positive
from ..priors import Prior

class Power(Module):
    def __init__(
        self, 
        power_init: torch.Tensor = torch.tensor(1.0),
        power_constraint: Optional[Interval] = None,
        power_prior: Optional[Prior] = None
    ):
        super(Power, self).__init__()
        if power_constraint is None:
            power_constraint = Positive()
        
        # set parameter
        self.register_parameter(
            name="raw_power", 
            parameter=torch.nn.Parameter(power_constraint.inverse_transform(power_init))
        )
        self.shape = self.raw_power.shape
        # set constraint
        self.register_constraint("raw_power", power_constraint)
        # set prior
        if power_prior is not None:
            if not isinstance(power_prior, Prior):
                raise TypeError("Expected gpytorch.priors.Prior but got " + type(power_prior).__name__)
            self.register_prior("power_prior", power_prior, self._power_param, self._power_closure)
    
    def _power_param(self, q: Power) -> torch.Tensor:
        # Used by the raw_power
        return q.power

    def _power_closure(self, q: Power, v: torch.Tensor) -> torch.Tensor:
        # Used by the raw_power
        return q._set_power(v)

    @property
    def power(self) -> torch.Tensor:
        return self.raw_power_constraint.transform(self.raw_power)

    @power.setter
    def power(self, value: torch.Tensor) -> torch.Tensor:
        self._set_power(value)

    def _set_power(self, value: torch.Tensor):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_power)
        self.initialize(raw_power=self.raw_power_constraint.inverse_transform(value))
    
    @property
    def data(self) -> torch.Tensor:
        return self.power.data
    
    def __truediv__(self, other):
        return self.power/other
    
    def __rtruediv__(self, other):
        return other/self.power
    
    def __rpow__(self, other):
        return other**self.power
    
    def __ne__(self, other):
        return self.power!=other
    
    def __lt__(self, other):
        return self.power<other
    
    def __gt__(self, other):
        return self.power>other
    
    def numel(self):
        return self.power.numel()
    
    def size(self):
        return self.power.size()