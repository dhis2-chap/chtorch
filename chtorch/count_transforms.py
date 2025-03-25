import abc
from abc import ABC

import numpy as np
import torch as xp

class CountTransform(ABC):

    @abc.abstractmethod
    def forward(self, numerator: float, denominator: float) -> float:
        ...

    @abc.abstractmethod
    def inverse(self, transformed: float, denominator: float) -> float:
        ...


class Log1pTransform(CountTransform):
    def forward(self, numerator: float, denominator: float) -> float:
        return np.log1p(numerator)

    def inverse(self, transformed: float, denominator: float) -> float:
        return xp.exp(transformed)


class Logp1RateTransform(CountTransform):

    def forward(self, numerator: float, denominator: float) -> float:
        return np.log1p(numerator) - np.log(denominator)

    def inverse(self, transformed: float, denominator: float) -> float:
        return xp.expm1(transformed + xp.log(denominator))
