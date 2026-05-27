import abc
from abc import ABC

import numpy as np
import torch as xp
import plotly.express as px


class CountTransform(ABC):

    def __init__(self, np_backend=None):
        self.xp = np_backend
        if np_backend is None:
            self.xp = xp

    @abc.abstractmethod
    def forward(self, numerator: float, denominator: float) -> float:
        ...

    @abc.abstractmethod
    def inverse(self, transformed: float, denominator: float) -> float:
        ...

    def plot_correlation(self, data: np.ndarray, denominator: np.ndarray):
        data = data.ravel()
        denominator = denominator.ravel()
        transformed = self.forward(data, denominator)
        returned = self.inverse(transformed, denominator)
        px.histogram(x=transformed, title=f"Transformed values for {self.__class__.__name__}").show()
        px.histogram(x=returned, title=f"Returned values for {self.__class__.__name__}").show()
        px.scatter(x=data, y=returned,
                   title=f"Correlation between original and returned values for {self.__class__.__name__}").show()


class IdentityTransform(CountTransform):
    def forward(self, numerator: float, denominator: float) -> float:
        return numerator

    def inverse(self, transformed: float, denominator: float) -> float:
        return transformed


class Log1pTransform(CountTransform):
    def forward(self, numerator, denominator):
        # forward is called during dataset construction (numpy); inverse runs
        # inside the model (torch). Dispatch on input type so either works.
        xp = _array_module(numerator)
        return xp.log1p(numerator)

    def inverse(self, transformed, denominator):
        xp = _array_module(transformed)
        return xp.exp(transformed)


def _array_module(x):
    import torch as _torch
    if isinstance(x, _torch.Tensor):
        return _torch
    return np


class Logp1RateTransform(CountTransform):

    def forward(self, numerator, denominator):
        xp = _array_module(numerator)
        return xp.log1p(numerator) - xp.log(denominator)

    def inverse(self, transformed, denominator):
        xp = _array_module(transformed)
        return xp.exp(transformed + xp.log(denominator))


class IncidenceRateTransform(CountTransform):
    """Target = log1p(cases / population * SCALE).

    Cases are first converted to incidence per `SCALE` people, then log1p'd.
    Unlike Log1pTransform this is a true inverse-pair (`expm1` is exact),
    and the target is on a per-capita scale that's comparable across
    locations of very different sizes (e.g. provinces ranging 0.4M → 8M).
    """
    SCALE = 1000.0

    def forward(self, numerator, denominator):
        xp = _array_module(numerator)
        return xp.log1p(numerator / denominator * self.SCALE)

    def inverse(self, transformed, denominator):
        xp = _array_module(transformed)
        return xp.expm1(transformed) * denominator / self.SCALE
