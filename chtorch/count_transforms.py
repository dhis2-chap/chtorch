import abc
from abc import ABC

import numpy as np
import torch as xp
from matplotlib import pyplot as plt
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
        print(np.histogram(data))
        print(np.histogram(transformed))
        px.histogram(x=transformed, title=f"Transformed values for {self.__class__.__name__}").show()
        px.histogram(x=returned, title=f"Returned values for {self.__class__.__name__}").show()
        px.scatter(x=data, y=returned,
                   title=f"Correlation between original and returned values for {self.__class__.__name__}").show()


class Log1pTransform(CountTransform):
    def forward(self, numerator: float, denominator: float) -> float:
        return np.log1p(numerator)

    def inverse(self, transformed: float, denominator: float) -> float:
        return self.xp.exp(transformed)


class Logp1RateTransform(CountTransform):

    def forward(self, numerator: float, denominator: float) -> float:
        return np.log1p(numerator) - np.log(denominator)

    def inverse(self, transformed: float, denominator: float) -> float:
        return self.xp.exp(transformed + self.xp.log(denominator))
