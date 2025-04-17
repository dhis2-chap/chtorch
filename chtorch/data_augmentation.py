from abc import ABC

import numpy as np

import torch.distributions


class Augmentation(ABC):
    def transform(self, data):
        raise NotImplementedError("Subclasses should implement this method.")


class PoissonAugmentation:
    def transform(self, data: tuple):
        *x, y, population = data
        new_y = y.copy()
        is_nan = np.isnan(new_y)
        new_y[~is_nan] = np.random.poisson(new_y[~is_nan])
        #y = scipy.stats.poisson.rvs(y)
        return *x, new_y, population
        #return *x, torch.distributions.Poisson(y).sample(), population
