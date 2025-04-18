from abc import ABC

import numpy as np

import torch.distributions

from chtorch.tensorifier import interpolate_nans

register = {}



def register_augmentation(cls):
    """
    Register an augmentation class.
    """
    if not issubclass(cls, Augmentation):
        raise ValueError(f"Class {cls.__name__} is not a subclass of Augmentation.")
    register[cls.name] = cls
    return cls

def get_augmentation(name: str):
    """
    Get an augmentation class by name.
    """
    if name not in register:
        raise ValueError(f"Augmentation {name} not found.")
    return register[name]

class Augmentation(ABC):
    name: str
    def transform(self, data):
        raise NotImplementedError("Subclasses should implement this method.")

@register_augmentation
class PoissonAugmentation(Augmentation):
    name = 'poisson'
    def transform(self, data: tuple):
        *x, y, population = data
        new_y = y.copy()
        is_nan = np.isnan(new_y)
        new_y[~is_nan] = np.random.poisson(new_y[~is_nan])
        return *x, new_y, population

@register_augmentation
class MaskingAugmentation(Augmentation):
    name = 'masking'
    def __init__(self, mask_prob: float = 0.1):
        self.mask_prob = mask_prob

    def transform(self, data: tuple):
        x, *rest = data
        x = x.copy()
        mask = np.random.rand(len(x) - 1) < self.mask_prob
        x[1:, ..., -2][mask] = True
        x[1:, ..., -1][mask] = np.nan
        x[1:, ..., -1] = interpolate_nans(x[1:, ..., -1])

@register_augmentation
class ScalingAugmentation(Augmentation):
    name = 'scaling'
    def __init__(self, max_scale: float = 1.5):
        self.max_scale = max_scale
        self.log_max_scale = np.log(max_scale)

    def transform(self, data: tuple):
        log_scale = np.random.uniform(-self.log_max_scale, self.log_max_scale)
        scale = np.exp(log_scale)
        x, *_, y, population = data
        y = y*scale
        population = po
        x = x.copy()
        x[..., -1] = x[..., -1] * scale
        return x, *rest
