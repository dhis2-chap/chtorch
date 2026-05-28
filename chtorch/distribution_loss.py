import abc

import torch
from chtorch.count_transforms import CountTransform
from torch import nn

from chtorch.distributions import NegativeBinomialWithNan


class MaskedNANLoss(abc.ABC, nn.Module):
    n_parameters = 0

    @staticmethod
    @abc.abstractmethod
    def get_dist(eta, population, count_transform):
        pass

    def __init__(self, count_transform: CountTransform):
        super().__init__()
        self._count_transform = count_transform

    def forward(self, eta, y_true, population):
        na_mask = ~torch.isnan(y_true)
        # Align population to (...,) y_true shape so transforms that depend
        # on it (e.g. IncidenceRateTransform) get a matching tensor.
        if population.shape == y_true.shape:
            population_aligned = population
        else:
            # Population is per-location-constant (smooth_population maps to
            # the median), so broadcasting one column to the y shape is OK.
            ref = population[..., :1] if population.ndim >= 1 else population
            population_aligned = ref.expand_as(y_true)
        y_true = y_true[na_mask]
        eta = eta[na_mask]
        population_aligned = population_aligned[na_mask]
        nb_dist = self.get_dist(eta, population_aligned, self._count_transform)
        loss = -nb_dist.log_prob(y_true).mean()
        return loss


_MIN_TOTAL_COUNT = 1e-6
_MAX_TOTAL_COUNT = 1e8


def _safe_total_count(eta, population, count_transform):
    """Convert (eta, population) to NB total_count, clamped to a safe finite
    range. Without the persistence skip the network can output very large or
    very negative eta values at the start of training, which combined with
    expm1(·) overflows or makes total_count negative / NaN. clamp + nan_to_num
    keeps torch.distributions.NegativeBinomial valid."""
    mean = count_transform.inverse(eta[..., 0], population)
    total = mean / torch.exp(eta[..., 1])
    total = torch.nan_to_num(
        total,
        nan=_MIN_TOTAL_COUNT,
        posinf=_MAX_TOTAL_COUNT,
        neginf=_MIN_TOTAL_COUNT,
    )
    return total.clamp(min=_MIN_TOTAL_COUNT, max=_MAX_TOTAL_COUNT)


class NegativeBinomialLoss(MaskedNANLoss):
    n_parameters = 2

    @staticmethod
    def get_dist(eta, population, count_transform):
        return torch.distributions.NegativeBinomial(
            total_count=_safe_total_count(eta, population, count_transform),
            logits=eta[..., 1])


class PoissonLoss(MaskedNANLoss):
    n_parameters = 1

    @staticmethod
    def get_dist(eta, population, count_transform):
        rate = count_transform.inverse(eta[..., 0], population).clamp_min(_MIN_TOTAL_COUNT)
        return torch.distributions.Poisson(rate=rate)


class NBLossWithNaN(NegativeBinomialLoss):
    n_parameters = 3

    @staticmethod
    def get_dist(eta, population, count_transform):
        return NegativeBinomialWithNan(
            nan_logits=eta[..., 2],
            total_count=_safe_total_count(eta, population, count_transform),
            logits=eta[..., 1])

    def forward(self, eta, y_true, population):
        """NB-with-NaN loss. Unlike MaskedNANLoss we don't drop NaN cells —
        NBLossWithNaN models them via the third eta channel — but we still
        need population to broadcast to y_true's shape for transforms that
        use it (e.g. IncidenceRateTransform)."""
        if population.shape != y_true.shape:
            ref = population[..., :1] if population.ndim >= 1 else population
            population = ref.expand_as(y_true)
        dist = self.get_dist(eta, population, self._count_transform)
        loss = -dist.log_prob(y_true).mean()
        return loss


def get_dist(eta, population, count_transform):
    return torch.distributions.NegativeBinomial(
        total_count=count_transform.inverse(eta[..., 0], population) / torch.exp(eta[..., 1]),
        logits=eta[..., 1])
