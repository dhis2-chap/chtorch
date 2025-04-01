import torch
from chtorch.count_transforms import CountTransform
from torch import nn


class NegativeBinomialLoss(nn.Module):
    def __init__(self, count_transform: CountTransform):
        super().__init__()
        self._count_transform = count_transform

    def forward(self, eta, y_true, population):
        """
        y_pred: (batch_size, 2)  - First column: mean (μ), Second column: dispersion (θ)
        y_true: (batch_size, 1)  - Observed counts
        """
        na_mask = ~torch.isnan(y_true)
        y_true = y_true[na_mask]
        population = population[na_mask]
        eta = eta[na_mask]
        nb_dist = get_dist(eta, population, self._count_transform)
        loss = -nb_dist.log_prob(y_true).mean()
        return loss


def get_dist(eta, population, count_transform):
    return torch.distributions.NegativeBinomial(
        total_count=count_transform.inverse(eta[..., 0], population)/torch.exp(eta[..., 1]),
        logits=eta[..., 1])
