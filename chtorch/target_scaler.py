import numpy as np
import torch


class TargetScaler:
    def __init__(self, target_matrix):
        self._mu = torch.from_numpy(np.nanmean(target_matrix, axis=0))
        self._mu = torch.where(torch.isnan(self._mu), torch.zeros_like(self._mu), self._mu)
        assert not torch.isnan(self._mu).any(), f"NaN in mu {self._mu}"
        self._std = torch.from_numpy(np.nanstd(target_matrix, axis=0))
        self._std = torch.where(
            (self._std == 0) | torch.isnan(self._std), torch.ones_like(self._std), self._std
        )
        assert not torch.isnan(self._std).any(), f"NaN in std {self._std}"

    @property
    def mu(self):
        return self._mu

    @property
    def std(self):
        return self._std

    def scale_by_location(self, locations, eta):
        """
        Un-standardize the first column of eta (the mean parameter) using the
        per-location mean and std of the (transformed) target. Other columns
        are passed through unchanged.
        """
        if isinstance(locations, torch.Tensor):
            locations = locations.long()
        assert locations.ndim == 1, f"locations should be 1D, got {locations.ndim}"
        mu = self._mu[locations].to(dtype=eta.dtype, device=eta.device)
        std = self._std[locations].to(dtype=eta.dtype, device=eta.device)
        while mu.ndim < eta.ndim - 1:
            mu = mu.unsqueeze(-1)
            std = std.unsqueeze(-1)
        new_first = eta[..., 0] * std + mu
        return torch.cat([new_first.unsqueeze(-1), eta[..., 1:]], dim=-1)


class MultiTargetScaler(TargetScaler):
    def __init__(self, scalers: list[TargetScaler]):
        print([s.mu.shape for s in scalers])
        self._mu = torch.cat([scaler.mu for scaler in scalers])
        self._std = torch.cat([scaler.std for scaler in scalers])
