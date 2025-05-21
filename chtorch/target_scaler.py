import numpy as np
import torch


class TargetScaler:
    def __init__(self, target_matrix):
        self._mu = torch.from_numpy(np.nanmean(target_matrix, axis=0))
        self._mu = torch.where(torch.isnan(self._mu), 0, self._mu)
        assert not np.isnan(self._mu).any(), f"NaN in mu {self._mu}"
        self._std = torch.from_numpy(np.nanstd(target_matrix, axis=0))
        self._std = torch.where((self._std == 0) | torch.isnan(self._std), 1, self._std)
        assert not torch.isnan(self._std).any(), f"NaN in std {self._std}"

    @property
    def mu(self):
        return self._mu

    @property
    def std(self):
        return self._std

    def scale_by_location(self, locations, eta):
        """
        Scale the eta values by the mean and standard deviation of the target matrix for each location.
        """
        mu = self._mu[locations, None]
        std = self._std[locations, None]
        new_eta = eta[..., 0] * std + mu
        return torch.concat([new_eta[..., None], eta[..., 1:]], dim=-1)


class MultiTargetScaler(TargetScaler):
    def __init__(self, scalers: list[TargetScaler]):
        print([s.mu.shape for s in scalers])
        self._mu = torch.cat([scaler.mu for scaler in scalers])
        self._std = torch.cat([scaler.std for scaler in scalers])
