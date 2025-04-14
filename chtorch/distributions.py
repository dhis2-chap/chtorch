import torch.distributions


class NegativeBinomialWithNan:
    def __init__(self, nan_sigmoids, *args, **kwargs):
        self._sigmoids = nan_sigmoids
        self._nb = torch.distributions.NegativeBinomial(*args, **kwargs)

    def log_prob(self, y_true):
        '''
        P(y is nan) = expit(sigmoid)
        P(y) =
        '''
        is_nan = torch.isnan(y_true)
        nan_log_probs = torch.binary_cross_entropy_with_logits(self._sigmoids, is_nan)
        y_log_probs = torch.where(is_nan, 0, self._nb.log_prob(y_true))
        return y_log_probs+nan_log_probs

