import torch.distributions


class NegativeBinomialWithNan:
    def __init__(self, nan_logits, *args, **kwargs):
        #self._sigmoids = nan_sigmoids
        self._nb = torch.distributions.NegativeBinomial(*args, **kwargs)
        self._bernoulli = torch.distributions.Bernoulli(logits=nan_logits)

    def log_prob(self, y_true):
        '''
        P(y is nan) = expit(sigmoid)
        P(y) =
        '''
        is_nan = torch.isnan(y_true)
        nan_log_probs = self._bernoulli.log_prob(torch.where(is_nan, 1., 0.))
        #nan_log_probs = torch.binary_cross_entropy_with_logits(self._sigmoids,
        y_repl = torch.where(is_nan, 10., y_true)
        y_log_probs = torch.where(is_nan, 0, self._nb.log_prob(y_repl))
        return y_log_probs+nan_log_probs

    def sample(self, shape):
        is_nan = self._bernoulli.sample(shape)
        y = self._nb.sample(shape)
        return torch.where(is_nan == 1, 0, y) #TODO: maybe put in nans and replace later


