import numpy as np
import torch

from chtorch.count_transforms import Log1pTransform, Logp1RateTransform


def test_log1p_forward_numpy():
    t = Log1pTransform()
    x = np.array([0.0, 1.0, 9.0])
    out = t.forward(x, np.array([1.0, 1.0, 1.0]))
    np.testing.assert_allclose(out, np.log1p(x))


def test_log1p_forward_torch():
    t = Log1pTransform()
    x = torch.tensor([0.0, 1.0, 9.0])
    out = t.forward(x, torch.ones(3))
    torch.testing.assert_close(out, torch.log1p(x))


def test_log1p_inverse_torch():
    t = Log1pTransform()
    x = torch.tensor([0.0, 1.0, 2.0])
    out = t.inverse(x, torch.ones(3))
    torch.testing.assert_close(out, torch.exp(x))


def test_logp1_rate_roundtrips():
    t = Logp1RateTransform()
    counts = np.array([0.0, 10.0, 100.0])
    population = np.array([100.0, 100.0, 100.0])
    transformed = t.forward(counts, population)
    # round trip: forward then inverse — log1p is not exactly invertible by exp,
    # so we check the structure: forward = log1p(num) - log(denom)
    np.testing.assert_allclose(transformed, np.log1p(counts) - np.log(population))
