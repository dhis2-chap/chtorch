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


def test_log1p_roundtrip_numpy():
    t = Log1pTransform()
    counts = np.array([0.0, 1.0, 9.0, 99.0])
    pop = np.ones_like(counts)
    np.testing.assert_allclose(t.inverse(t.forward(counts, pop), pop), counts, atol=1e-9)


def test_log1p_roundtrip_torch():
    t = Log1pTransform()
    counts = torch.tensor([0.0, 1.0, 9.0, 99.0])
    pop = torch.ones_like(counts)
    torch.testing.assert_close(t.inverse(t.forward(counts, pop), pop), counts)


def test_log1p_inverse_at_zero_is_zero():
    """Regression for the off-by-one: inverse(0) used to be 1.0."""
    t = Log1pTransform()
    assert float(t.inverse(torch.tensor(0.0), torch.tensor(1.0))) == 0.0


def test_logp1_rate_roundtrip():
    t = Logp1RateTransform()
    counts = np.array([0.0, 10.0, 100.0])
    population = np.array([100.0, 100.0, 100.0])
    np.testing.assert_allclose(t.inverse(t.forward(counts, population), population), counts, atol=1e-9)
