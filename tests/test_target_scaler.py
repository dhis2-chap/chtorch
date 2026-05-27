import numpy as np
import torch

from chtorch.target_scaler import TargetScaler


def _make_targets(n_locations=3, n_periods=20, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(loc=[1.0, 2.0, 5.0], scale=[0.5, 1.0, 2.0], size=(n_periods, n_locations))


def test_scaler_recovers_mean_and_std():
    targets = _make_targets()
    scaler = TargetScaler(targets)
    expected_mu = torch.from_numpy(np.nanmean(targets, axis=0))
    expected_std = torch.from_numpy(np.nanstd(targets, axis=0))
    torch.testing.assert_close(scaler.mu, expected_mu)
    torch.testing.assert_close(scaler.std, expected_std)


def test_scale_by_location_unstandardizes_first_column():
    targets = _make_targets()
    scaler = TargetScaler(targets)

    n_locations = 3
    eta = torch.zeros(n_locations, 2)  # standardized eta == 0 → should map to per-location mu
    locations = torch.arange(n_locations)
    scaled = scaler.scale_by_location(locations, eta)

    torch.testing.assert_close(scaled[:, 0], scaler.mu.to(scaled.dtype))
    # second column is preserved
    torch.testing.assert_close(scaled[:, 1], eta[:, 1])


def test_scale_by_location_uses_std():
    targets = _make_targets()
    scaler = TargetScaler(targets)

    n_locations = 3
    eta = torch.ones(n_locations, 2)
    locations = torch.arange(n_locations)
    scaled = scaler.scale_by_location(locations, eta)
    expected = scaler.std.to(scaled.dtype) + scaler.mu.to(scaled.dtype)
    torch.testing.assert_close(scaled[:, 0], expected)


def test_scale_by_location_works_with_extra_dims():
    targets = _make_targets()
    scaler = TargetScaler(targets)

    prediction_length = 4
    n_locations = 3
    eta = torch.zeros(n_locations, prediction_length, 2)
    locations = torch.arange(n_locations)
    scaled = scaler.scale_by_location(locations, eta)
    expected_first = scaler.mu.to(scaled.dtype).unsqueeze(-1).expand(n_locations, prediction_length)
    torch.testing.assert_close(scaled[..., 0], expected_first)


def test_zero_std_replaced_with_one():
    # All-equal column → std = 0 → should be replaced with 1
    targets = np.tile([1.0, 2.0, 3.0], (10, 1))
    scaler = TargetScaler(targets)
    torch.testing.assert_close(scaler.std, torch.ones(3, dtype=torch.float64))


def test_handles_nan_targets():
    targets = _make_targets()
    targets[:5, 0] = np.nan
    scaler = TargetScaler(targets)
    assert not torch.isnan(scaler.mu).any()
    assert not torch.isnan(scaler.std).any()
