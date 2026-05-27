import numpy as np

from chtorch.configuration import ModelConfiguration
from chtorch.count_transforms import Log1pTransform
from chtorch.tensorifier import Tensorifier, _shift_back


def test_shift_back_pads_with_first_value():
    col = np.array([10.0, 20.0, 30.0, 40.0])
    out = _shift_back(col, 2)
    np.testing.assert_array_equal(out, [10.0, 10.0, 10.0, 20.0])


def test_shift_back_zero_is_identity():
    col = np.array([10.0, 20.0, 30.0])
    np.testing.assert_array_equal(_shift_back(col, 0), col)


def test_climate_lags_extend_feature_count(ch_dataset):
    """Each lag adds one column per `additional_covariate`."""
    base_cfg = ModelConfiguration(context_length=12)
    lag_cfg = ModelConfiguration(context_length=12, climate_lags=[1, 3])
    base = Tensorifier(Log1pTransform(), base_cfg)
    lagged = Tensorifier(Log1pTransform(), lag_cfg)
    X_base, *_ = base.convert(ch_dataset)
    X_lag, *_ = lagged.convert(ch_dataset)
    assert X_lag.shape[-1] == X_base.shape[-1] + 2 * len(base_cfg.additional_covariates)


def test_climate_lag_value_matches_shifted_input(ch_dataset):
    """A lag-k feature at time t should equal the raw feature at t-k."""
    cfg = ModelConfiguration(context_length=12, climate_lags=[2])
    tensorifier = Tensorifier(Log1pTransform(), cfg)
    X, *_ = tensorifier.convert(ch_dataset)
    n_base = len(cfg.additional_covariates)
    # First location, first base feature (e.g. rainfall) and its lag-2 copy
    base_col = X[:, 0, 0]
    lagged_col = X[:, 0, n_base]
    # lagged[t] == base[t-2] for t >= 2
    np.testing.assert_allclose(lagged_col[2:], base_col[:-2])
