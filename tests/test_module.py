import pytest
import torch

from chtorch.configuration import ModelConfiguration
from chtorch.module import FlatRNN, RNNConfiguration, RNNWithLocationEmbedding, main, main_flat


@pytest.fixture
def small_cfg():
    return ModelConfiguration(
        context_length=8,
        embed_dim=2,
        n_hidden=2,
        state_dim=2,
        max_dim=8,
        num_rnn_layers=1,
        output_embedding_dim=0,
    )


def test_flat_rnn_forward_shape(small_cfg):
    batch_size, time_steps, feature_dim = 3, 8, 4
    prediction_length = 3
    num_locations_total = 7

    model = FlatRNN(
        num_categories=[num_locations_total],
        input_feature_dim=feature_dim,
        prediction_length=prediction_length,
        cfg=small_cfg,
    )
    x = torch.randn(batch_size, time_steps, feature_dim)
    locations = torch.randint(0, num_locations_total, (batch_size, time_steps, 1))

    eta, past_eta = model(x, locations)
    assert eta.shape == (batch_size, prediction_length, 2)
    assert past_eta.shape == (batch_size, time_steps - 1, 2)


def test_rnn_with_location_embedding_forward_shape(small_cfg):
    batch_size, time_steps, num_locations, feature_dim = 2, 6, 4, 5
    prediction_length = 3
    num_locations_total = 9

    model = RNNWithLocationEmbedding(
        num_categories=[num_locations_total],
        input_feature_dim=feature_dim,
        prediction_length=prediction_length,
        cfg=small_cfg,
    )
    x = torch.randn(batch_size, time_steps, num_locations, feature_dim)
    locations = torch.randint(0, num_locations_total, (batch_size, time_steps, num_locations, 1))

    out = model(x, locations)
    assert out.shape == (batch_size, prediction_length, num_locations, 2)


def test_module_smoke_main():
    # Catches regressions in the in-file smoke tests.
    main_flat()
    main()
