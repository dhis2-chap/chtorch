from collections import namedtuple

import numpy as np
import torch
import torch.nn as nn
from typing import Literal
import logging
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers, dropout=0.0):
        super().__init__()
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        layers = []
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.Dropout(dropout))
            layers.append(nn.ReLU())
        self.hidden_layers = nn.Sequential(*layers)
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        self.n_layers = n_layers

    def forward(self, x):
        x = self.input_layer(x)
        x = nn.ReLU()(x)
        x = self.hidden_layers(x)
        x = self.output_layer(x)
        return x


def is_power_of_two(n):
    return (n != 0) and (n & (n - 1) == 0)


class FeatureCompressor(nn.Module):
    def __init__(self, input_dim, max_dim, output_dim, dropout=0.0):
        super().__init__()
        assert is_power_of_two(max_dim)
        # assert is_power_of_two(output_dim)
        output_power = int(np.floor(np.log2(output_dim)))
        max_power = int(np.log2(max_dim))
        # self.input_layer = nn.Linear(input_dim, max_dim)
        prev_dim = input_dim
        layers = []
        for power in range(max_power, int(output_power), -1):
            dim = int(2 ** power)
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(nn.Dropout(dropout))
            layers.append(nn.ReLU())
            prev_dim = dim
        self.hidden_layers = nn.Sequential(*layers)
        self.output_layer = nn.Linear(prev_dim, output_dim)

    def forward(self, x):
        x = self.hidden_layers(x)
        x = self.output_layer(x)
        return x


Output = namedtuple('Output', ['eta', 'past_eta'])


class RNNConfiguration(BaseModel):
    rnn_type: Literal['GRU', 'LSTM'] = 'GRU'
    embed_dim: int = 4 # Must be same as n_hidden
    num_rnn_layers: int = 1 # Capacity
    n_layers: int = 1 # Not in use
    embedding_type: Literal['sum', 'concat'] = 'concat'
    n_hidden: int = 4 # Capacity
    max_dim: int = 32 # Capacity
    state_dim: int = 4 # Capacity
    output_embedding_dim: int = 0 # Capacity
    dropout: float = 0.0 # Regulatization
    direct_ar: bool = False # Convergence


class RNNWithLocationEmbedding(nn.Module):
    def __init__(self,
                 num_categories: int,
                 input_feature_dim: int,
                 prediction_length: int,
                 output_dim: int = 2,
                 cfg: RNNConfiguration = RNNConfiguration()):
        logger.info(f"Creating RNN with config: {cfg}")
        super().__init__()
        self.output_dim = output_dim
        if cfg.embedding_type == 'sum':
            assert cfg.embed_dim == cfg.n_hidden, "Embedding dimension must be equal to hidden dimension for sum embedding"
            init_dim = input_feature_dim
        else:
            init_dim = input_feature_dim + cfg.embed_dim
        self.num_categories = num_categories
        self.location_embeddings = nn.ModuleList([nn.Embedding(num_cat, cfg.embed_dim) for num_cat in num_categories])
        self.hidden_dim = cfg.n_hidden
        self.embedding_type = cfg.embedding_type
        #self.preprocess = MLP(init_dim, cfg.n_hidden, cfg.n_hidden, cfg.n_layers, dropout=cfg.dropout)
        self.preprocess = FeatureCompressor(init_dim, cfg.max_dim, cfg.n_hidden, dropout=cfg.dropout)
        rnn_input_dim = cfg.n_hidden
        self.direct_ar = cfg.direct_ar
        if cfg.direct_ar:
            rnn_input_dim += 2
        self.rnn_input_dim = rnn_input_dim
        if cfg.rnn_type == 'GRU':
            self.rnn = nn.GRU(rnn_input_dim, cfg.state_dim, num_layers=cfg.num_rnn_layers, batch_first=True,
                              dropout=cfg.dropout)
        elif cfg.rnn_type == 'LSTM':
            self.rnn = nn.LSTM(rnn_input_dim, cfg.state_dim, num_layers=cfg.num_rnn_layers, batch_first=True,
                               dropout=cfg.dropout)
        else:
            raise ValueError("Unsupported RNN type. Use 'GRU' or 'LSTM'.")

        self.decoder = nn.GRU(1, cfg.state_dim, num_layers=cfg.num_rnn_layers, batch_first=True, dropout=cfg.dropout)
        self.output_embedding = nn.Embedding(num_categories[0], cfg.output_embedding_dim)
        dim = cfg.state_dim + cfg.output_embedding_dim
        #self.output_decoder = nn.Linear(dim, cfg.n_hidden)
        #self.ouput_layer = MLP(cfg.n_hidden, cfg.n_hidden, self.output_dim, cfg.n_layers, dropout=cfg.dropout)
        self.output_layer = FeatureCompressor(dim, cfg.max_dim, self.output_dim, dropout=cfg.dropout)
        self.prediction_length = prediction_length

    def forward(self, x, locations):
        """
        x: (batch, time, location, features)
        locations: (batch, time, location) - location indices
        """
        batch_size, time_steps, num_locations, feature_dim = x.shape

        # Embed locations: (batch, time, location) -> (batch, time, location, 4)
        loc_embeds = sum(embedding(locations[..., i]) for i, embedding in enumerate(self.location_embeddings))

        # Concatenate features with location embeddings
        x_with_loc = torch.cat([x, loc_embeds], dim=-1)  # (batch, time, location, feature_dim + 4)
        x_with_loc = x_with_loc.swapaxes(1, 2)  # (batch, location, time, feature_dim + 4

        # Reshape for RNN: merge location into feature dimension
        x_rnn = x_with_loc.reshape(batch_size * num_locations, time_steps,
                                   -1)  # (batch, time, location * (feature_dim + 4))
        x_rnn = self.preprocess(x_rnn)
        x_rnn = nn.ReLU()(x_rnn)

        # Pass through RNN
        rnn_out, end_state = self.rnn(x_rnn)  # Output: (batch, time, hidden_dim)
        dummy_input = torch.zeros(batch_size * num_locations, self.prediction_length, 1)
        decoded, _ = self.decoder(dummy_input, end_state)
        decoded = self.output_decoder(decoded)
        decoded = nn.ReLU()(decoded)
        decoded = self.ouput_layer(decoded)
        return decoded.reshape(batch_size, num_locations, self.prediction_length, self.output_dim).swapaxes(1, 2)
        # return rnn_out.reshape(batch_size, num_locations, time_steps, self.hidden_dim).swapaxes(1, 2)


class FlatRNN(RNNWithLocationEmbedding):

    def forward(self, x, locations):
        offset_time = True
        batch_size, time_steps, feature_dim = x.shape
        total_length = self.prediction_length + time_steps - 1
        x_rnn = self._encode(locations, x)
        if self.direct_ar:
            x_rnn = torch.cat([x_rnn, x[..., -3:-1]], dim=-1)

        rnn_out, end_state = self.rnn(x_rnn)  # Output: (batch, time, hidden_dim)

        dummy_input = torch.zeros(batch_size, self.prediction_length - offset_time, 1)
        decoded, _ = self.decoder(dummy_input, end_state)

        if offset_time:
            decoded = torch.cat([rnn_out, decoded], dim=1)
        embedding = self.output_embedding(locations[..., 0])
        output_embedding = torch.concat([embedding[..., 1:, :],
                                         embedding[..., :self.prediction_length, :]], axis=-2)
        decoded = torch.cat([decoded, output_embedding], dim=-1)
        decoded = self.output_layer(decoded)

        reshaped = decoded.reshape(batch_size, total_length, self.output_dim)

        return (reshaped[:, -self.prediction_length:].squeeze(-1),
                reshaped[:, :-self.prediction_length].squeeze(-1))

    def _encode(self, locations, x):
        assert all(locations[..., i].max() < num_cat for i, num_cat in enumerate(self.num_categories)), \
            (
                [locations[..., i].max() for i in range(len(self.num_categories))], self.num_categories,
                locations[..., -1])
        loc_embeds = sum(embedding(locations[..., i]) for i, embedding in enumerate(self.location_embeddings))
        if self.embedding_type == 'sum':
            x_rnn = self.preprocess(x)
            x_rnn = x_rnn + loc_embeds
        else:
            x_rnn = torch.cat([x, loc_embeds], dim=-1)  # (batch, time, feature_dim + 4)
            x_rnn = self.preprocess(x_rnn)
        x_rnn = nn.ReLU()(x_rnn)
        return x_rnn




def main():
    batch_size, time_steps, num_locations, feature_dim = 8, 10, 5, 16
    num_locations_total = 100  # Total number of location indices
    hidden_dim = 32

    # Dummy data
    x = torch.randn(batch_size, time_steps, num_locations, feature_dim)
    locations = torch.randint(0, num_locations_total, (batch_size, time_steps, num_locations))

    # Initialize model
    model = RNNWithLocationEmbedding([num_locations_total], feature_dim, hidden_dim)

    # Forward pass
    model(x, locations)


def main_flat():
    batch_size, time_steps, _, feature_dim = 8, 10, 5, 16
    num_locations_total = 100  # Total number of location indices
    hidden_dim = 32

    # Dummy data
    x = torch.randn(batch_size, time_steps, feature_dim)
    locations = torch.randint(0, num_locations_total, (batch_size, time_steps))

    # Initialize model
    model = FlatRNN(num_locations_total, feature_dim, hidden_dim)

    # Forward pass
    output = model(x, locations)


if __name__ == '__main__':
    # Example usage
    main_flat()
