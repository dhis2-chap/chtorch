import torch
import torch.nn as nn


class RNNWithLocationEmbedding(nn.Module):
    def __init__(self, num_locations, input_feature_dim, hidden_dim, rnn_type='GRU', prediction_length=3):
        super().__init__()
        self.location_embedding = nn.Embedding(num_locations, 4)  # Embedding layer
        init_dim = input_feature_dim + 4
        self.hidden_dim = hidden_dim
        self.preprocess = nn.Linear(init_dim, hidden_dim)
        # Define RNN (GRU or LSTM)
        if rnn_type == 'GRU':
            self.rnn = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        elif rnn_type == 'LSTM':
            self.rnn = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        else:
            raise ValueError("Unsupported RNN type. Use 'GRU' or 'LSTM'.")

        self.decoder = nn.GRU(1, hidden_dim, batch_first=True)
        self.output_dim = 2
        self.output_decoder = nn.Linear(hidden_dim, hidden_dim)
        self.ouput_layer = nn.Linear(hidden_dim, self.output_dim)

        self.prediction_length = prediction_length

    def forward(self, x, locations):
        """
        x: (batch, time, location, features)
        locations: (batch, time, location) - location indices
        """
        batch_size, time_steps, num_locations, feature_dim = x.shape

        # Embed locations: (batch, time, location) -> (batch, time, location, 4)
        loc_embeds = self.location_embedding(locations)

        # Concatenate features with location embeddings
        x_with_loc = torch.cat([x, loc_embeds], dim=-1)  # (batch, time, location, feature_dim + 4)
        x_with_loc = x_with_loc.swapaxes(1, 2)  # (batch, location, time, feature_dim + 4
        # Reshape for RNN: merge location into feature dimension
        x_rnn = x_with_loc.reshape(batch_size*num_locations, time_steps, -1)  # (batch, time, location * (feature_dim + 4))
        x_rnn = self.preprocess(x_rnn)
        x_rnn = nn.ReLU()(x_rnn)
        # Pass through RNN
        rnn_out, end_state = self.rnn(x_rnn)  # Output: (batch, time, hidden_dim)
        dummy_input = torch.zeros(batch_size*num_locations, self.prediction_length, 1)
        decoded, _ = self.decoder(dummy_input, end_state)
        decoded = self.output_decoder(decoded)
        decoded = nn.ReLU()(decoded)
        decoded = self.ouput_layer(decoded)
        return decoded.reshape(batch_size, num_locations, self.prediction_length, self.output_dim).swapaxes(1, 2)
        #return rnn_out.reshape(batch_size, num_locations, time_steps, self.hidden_dim).swapaxes(1, 2)


class SeparatedRNNWithLocationEmbedding(nn.Module):
    def __init__(self, num_locations,
                 input_feature_dim,
                 hidden_dim,
                 prediction_length=3, n_ar_columns=4):
        super().__init__()
        embed_dim = 3
        self.location_embedding = nn.Embedding(num_locations, embed_dim)  # Embedding layer
        init_dim = input_feature_dim + embed_dim-n_ar_columns
        self.hidden_dim = hidden_dim
        self.preprocess = nn.Linear(init_dim, hidden_dim)
        self.rnn = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.decoder = nn.GRU(1, hidden_dim, batch_first=True)
        self.output_dim = 2
        self.output_decoder = nn.Linear(hidden_dim, hidden_dim)
        self.ouput_layer = nn.Linear(hidden_dim, self.output_dim)

        self.prediction_length = prediction_length

    def forward(self, x, locations):
        """
        x: (batch, time, location, features)
        locations: (batch, time, location) - location indices
        """
        batch_size, time_steps, num_locations, feature_dim = x.shape
        rate = x[..., -1]-x[..., -2]
        # Embed locations: (batch, time, location) -> (batch, time, location, 4)
        loc_embeds = self.location_embedding(locations)

        # Concatenate features with location embeddings
        x_with_loc = torch.cat([x, loc_embeds], dim=-1)  # (batch, time, location, feature_dim + 4)
        x_with_loc = x_with_loc.swapaxes(1, 2)  # (batch, location, time, feature_dim + 4
        # Reshape for RNN: merge location into feature dimension
        x_rnn = x_with_loc.reshape(batch_size*num_locations, time_steps, -1)  # (batch, time, location * (feature_dim + 4))
        x_rnn = self.preprocess(x_rnn)
        x_rnn = nn.ReLU()(x_rnn)
        # Pass through RNN
        rnn_out, end_state = self.rnn(x_rnn)  # Output: (batch, time, hidden_dim)
        dummy_input = torch.zeros(batch_size*num_locations, self.prediction_length, 1)
        decoded, _ = self.decoder(dummy_input, end_state)
        decoded = self.output_decoder(decoded)
        decoded = nn.ReLU()(decoded)
        decoded = self.ouput_layer(decoded)
        return decoded.reshape(batch_size, num_locations, self.prediction_length, self.output_dim).swapaxes(1, 2)
        #return rnn_out.reshape(batch_size, num_locations, time_steps, self.hidden_dim).swapaxes(1, 2)

if __name__ == '__main__':
    # Example usage
    batch_size, time_steps, num_locations, feature_dim = 8, 10, 5, 16
    num_locations_total = 100  # Total number of location indices
    hidden_dim = 32

    # Dummy data
    x = torch.randn(batch_size, time_steps, num_locations, feature_dim)
    locations = torch.randint(0, num_locations_total, (batch_size, time_steps, num_locations))

    # Initialize model
    model = RNNWithLocationEmbedding(num_locations_total, feature_dim, hidden_dim)

    # Forward pass
    output = model(x, locations)

    print("Output shape:", output.shape)  # (batch, time, hidden_dim)

