import torch.nn as nn


class SpatioTemporalModel(nn.Module):
    def __init__(self, input_features, dense_hidden_size, conv_channels, kernel_size, rnn_hidden_size, rnn_type="LSTM"):
        super(SpatioTemporalModel, self).__init__()

        # Dense layer for features
        self.feature_dense = nn.Linear(input_features, dense_hidden_size)

        # Convolutional layers for spatial dimensions
        self.conv = nn.Sequential(
            nn.Conv2d(dense_hidden_size, conv_channels, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.ReLU(),
            nn.Conv2d(conv_channels, conv_channels, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.ReLU()
        )

        # RNN for time dimension
        rnn_class = nn.LSTM if rnn_type == "LSTM" else nn.GRU
        X, Y = 32, 32  # Example spatial dimensions
        self.rnn = rnn_class(input_size=conv_channels * X * Y, hidden_size=rnn_hidden_size, batch_first=True)

        # Fully connected layer for final output
        self.fc = nn.Linear(rnn_hidden_size, 1)  # Example output size: 1

    def forward(self, x):
        # x shape: (batch, time, X, Y, features)
        batch, time, X, Y, features = x.shape

        # Process features with dense layer
        x = x.view(batch * time * X * Y, features)  # Flatten for dense layer
        x = self.feature_dense(x)
        x = x.view(batch * time, -1, X, Y)  # Reshape for convolution

        # Apply convolutional layers
        x = self.conv(x)  # Shape: (batch*time, conv_channels, X, Y)

        # Flatten spatial dimensions for RNN
        x = x.view(batch, time, -1)  # Shape: (batch, time, conv_channels * X * Y)

        # Apply RNN
        x, _ = self.rnn(x)  # Shape: (batch, time, rnn_hidden_size)

        # Final output layer (e.g., regression or classification)
        x = self.fc(x[:, -1, :])  # Use the last time step's output

        return x
