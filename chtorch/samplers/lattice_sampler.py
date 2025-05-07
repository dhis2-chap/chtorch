'''
This is a hyperparameter sampler for the case where every hyperparameter has a designated role where increasing
the hp increases both the capacity and the train/validation gap (i.e. num_layers, num_features, -dropout, 1/l2).
The goal is to fit a multidimensional monotonically increasing function to both the capacity (- train loss) and gap
(val_loss-train_loss), and use that function to suggest where the biggest improvements might be made.
In the end we might choose either the one with the best validation loss, or the one with the best validation loss according to the approximated functions
'''
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from chap_core.datatypes import FullData
from chap_core.spatio_temporal_data.temporal_dataclass import DataSet
import plotly.express as px

from chtorch.estimator import Estimator
from chtorch.configuration import ModelConfiguration, ProblemConfiguration


class MonotonicSampler:
    ...


class MonotonicNN(nn.Module):
    def __init__(self, input_dim, hidden_dim=16):
        super().__init__()
        self.input_dim = input_dim

        # Constrain weights using softplus to ensure non-negativity
        self.raw_weights = nn.Parameter(torch.randn(hidden_dim, input_dim))
        self.bias = nn.Parameter(torch.zeros(hidden_dim))
        self.raw_out_weights = nn.Parameter(torch.randn(1, hidden_dim))
        self.out_bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        W1 = F.softplus(self.raw_weights)
        W2 = F.softplus(self.raw_out_weights)
        x = F.relu(F.linear(x, W1, self.bias))  # Monotonic hidden layer
        x = F.linear(x, W2, self.out_bias)  # Monotonic output layer
        return x.squeeze(-1)


def main():
    model_config = ModelConfiguration(
        max_epochs=20,
        output_embedding_dim=1,
        dropout=0.2,
        embedding_type='sum'
    )
    problem_config = ProblemConfiguration(
        replace_zeros=True,
        validate=True,
        validataion_splits=5,
        validation_index=4
    )
    dataset = DataSet.from_csv('/home/knut/Sources/climate_health/example_data/vietnam_monthly.csv', FullData)
    f = open('hpo_loss.csv', 'w')
    for n_hidden in range(3, 8):
        for w_factor in np.linspace(1, 6, 10):
            for dropout in np.linspace(0, 0.5, 10):
                model_config.n_hidden = n_hidden
                model_config.embed_dim = n_hidden
                model_config.weight_decay = 10 ** (-w_factor)
                model_config.dropout = dropout
                estimator = Estimator(problem_config, model_config)
                _ = estimator.train(dataset)
                val_loss = estimator.last_val_loss
                train_loss = estimator.last_train_loss
                row = ','.join(str(c) for c in (n_hidden, float(w_factor), float(dropout), float(val_loss), float(train_loss)))
                print(row)
                f.write(row + '\n')



def plot():
    df = pd.read_csv('hpo_loss.csv', header=None)
    print(df)
    for d, pic in df.groupby(2):
        rows = []
        vals = []
        for _, row in pic.groupby(0):
            print(row)
            rows.append(row[3].tolist())
            vals.append(row[4].tolist())
        print(rows)
        rows = np.array(rows)

        vals = np.array(vals)
        px.imshow(rows, title=f'train{d}').show()
        px.imshow(vals, title=f'val{d}').show()
        px.imshow(vals - rows, title=f'gap{d}').show()

if __name__ == '__main__':
    plot()
    #main()
