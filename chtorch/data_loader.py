# import torch
# from pytorch_forecasting import Baseline, DeepAR, TimeSeriesDataSet
from array import ArrayType

import numpy as np
import torch
from pytorch_forecasting import TimeSeriesDataSet


class TSDataSet(torch.utils.data.Dataset):
    def __init__(self, X, y, population, context_length, prediction_length):
        if y is not None:
            assert y.shape == population.shape, f"y and population should have the same shape, got {y.shape} and {population.shape}"
        self.X = X  # time, location, feature
        self.y = y
        self.population = population
        self.total_length = context_length + prediction_length
        self.context_length = context_length
        self.prediction_length = prediction_length
        n_locations = X.shape[1]
        self.locations = np.array([np.arange(n_locations) for _ in range(context_length)])

    def __len__(self):
        return len(self.X) - self.total_length + 1

    def __getitem__(self, i) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        x = self.X[i:i + self.context_length]
        y = self.y[i + self.context_length:i + self.total_length]
        population = self.population[i + self.context_length:i + self.total_length]
        assert y.shape == population.shape, f"y and population should have the same shape, got {y.shape} and {population.shape}"
        return x, self.locations, y, population

    def last_prediction_instance(self):
        last_population = self.population[-1]
        repeated_population = np.array([last_population for _ in range(self.prediction_length)])
        return (torch.from_numpy(self.X[None, -self.context_length:, ...]),
                torch.from_numpy(self.locations[None, ...]),
                torch.from_numpy(repeated_population[None, ...]))


class FlatTSDataSet(TSDataSet):
    def __len__(self):
        return (len(self.X) - self.total_length + 1) * self.X.shape[1]

    def __getitem__(self, item):
        i, j = divmod(item, self.X.shape[1])
        x = self.X[i:i + self.context_length, j]
        y = self.y[i + self.context_length:i + self.total_length, j]
        population = self.population[i + self.context_length:i + self.total_length, j]
        assert y.shape == population.shape, f"y and population should have the same shape, got {y.shape} and {population.shape}"
        return x, np.full(self.context_length, j), y, population

    def last_prediction_instance(self):
        last_population = self.population[-1:].T
        repeated_population = np.repeat(last_population, self.prediction_length, axis=1)
        location = np.array([self.locations[0] for t in range(self.context_length)]).T
        return (torch.from_numpy(self.X[-self.context_length:, ...].swapaxes(0, 1)),
                torch.from_numpy(location),
                torch.from_numpy(repeated_population))


#
# class DataLoader:
#     def __init__(self, X, y, context_length, prediction_length, batch_size):
#         self.X = X
#         self.y = y
#         self.context_length = context_length
#         self.prediction_length = prediction_length
#         self.total_length = context_length + prediction_length + batch_size - 1
#         self.batch_size = batch_size
#
#     def __iter__(self):
#         for i in range(len(self.X) - self.total_length + 1):
#             x = np.array([self.X[j:j + self.context_length] for j in range(i, i + self.batch_size)])
#             y = np.array([self.y[j + self.context_length: j + self.context_length + self.prediction_length]
#                           for j in range(i, i + self.batch_size)])
#             yield x, y
