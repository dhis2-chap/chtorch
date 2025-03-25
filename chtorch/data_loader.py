# import torch
# from pytorch_forecasting import Baseline, DeepAR, TimeSeriesDataSet
from array import ArrayType

import numpy as np
import torch
from pytorch_forecasting import TimeSeriesDataSet


class TSDataSet(torch.utils.data.Dataset):
    def __init__(self, X, y, context_length, prediction_length):
        self.X = X
        self.y = y
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
        return x, self.locations, y

    def last_prediction_instance(self):
        return torch.from_numpy(self.X[None, -self.context_length:, ...]), torch.from_numpy(self.locations[None, ...])

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
