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

    def __len__(self):
        return len(self.X) - self.total_length + 1

    def __getitem__(self, i):
        x = self.X[i:i + self.context_length]
        y = self.y[i + self.context_length:i + self.total_length]
        return x, y


class DataLoader:
    def __init__(self, X, y, context_length, prediction_length, batch_size):
        self.X = X
        self.y = y
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.total_length = context_length + prediction_length + batch_size - 1
        self.batch_size = batch_size

    def __iter__(self):
        for i in range(len(self.X) - self.total_length + 1):
            x = np.array([self.X[j:j + self.context_length] for j in range(i, i + self.batch_size)])
            y = np.array([self.y[j + self.context_length: j + self.context_length + self.prediction_length]
                          for j in range(i, i + self.batch_size)])
            yield x, y


if False:
    class Instance(BaseModel):
        historic_data: ArrayType
        target_values: ArrayType
        future_predictors: ArrayType | None = None


    class Adaptor:
        def __init__(self, ch_dataset):
            self._ch_dataset = ch_dataset
            df = ch_dataset.to_pandas()
            dataset = TimeSeriesDataSet(
                df,
                group_ids=["location"],
                target="disease_cases",
                time_idx="time_period",
                min_encoder_length=0,
                max_encoder_length=12,
                min_prediction_length=3,
                max_prediction_length=3,
                time_varying_unknown_reals=["value"],
            )
