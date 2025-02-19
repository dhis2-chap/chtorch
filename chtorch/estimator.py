import numpy as np
import torch
from chap_core.data import DataSet
from chap_core.datatypes import FullData
from sklearn.preprocessing import StandardScaler
from torch import nn

from chtorch.data_loader import DataLoader
from chtorch.module import RNNWithLocationEmbedding
from chtorch.tensorifier import Tensorifier


class Estimator:
    tensorifier = Tensorifier(['rainfall', 'mean_temperature', 'population'])
    loader = DataLoader

    def train(self, data: DataSet):
        array_dataset = self.tensorifier.convert(data)
        n_locations = array_dataset.shape[1]
        transormer = StandardScaler()
        transformed_dataset = transormer.fit_transform(array_dataset.reshape(-1, array_dataset.shape[-1]))
        X = transformed_dataset.reshape(array_dataset.shape).astype(np.float32)
        y = np.array([series.disease_cases for series in data.values()]).T
        assert len(X) == len(y)
        loader = self.loader(X, y, 12, 3, 5)
        module = RNNWithLocationEmbedding(n_locations, array_dataset.shape[-1], 4)
        locations = np.array([[np.arange(n_locations) for _ in range(12)] for _ in range(5)])
        locations = torch.from_numpy(locations)
        for X, y in loader:
            assert X.shape[:2] == (5, 12)
            assert y.shape[:2] == (5, 3)
            log_rate = module(torch.from_numpy(X), locations)
            assert log_rate.shape == (5, 3, n_locations, 1)
            loss = nn.PoissonNLLLoss(log_input=True)(log_rate.reshape(5, 3, n_locations), torch.from_numpy(y))
            print(loss)

def test():
    dataset = DataSet.from_csv('~/Data/ch_data/rwanda_harmonized.csv', FullData)
    estimator = Estimator()
    predictor = estimator.train(dataset)
