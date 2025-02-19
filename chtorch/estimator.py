import numpy as np
from chap_core.data import DataSet
from chap_core.datatypes import FullData
from sklearn.preprocessing import StandardScaler

from chtorch.data_loader import DataLoader
from chtorch.tensorifier import Tensorifier


class Estimator:
    tensorifier = Tensorifier(['rainfall', 'mean_temperature', 'population'])
    loader = DataLoader

    def train(self, data: DataSet):
        array_dataset = self.tensorifier.convert(data)
        transormer = StandardScaler()
        transformed_dataset = transormer.fit_transform(array_dataset.reshape(-1, array_dataset.shape[-1]))
        X = transformed_dataset.reshape(array_dataset.shape)
        y = np.array([series.disease_cases for series in data.values()]).T
        assert len(X) == len(y)
        loader = self.loader(X, y, 12, 3, 5)
        for X, y in loader:
            assert X.shape[:2] == (5, 12)
            assert y.shape[:2] == (5, 3)


def test():
    dataset = DataSet.from_csv('~/Data/ch_data/rwanda_harmonized.csv', FullData)
    estimator = Estimator()
    predictor = estimator.train(dataset)
