import numpy as np
import pytest

from chtorch.count_transforms import Log1pTransform
from chtorch.data_loader import FlatTSDataSet, TSDataSet
from chtorch.tensorifier import Tensorifier


def test_data_loader():
    ...


@pytest.fixture
def flat_dataset(ch_dataset):
    tensorifier = Tensorifier(['rainfall', 'mean_temperature'], Log1pTransform())
    X, population, *_ = tensorifier.convert(ch_dataset)
    y = np.array([series.disease_cases for series in ch_dataset.values()]).T
    dataset = FlatTSDataSet(X, y, population, 12, 3)
    return dataset


@pytest.fixture()
def ts_dataset(ch_dataset):
    tensorifier = Tensorifier(['rainfall', 'mean_temperature'], Log1pTransform())
    X, population, *_ = tensorifier.convert(ch_dataset)
    y = np.array([series.disease_cases for series in ch_dataset.values()]).T
    dataset = TSDataSet(X, y, population, 12, 3)
    return dataset


def test_getitem(ts_dataset):
    X, location, y, population = ts_dataset[0]
    assert X.shape == (12, 19, 7)
    assert location.shape == (12, 19)
    assert y.shape == (3, 19)
    assert population.shape == (3, 19)

def test_last_prediction(ts_dataset):
    X, location, population = ts_dataset.last_prediction_instance()
    assert X.shape == (1, 12, 19, 7)
    assert location.shape == (1, 12, 19)
    assert population.shape == (1, 3, 19)


def test_getitem_flat(flat_dataset):
    X, location, y, population = flat_dataset[0]
    assert X.shape == (12, 7)
    assert y.shape == (3,)
    assert location.shape == (12,)
    assert population.shape == (3,)


def test_last_prediction_flat(flat_dataset):
    X, location, population = flat_dataset.last_prediction_instance()
    n_location = 19
    assert X.shape == (n_location, 12, 7)
    assert location.shape == (n_location, 12)
    assert population.shape == (n_location, 3)
    print(location, population)
