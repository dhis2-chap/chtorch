import numpy as np
import pytest

from chtorch.count_transforms import Log1pTransform
from chtorch.data_loader import Entry, FlatTSDataSet, MultiDataset, TSDataSet
from chtorch.tensorifier import Tensorifier


def test_data_loader():
    ...

@pytest.fixture()
def tensorifier(model_configuration):
    return Tensorifier(
        Log1pTransform(),
        config=model_configuration,
    )

@pytest.fixture
def flat_dataset(ch_dataset, tensorifier):
    X, population, parents = tensorifier.convert(ch_dataset)
    y = np.array([series.disease_cases for series in ch_dataset.values()]).T
    dataset = FlatTSDataSet(X, y, population, 12, 3, parents=parents)
    return dataset


@pytest.fixture()
def ts_dataset(ch_dataset, tensorifier):
    X, population, *_ = tensorifier.convert(ch_dataset)
    y = np.array([series.disease_cases for series in ch_dataset.values()]).T
    dataset = TSDataSet(X, y, population, 12, 3)
    return dataset


def test_getitem(ts_dataset):
    batch = ts_dataset[0]
    assert batch.X.shape == (12, 19, 8)
    assert batch.locations.shape == (12, 19, 1)
    assert batch.y.shape == (3, 19)
    assert batch.population.shape == (3, 19)


def test_last_prediction(ts_dataset):
    X, location, population = ts_dataset.last_prediction_instance()
    assert X.shape == (1, 12, 19, 8)
    assert location.shape == (1, 12, 19, 1)
    assert population.shape == (1, 3, 19)


def test_getitem_flat(flat_dataset):
    batch = flat_dataset[0]
    assert batch.X.shape == (12, 8)
    assert batch.y.shape == (3,)
    assert batch.locations.shape == (12, 2)
    assert batch.population.shape == (3,)


def test_last_prediction_flat(flat_dataset):
    batch = flat_dataset.last_prediction_instance()
    n_location = 19
    assert batch.X.shape == (n_location, 12, 8)
    assert batch.locations.shape == (n_location, 12, 2)
    assert batch.population.shape == (n_location, 3)


def test_multidataset_getitem_returns_entry(ch_dataset, tensorifier):
    X, population, parents = tensorifier.convert(ch_dataset)
    y = np.array([series.disease_cases for series in ch_dataset.values()]).T
    ds_a = FlatTSDataSet(X, y, population, 12, 3, parents=parents)
    ds_b = FlatTSDataSet(X, y, population, 12, 3, parents=parents)
    multi = MultiDataset([ds_a, ds_b])
    item = multi[0]
    assert isinstance(item, Entry)
    assert item.X.shape == (12, 8)
    assert item.locations.shape == (12, 2)
    assert item.y.shape == (3,)
    assert item.population.shape == (3,)
    assert item.past_y.shape == (12,)


def test_multidataset_remaps_categories(ch_dataset, tensorifier):
    X, population, parents = tensorifier.convert(ch_dataset)
    y = np.array([series.disease_cases for series in ch_dataset.values()]).T
    ds_a = FlatTSDataSet(X, y, population, 12, 3, parents=parents)
    ds_b = FlatTSDataSet(X, y, population, 12, 3, parents=parents)
    multi = MultiDataset([ds_a, ds_b])

    # An item drawn from the second dataset should have locations offset by
    # the first dataset's category count, and dataset_idx in column 1.
    first_dataset_len = len(ds_a)
    item = multi[first_dataset_len]
    assert (item.locations[:, 1] == 1).all()
    assert item.locations[:, 0].min() >= ds_a.n_categories[0]
