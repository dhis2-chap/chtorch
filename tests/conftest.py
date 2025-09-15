from math import prod
from pathlib import Path

import numpy as np
import torch
from chap_core.data.datasets import ISIMIP_dengue_harmonized
from chap_core.assessment.dataset_splitting import (
    train_test_generator,
)

from gluonts.dataset.repository import get_dataset
import pytest

from chtorch.configuration import ModelConfiguration
from chtorch.data_loader import Entry


@pytest.fixture()
def electricity_dataset():
    return get_dataset("electricity")


@pytest.fixture
def ch_dataset():
    return ISIMIP_dengue_harmonized['vietnam']

@pytest.fixture
def ch_dataset_brazil():
    return ISIMIP_dengue_harmonized['brazil']

@pytest.fixture
def ch_dataset_laos():
    return ISIMIP_dengue_harmonized['laos']



@pytest.fixture()
def model_configuration():
    return ModelConfiguration(context_length=12)


@pytest.fixture
def train_test(ch_dataset):
    train, test_generator = train_test_generator(ch_dataset, 3, 1)
    return train, next(test_generator)


@pytest.fixture
def data_path():
    return Path('~/Data/ch_data/full_data/')


@pytest.fixture
def auxilliary_datasets(data_path):
    return {
        country_name: ISIMIP_dengue_harmonized[country_name]
        for country_name in ['brazil', 'thailand']}


@pytest.fixture()
def entry():
    random_fun = lambda shape: np.random.rand(prod(shape)).reshape(shape)
    X = random_fun((4, 3, 5))
    y = random_fun((2, 3))
    past_y = random_fun((4, 3))
    locations = random_fun((4, 3))
    population = random_fun((4, 3))
    return Entry(X, locations, y, population, past_y=y[0:1])

