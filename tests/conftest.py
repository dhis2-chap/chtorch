from pathlib import Path

from chap_core.data.datasets import ISIMIP_dengue_harmonized
from chap_core.data import DataSet
from gluonts.dataset.repository import get_dataset
import pytest


@pytest.fixture()
def electricity_dataset():
    return get_dataset("electricity")


@pytest.fixture
def ch_dataset():
    return ISIMIP_dengue_harmonized['vietnam']


@pytest.fixture
def data_path():
    return Path('~/Data/ch_data/full_data/')


@pytest.fixture
def auxilliary_datasets(data_path):
    print(ISIMIP_dengue_harmonized.items())
    return {
        country_name: ISIMIP_dengue_harmonized[country_name]
        for country_name in ['vietnam']}
