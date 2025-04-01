from chap_core.data.datasets import ISIMIP_dengue_harmonized
from gluonts.dataset.repository import get_dataset
import pytest


@pytest.fixture()
def electricity_dataset():
    return get_dataset("electricity")


@pytest.fixture
def ch_dataset():
    return ISIMIP_dengue_harmonized['vietnam']

