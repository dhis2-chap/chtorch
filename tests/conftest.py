from chap_core.data.datasets import ISIMIP_dengue_harmonized
import pytest


@pytest.fixture
def ch_dataset():
    return ISIMIP_dengue_harmonized['vietnam']
