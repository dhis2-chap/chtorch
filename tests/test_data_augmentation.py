import pytest

from chtorch.data_augmentation import Augmentation, PoissonAugmentation, MaskingAugmentation
from chtorch.data_loader import Entry


def sanity_check_agumentation(augmentation: Augmentation, entry: Entry):
    new_entry = augmentation.transform(entry)
    for elem, new_elem in zip(entry, new_entry):
        if elem is None:
            assert new_elem is None, f"Expected None but got {new_elem}"
            continue
        assert elem.shape == new_elem.shape, f"Shape mismatch: {elem.shape} vs {new_elem.shape}"


@pytest.mark.parametrize('augmentation', [PoissonAugmentation(), MaskingAugmentation()])
def test_augmentation_sanity(entry, augmentation):
    sanity_check_agumentation(augmentation, entry)
