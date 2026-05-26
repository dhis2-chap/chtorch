"""Unit-level tests for the AuxilliaryEstimator data plumbing.

These exercise the path that previously crashed because of three separate
bugs in _get_transformed_dataset: a mismatched signature with the parent,
an unpacking-arity mismatch, and an off-by-one index when collecting the
per-dataset target scalers.
"""
import pytest

from chtorch.auxilliary_estimator import AuxilliaryEstimator
from chtorch.configuration import ModelConfiguration, ProblemConfiguration
from chtorch.data_loader import MultiDataset
from chtorch.target_scaler import MultiTargetScaler


@pytest.fixture
def auxilliary_datasets(ch_dataset):
    # Reuse the same dataset twice — the contents don't matter for these
    # tests, only the plumbing.
    return {'aux_a': ch_dataset, 'aux_b': ch_dataset}


def _make_estimator(auxilliary_datasets):
    model_cfg = ModelConfiguration(context_length=12, embed_dim=2, n_hidden=2,
                                   state_dim=2, max_dim=8)
    prob_cfg = ProblemConfiguration(prediction_length=3, debug=True)
    return AuxilliaryEstimator(
        problem_configuration=prob_cfg,
        model_configuration=model_cfg,
        auxilliary_datasets=auxilliary_datasets,
    )


def test_aux_get_transformed_dataset_returns_four_values(ch_dataset, auxilliary_datasets):
    estimator = _make_estimator(auxilliary_datasets)
    train, transformer, target_scaler, val = estimator._get_transformed_dataset(ch_dataset)
    assert isinstance(train, MultiDataset)
    assert isinstance(target_scaler, MultiTargetScaler)
    assert val is None  # no validation_dataset passed


def test_aux_get_transformed_dataset_accepts_validation_dataset(ch_dataset, auxilliary_datasets):
    """Regression: the override previously didn't accept validation_dataset
    and crashed when Estimator.train called it with self._validation_dataset."""
    estimator = _make_estimator(auxilliary_datasets)
    # Just call with None — exercises the kwarg path. The contents of a real
    # validation set aren't important for this signature test.
    train, transformer, target_scaler, val = estimator._get_transformed_dataset(
        ch_dataset, validation_dataset=None
    )
    assert val is None


def test_aux_multidataset_iterates_without_crash(ch_dataset, auxilliary_datasets):
    """Regression: MultiDataset.__getitem__ used to drop past_y and crash."""
    estimator = _make_estimator(auxilliary_datasets)
    train, *_ = estimator._get_transformed_dataset(ch_dataset)
    item = train[0]
    assert item.X is not None
    assert item.past_y is not None
