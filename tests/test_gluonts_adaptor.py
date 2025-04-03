from gluonts.dataset.field_names import FieldName
from gluonts.transform import InstanceSplitter, ExpectedNumInstanceSampler, AddObservedValuesIndicator
from gluonts.dataset.loader import TrainDataLoader
import pytest
from chtorch.gluonts_adaptor import GluonTSAdaptor


def test_to_gluonts(ch_dataset):
    adapter = GluonTSAdaptor()
    for series in adapter.to_gluonts(ch_dataset):
        assert 'start' in series


training_splitter = InstanceSplitter(
    target_field=FieldName.TARGET,
    is_pad_field=FieldName.IS_PAD,
    start_field=FieldName.START,
    forecast_start_field=FieldName.FORECAST_START,
    instance_sampler=ExpectedNumInstanceSampler(
        num_instances=1,
        min_future=12,
    ),
    past_length=20,
    future_length=12,
    time_series_fields=[FieldName.OBSERVED_VALUES, FieldName.FEAT_DYNAMIC_REAL],
)
mask_unobserved = AddObservedValuesIndicator(
    target_field=FieldName.TARGET,
    output_field=FieldName.OBSERVED_VALUES,
)


def test_train_dataset(ch_dataset):
    adapter = GluonTSAdaptor()
    dataset = adapter.to_gluonts(ch_dataset)
    loader = TrainDataLoader(
        dataset=dataset,
        transform=mask_unobserved + training_splitter,
        batch_size=2,
        stack_fn=lambda x: x,
    )
    first_batch = next(iter(loader))
    assert len(first_batch) == 2

@pytest.mark.skip
def test_splitter(electricity_dataset):
    loader = TrainDataLoader(
        dataset=electricity_dataset.train,
        transform=mask_unobserved + training_splitter,
        batch_size=2,
        stack_fn=lambda x: x,
    )

