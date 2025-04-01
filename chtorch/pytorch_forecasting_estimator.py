import numpy as np
import torch
from chap_core.data import DataSet
from chap_core.datatypes import FullData
from pytorch_forecasting import TimeSeriesDataSet, BaseModel
from pydantic import BaseModel as PydanticModel


class PTFElement(PydanticModel, arbitrary_types_allowed=True):
    encoder_cont: torch.Tensor
    encoder_target: torch.Tensor
    encoder_lengths: torch.Tensor
    decoder_cat: torch.Tensor
    decoder_cont: torch.Tensor
    decoder_target: torch.Tensor
    decoder_lengths: torch.Tensor
    decoder_time_idx: torch.Tensor
    groups: torch.Tensor
    target_scale: torch.Tensor


class MyModel(BaseModel):
    def __init__(self, input_size: int, **kwargs):
        self.save_hyperparameters()
        super().__init__(**kwargs)
        self.rnn = torch.nn.RNN(input_size=input_size, hidden_size=2, batch_first=True)
        self.linear = torch.nn.Linear(2, 1)

    def forward(self, x):
        # x is a batch generated based on the TimeSeriesDataset, here we just use the
        # continuous variables for the encoder
        network_input = x["encoder_cont"]
        _, hidden= self.rnn(network_input)
        prediction = self.linear(hidden)
        prediction = self.transform_output(prediction, target_scale=x["target_scale"])
        return self.to_network_output(prediction=prediction)

class PyTorchForecastingEstimator:
    features = ['mean_temperature', 'rainfall']

    def __init__(self, context_length, prediction_length, validate=True):
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.validate = validate

    def train(self, data: DataSet):
        n_periods = len(data.period_range)
        df = data.to_pandas()
        df['time_idx'] = np.arange(len(df)) % n_periods
        df.reset_index(inplace=True)
        training = TimeSeriesDataSet(
            df, target='disease_cases', time_idx='time_idx',
            group_ids=['location'],
            static_categoricals=['location'],
            time_varying_unknown_reals=['rainfall', 'mean_temperature', 'disease_cases'],
            min_encoder_length=self.context_length,
            max_encoder_length=self.context_length,
            min_prediction_length=self.prediction_length,
            max_prediction_length=self.prediction_length)
        model = MyModel.from_dataset(training, input_size=3)
        data_loader = training.to_dataloader(batch_size=32, num_workers=0)
        batch = next(iter(data_loader))
        x, y = batch
        out = model(x)
        print(out)
        # e = PTFElement(**x)



def test():
    dataset = DataSet.from_csv('/home/knut/Data/ch_data/rwanda_harmonized.csv', FullData)
    estimator = PyTorchForecastingEstimator(context_length=12, prediction_length=3, validate=True)
    predictor = estimator.train(dataset)
