import dataclasses

import numpy as np
import torch
from chap_core.data import DataSet
from chap_core.datatypes import FullData, Samples
from chtorch.distribution_loss import NegativeBinomialLoss, get_dist
from chtorch.lightning_module import DeepARLightningModule
from sklearn.preprocessing import StandardScaler
from pydantic import BaseModel
from chtorch.count_transforms import Log1pTransform
from chtorch.data_loader import TSDataSet, FlatTSDataSet
from chtorch.module import RNNWithLocationEmbedding, FlatRNN
from chtorch.tensorifier import Tensorifier
import lightning as L

'''
mean =  total_count*exp(logits)
exp(logmean-logits) = total_count


variance = mean / sigmoid(-logits)
sigmoid(-logits) = mean / variance
logits = -logit(mean / variance)
logits = -log(mean/variance)

'''


class Predictor:
    is_flat = True

    def __init__(self, module, tensorifier, transformer, context_length=12, prediction_length=3, count_transform=None):
        super().__init__()
        self.module = module
        self.tensorifier = tensorifier
        self.transformer = transformer
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.count_transform = count_transform

    def predict(self, historic_data: DataSet, future_data: DataSet):
        historic_tensor, population = self.tensorifier.convert(historic_data)
        tmp = self.transformer.transform(historic_tensor.reshape(-1, historic_tensor.shape[-1]))
        historic_tensor = tmp.reshape(historic_tensor.shape).astype(np.float32)
        _DataSet = TSDataSet if not self.is_flat else FlatTSDataSet
        ts_dataset = _DataSet(historic_tensor, None, population, self.context_length, self.prediction_length)
        *instance, population = ts_dataset.last_prediction_instance()
        with torch.no_grad():
            eta = self.module(*instance)
        samples = get_dist(eta, population, self.count_transform).sample((100,))
        print(samples.shape, eta.shape, population.shape)
        output = {}
        period_range = future_data.period_range

        for i, location in enumerate(historic_data.keys()):
            if not self.is_flat:
                s = samples[:, 0, :, i].T
            else:
                s = samples[:, i, :].T
            output[location] = Samples(period_range, s)
        return DataSet(output)


class ModelConfiguration(BaseModel):
    weight_decay: float = 1e-6
    n_hidden: int = 4
    max_epochs: int | None = None
    context_length: int = 12
    embed_dim: int = 2
    num_rnn_layers: int = 1
    n_layers: int = 0


class ProblemConfiguration(BaseModel):
    prediction_length: int = 3
    replace_zeros: bool = False


model_config = ModelConfiguration(weight_decay=1e-6, n_hidden=4, max_epochs=200, context_length=12, embed_dim=2,
                                  num_rnn_layers=1, n_layers=0)
with open('model_config.json', 'w') as f:
    model_config.model_dump()


class Estimator:
    features = ['rainfall', 'mean_temperature']
    count_transform = Log1pTransform()
    is_flat = True

    def __init__(self, problem_configuration: ProblemConfiguration,
                 model_configuration: ModelConfiguration,
                 debug=False, validate=False):
        self.last_val_loss = None
        self.context_length = model_configuration.context_length
        self.prediction_length = problem_configuration.prediction_length
        self.debug = debug
        self.validate = validate
        self.max_epochs = model_configuration.max_epochs
        self.tensorifier = Tensorifier(self.features, self.count_transform, problem_configuration.replace_zeros)
        self.model_configuration = model_configuration

        if self.max_epochs is None:
            self.max_epochs = 2500 // self.context_length

    def train(self, data: DataSet):
        array_dataset, population = self.tensorifier.convert(data)
        n_locations = array_dataset.shape[1]
        transformer = StandardScaler()
        transformed_dataset = transformer.fit_transform(array_dataset.reshape(-1, array_dataset.shape[-1]))
        X = transformed_dataset.reshape(array_dataset.shape).astype(np.float32)
        y = np.array([series.disease_cases for series in data.values()]).T

        DataSet, Module = (TSDataSet, RNNWithLocationEmbedding) if (not self.is_flat) else (FlatTSDataSet, FlatRNN)
        train_dataset = DataSet(X, y, population, self.context_length, self.prediction_length)
        if self.validate:
            cutoff = int(len(train_dataset) * 0.8)
            val_dataset = torch.utils.data.Subset(train_dataset, range(cutoff, len(train_dataset)))
            train_dataset = torch.utils.data.Subset(train_dataset, range(cutoff))

        assert len(X) == len(y)

        batch_size = 64 if self.is_flat else 8
        loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True,
                                             num_workers=3)
        if self.validate:
            val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True,
                                                     num_workers=3)

        module = Module(n_locations, array_dataset.shape[-1],
                        hidden_dim=self.model_configuration.n_hidden,
                        prediction_length=self.prediction_length,
                        embed_dim=self.model_configuration.embed_dim,
                        num_rnn_layers=self.model_configuration.num_rnn_layers,
                        n_layers=self.model_configuration.n_layers)
        lightning_module = DeepARLightningModule(
            module,
            NegativeBinomialLoss(count_transform=self.count_transform))
        trainer = L.Trainer(max_epochs=self.max_epochs if not self.debug else 3,
                            accelerator="cpu")  # "gpu" if torch.cuda.is_available() else "cpu")

        trainer.fit(lightning_module, loader, val_loader if self.validate else None)
        self.last_val_loss = lightning_module.last_validation_loss
        print('####################', float(self.last_val_loss))
        return Predictor(module, self.tensorifier, transformer, self.context_length, self.prediction_length,
                         self.count_transform)


def test():
    dataset = DataSet.from_csv('/home/knut/Data/ch_data/rwanda_harmonized.csv', FullData)
    estimator = Estimator(context_length=12, prediction_length=3, validate=True)
    predictor = estimator.train(dataset)
