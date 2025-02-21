from typing import Any

import numpy as np
import torch
from chap_core.assessment.dataset_splitting import train_test_generator
from chap_core.data import DataSet
from chap_core.datatypes import FullData, Samples
from sklearn.preprocessing import StandardScaler
from torch import nn, optim
from torch.distributions import NegativeBinomial

from chtorch import tensorifier
from chtorch.data_loader import DataLoader, TSDataSet
from chtorch.module import RNNWithLocationEmbedding
from chtorch.tensorifier import Tensorifier
import lightning as L


class DeepARLightningModule(L.LightningModule):
    def __init__(self, module, loss):
        super().__init__()
        self.module = module
        self.loss = loss

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return self.module(*args, **kwargs)

    def training_step(self, batch, batch_idx):
        X, locations, y = batch
        log_rate = self.module(X, locations).squeeze(-1)
        loss = self.loss(log_rate, y)
        self.log("train_loss", loss, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=1e-3)


class Predictor:
    def __init__(self, module, tensorifier, transformer):
        super().__init__()
        self.module = module
        self.tensorifier = tensorifier
        self.transformer = transformer

    def forecast(self, historic_data: DataSet, future_data: DataSet):
        historic_tensor = self.tensorifier.convert(historic_data)
        print(historic_tensor.shape)
        tmp = self.transformer.transform(historic_tensor.reshape(-1, historic_tensor.shape[-1]))
        historic_tensor = tmp.reshape(historic_tensor.shape).astype(np.float32)
        ts_dataset = TSDataSet(historic_tensor, None, 12, 3)
        instance = ts_dataset.last_prediction_instance()
        with torch.no_grad():
            eta = self.module(*instance)
        samples = torch.distributions.NegativeBinomial(
            total_count=torch.exp(eta[..., 0]),
            logits=eta[..., 1]).sample((100,))
        output = {}
        period_range = future_data.period_range
        print(period_range)

        for i, location in enumerate(historic_data.keys()):
            s = samples[..., i].T
            print(s.shape)
            output[location] = Samples(period_range, s)
        return DataSet(output)



class NegativeBinomialLoss(nn.Module):
    def forward(self, y_pred, y_true):
        """
        y_pred: (batch_size, 2)  - First column: mean (μ), Second column: dispersion (θ)
        y_true: (batch_size, 1)  - Observed counts
        """
        K = torch.exp(y_pred[..., 0]).ravel()  # Ensure mean (μ) is positive
        eta = y_pred[..., 1].ravel()  # Ensure dispersion (θ) is positive

        # Define Negative Binomial distribution
        nb_dist = NegativeBinomial(total_count=K, logits=eta)

        # Compute negative log likelihood loss
        loss = -nb_dist.log_prob(y_true.ravel()).mean()
        return loss


class Estimator:
    tensorifier = Tensorifier(['rainfall', 'mean_temperature', 'population'])
    loader = DataLoader

    def train(self, data: DataSet):
        array_dataset = self.tensorifier.convert(data)
        n_locations = array_dataset.shape[1]
        transormer = StandardScaler()
        transformed_dataset = transormer.fit_transform(array_dataset.reshape(-1, array_dataset.shape[-1]))
        X = transformed_dataset.reshape(array_dataset.shape).astype(np.float32)
        y = np.array([series.disease_cases for series in data.values()]).T
        ts_dataset = TSDataSet(X, y, 12, 3)
        assert len(X) == len(y)
        loader = torch.utils.data.DataLoader(ts_dataset, batch_size=5, shuffle=True, drop_last=True)
        module = RNNWithLocationEmbedding(n_locations, array_dataset.shape[-1], 4)
        lightning_module = DeepARLightningModule(module,
                                                 NegativeBinomialLoss())
        trainer = L.Trainer(max_epochs=10,
                            accelerator="gpu" if torch.cuda.is_available() else "cpu")

        trainer.fit(lightning_module, loader)
        # locations = np.array([[np.arange(n_locations) for _ in range(12)] for _ in range(5)])
        # Alocations = torch.from_numpy(locations)
        # for X, locations, y in loader:
        #     lightning_module.forward(X, locations)
        #     lightning_module.training_step((X, locations, y), 0)
        #     #assert X.shape[:2] == (5, 12), X.shape
        #     #assert y.shape[:2] == (5, 3), y.shape
        #     #log_rate = module(X, locations)
        #     #assert log_rate.shape == (5, 3, n_locations, 1)
        #     #loss = nn.PoissonNLLLoss(log_input=True)(log_rate.reshape(5, 3, n_locations), y)

        return Predictor(module, self.tensorifier, transormer)


def test():
    dataset = DataSet.from_csv('~/Data/ch_data/rwanda_harmonized.csv', FullData)
    train, test = train_test_generator(dataset, prediction_length=3, n_test_sets=1)
    historic, future, _ = next(test)
    estimator = Estimator()
    predictor = estimator.train(train)
    samples =predictor.forecast(historic, future)
    print(samples)
