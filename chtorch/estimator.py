from typing import Any

import numpy as np
import torch
from chap_core.data import DataSet
from chap_core.datatypes import FullData
from sklearn.preprocessing import StandardScaler
from torch import nn, optim

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
        print(loss)
        return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=1e-3)


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
        lightning_module = DeepARLightningModule(module, nn.PoissonNLLLoss(log_input=True))
        trainer = L.Trainer(max_epochs=1000,
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

        return module


def test():
    dataset = DataSet.from_csv('~/Data/ch_data/rwanda_harmonized.csv', FullData)
    estimator = Estimator()
    predictor = estimator.train(dataset)
