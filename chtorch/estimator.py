from typing import Any

import numpy as np
import torch
from chap_core.data import DataSet
from chap_core.datatypes import FullData, Samples
from sklearn.preprocessing import StandardScaler
from torch import nn, optim

from chtorch.count_transforms import Log1pTransform, CountTransform
from chtorch.data_loader import TSDataSet
from chtorch.module import RNNWithLocationEmbedding
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


class DeepARLightningModule(L.LightningModule):
    def __init__(self, module, loss, weight_decay=1e-6):
        super().__init__()
        self.module = module
        self.loss = loss
        self.weight_decay = weight_decay

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return self.module(*args, **kwargs)

    def training_step(self, batch, batch_idx):
        X, locations, y, population = batch
        log_rate = self.module(X, locations).squeeze(-1)
        loss = self.loss(log_rate, y, population)
        self.log("train_loss", loss, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        X, locations, y = batch
        log_rate = self.module(X, locations).squeeze(-1)
        loss = self.loss(log_rate, y)
        self.log("validation_loss", loss, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=1e-3, weight_decay=self.weight_decay)


def get_dist(eta, population, count_transform=Log1pTransform()):
    return torch.distributions.NegativeBinomial(
        total_count=count_transform.inverse(eta[..., 0], population)/torch.exp(eta[..., 1]),
        logits=eta[..., 1])


class Predictor:
    def __init__(self, module, tensorifier, transformer, context_length=12, prediction_length=3, count_transform=Log1pTransform()):
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
        ts_dataset = TSDataSet(historic_tensor, None, population, self.context_length, self.prediction_length)
        *instance, population = ts_dataset.last_prediction_instance()
        with torch.no_grad():
            eta = self.module(*instance)
        samples = get_dist(eta, population, self.count_transform).sample((100,))
        output = {}
        period_range = future_data.period_range

        for i, location in enumerate(historic_data.keys()):
            s = samples[:, 0, :, i].T
            output[location] = Samples(period_range, s)
        return DataSet(output)


class NegativeBinomialLoss(nn.Module):
    def __init__(self, count_transform: CountTransform = Log1pTransform()):
        super().__init__()
        self._count_transform = count_transform

    def forward(self, eta, y_true, population=0):
        """
        y_pred: (batch_size, 2)  - First column: mean (μ), Second column: dispersion (θ)
        y_true: (batch_size, 1)  - Observed counts
        """
        na_mask = ~torch.isnan(y_true)
        y_true = y_true[na_mask]
        eta = eta[na_mask]
        nb_dist = get_dist(eta, population, self._count_transform)
        loss = -nb_dist.log_prob(y_true).mean()
        return loss


class Estimator:
    features = ['rainfall', 'mean_temperature']
    count_transform = Log1pTransform()

    def __init__(self, context_length=12, prediction_length=3, debug=False, validate=False, weight_decay=1e-6, n_hidden=4, max_epochs=None):
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.debug = debug
        self.validate = validate
        self.max_epochs = max_epochs
        self.tensorifier = Tensorifier(self.features, self.count_transform)
        if max_epochs is None:
            self.max_epochs = 2500 // context_length

    def train(self, data: DataSet):
        array_dataset, population = self.tensorifier.convert(data)
        n_locations = array_dataset.shape[1]
        transformer = StandardScaler()
        transformed_dataset = transformer.fit_transform(array_dataset.reshape(-1, array_dataset.shape[-1]))
        X = transformed_dataset.reshape(array_dataset.shape).astype(np.float32)
        y = np.array([series.disease_cases for series in data.values()]).T
        train_dataset = TSDataSet(X, y, population, self.context_length, self.prediction_length)
        if self.validate:
            cutoff = int(len(train_dataset) * 0.8)
            val_dataset = torch.utils.data.Subset(train_dataset, range(cutoff, len(train_dataset)))
            train_dataset = torch.utils.data.Subset(train_dataset, range(cutoff))

        assert len(X) == len(y)

        loader = torch.utils.data.DataLoader(train_dataset, batch_size=5, shuffle=True, drop_last=True, num_workers=3)
        if self.validate:
            val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=5, shuffle=False, drop_last=True,
                                                     num_workers=3)
        module = RNNWithLocationEmbedding(n_locations, array_dataset.shape[-1], 4,
                                          prediction_length=self.prediction_length)
        lightning_module = DeepARLightningModule(
            module,
            NegativeBinomialLoss(count_transform=self.count_transform))
        trainer = L.Trainer(max_epochs=self.max_epochs if not self.debug else 3,
                            accelerator="cpu")  #"gpu" if torch.cuda.is_available() else "cpu")

        trainer.fit(lightning_module, loader, val_loader if self.validate else None)
        return Predictor(module, self.tensorifier, transformer, self.context_length, self.prediction_length, self.count_transform)


def test():
    dataset = DataSet.from_csv('/home/knut/Data/ch_data/rwanda_harmonized.csv', FullData)
    estimator = Estimator(context_length=12, prediction_length=3, validate=True)
    predictor = estimator.train(dataset)
