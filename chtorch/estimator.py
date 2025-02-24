from typing import Any

import numpy as np
import torch
from chap_core.assessment.prediction_evaluator import evaluate_model
from chap_core.climate_predictor import QuickForecastFetcher
from chap_core.data import DataSet
from chap_core.datatypes import FullData, Samples
from sklearn.preprocessing import StandardScaler
from torch import nn, optim
from torch.distributions import NegativeBinomial

from chtorch.data_loader import DataLoader, TSDataSet
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
        return optim.Adam(self.parameters(), lr=1e-3, weight_decay=1e-5)


def _get_dist(eta):
    return torch.distributions.NegativeBinomial(
        total_count=torch.exp(eta[..., 0]),
        logits=eta[..., 1])


def get_dist(eta):
    return torch.distributions.NegativeBinomial(
        total_count=torch.exp(eta[..., 0] - eta[..., 1]),
        logits=eta[..., 1])


class Predictor:
    def __init__(self, module, tensorifier, transformer, context_length=12, prediction_length=3):
        super().__init__()
        self.module = module
        self.tensorifier = tensorifier
        self.transformer = transformer
        self.context_length = context_length
        self.prediction_length = prediction_length

    def predict(self, historic_data: DataSet, future_data: DataSet):
        historic_tensor = self.tensorifier.convert(historic_data)
        tmp = self.transformer.transform(historic_tensor.reshape(-1, historic_tensor.shape[-1]))
        historic_tensor = tmp.reshape(historic_tensor.shape).astype(np.float32)
        ts_dataset = TSDataSet(historic_tensor, None, self.context_length, self.prediction_length)
        instance = ts_dataset.last_prediction_instance()
        with torch.no_grad():
            eta = self.module(*instance)
        samples = get_dist(eta).sample((100,))
        output = {}
        period_range = future_data.period_range

        for i, location in enumerate(historic_data.keys()):
            s = samples[:, 0, :, i].T
            output[location] = Samples(period_range, s)
        return DataSet(output)


class NegativeBinomialLoss(nn.Module):
    def forward(self, eta, y_true):
        """
        y_pred: (batch_size, 2)  - First column: mean (μ), Second column: dispersion (θ)
        y_true: (batch_size, 1)  - Observed counts
        """
        na_mask = ~torch.isnan(y_true)
        y_true = y_true[na_mask]
        eta = eta[na_mask]
        nb_dist = get_dist(eta)
        loss = -nb_dist.log_prob(y_true).mean()
        return loss


class Estimator:
    tensorifier = Tensorifier(['rainfall', 'mean_temperature'])
    loader = DataLoader

    def __init__(self, context_length=12, prediction_length=3, debug=False):
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.debug = debug

    def train(self, data: DataSet):
        array_dataset = self.tensorifier.convert(data)
        n_locations = array_dataset.shape[1]
        transormer = StandardScaler()
        transformed_dataset = transormer.fit_transform(array_dataset.reshape(-1, array_dataset.shape[-1]))
        X = transformed_dataset.reshape(array_dataset.shape).astype(np.float32)
        y = np.array([series.disease_cases for series in data.values()]).T
        ts_dataset = TSDataSet(X, y, self.context_length, self.prediction_length)
        assert len(X) == len(y)
        loader = torch.utils.data.DataLoader(ts_dataset, batch_size=5, shuffle=True, drop_last=True, num_workers=7)
        module = RNNWithLocationEmbedding(n_locations, array_dataset.shape[-1], 4,
                                          prediction_length=self.prediction_length)
        lightning_module = DeepARLightningModule(
            module,
            NegativeBinomialLoss())
        trainer = L.Trainer(max_epochs=50 if not self.debug else 3,
                            accelerator="gpu" if torch.cuda.is_available() else "cpu")

        trainer.fit(lightning_module, loader)
        return Predictor(module, self.tensorifier, transormer, self.context_length, self.prediction_length)


def test():
    if True:
        dataset = DataSet.from_csv('/home/knut/Data/ch_data/weekly_laos_data.csv', FullData)
        results = evaluate_model(Estimator(context_length=52, prediction_length=12), dataset, prediction_length=12,
                                 n_test_sets=10, report_filename='laos_debug.pdf',
                                 weather_provider=QuickForecastFetcher)
    else:
        dataset = DataSet.from_csv('/home/knut/Data/ch_data/rwanda_harmonized.csv', FullData)  # 5534153
        results = evaluate_model(Estimator(context_length=12, prediction_length=6), dataset, prediction_length=6,
                                 n_test_sets=10, report_filename='rwanda_debug.pdf',
                                 weather_provider=QuickForecastFetcher)

    # dataset = DataSet.from_csv('~/Data/ch_data/rwanda_harmonized.csv', FullData)

    print(results)
    #
    # evaluate_model
    #
    # train, test = train_test_generator(dataset, prediction_length=3, n_test_sets=1)
    # historic, future, _ = next(test)
    # estimator = Estimator()
    # predictor = estimator.train(train)
    # samples = predictor.forecast(historic, future)
    # print(samples)
