import logging
from pathlib import Path

from joblib import dump, load
import numpy as np
import torch
from chap_core.data import DataSet
from chap_core.datatypes import Samples
from chtorch.distribution_loss import NegativeBinomialLoss, get_dist
from chtorch.lightning_module import DeepARLightningModule
from sklearn.preprocessing import StandardScaler
from pydantic import BaseModel
from chtorch.count_transforms import Log1pTransform
from chtorch.data_loader import TSDataSet, FlatTSDataSet
from chtorch.module import RNNWithLocationEmbedding, FlatRNN, RNNConfiguration
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

    def __init__(self, module,
                 tensorifier: Tensorifier,
                 transformer: StandardScaler,
                 context_length: int = 12, prediction_length: int = 3, count_transform=None):
        super().__init__()
        self.module = module
        self.tensorifier = tensorifier
        self.transformer = transformer
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.count_transform = count_transform

    def save(self, path: Path):
        torch.save(self.module.state_dict(), path)
        dump(self.transformer, path.with_suffix('.transformer'))
        self.tensorifier.save(path.with_suffix('.tensorifier'))

    @classmethod
    def load(cls, path):
        module = cls.module_class()
        module.load_state_dict(torch.load(path))
        transformer = load(path+'.transformer')
        tensorifier = Tensorifier.load(path+'.tensorifier')
        return cls(module, tensorifier, transformer)

    def predict(self, historic_data: DataSet, future_data: DataSet):
        historic_tensor, population, parents = self._get_prediction_dataset(historic_data)
        #logging.warning('SUCH A SHITTY HACK')
        #parents = np.zeros_like(parents)
        tmp = self.transformer.transform(historic_tensor.reshape(-1, historic_tensor.shape[-1]))
        historic_tensor = tmp.reshape(historic_tensor.shape).astype(np.float32)
        _DataSet = TSDataSet if not self.is_flat else FlatTSDataSet
        ts_dataset = _DataSet(historic_tensor, None, population, self.context_length, self.prediction_length, parents)
        *instance, population = ts_dataset.last_prediction_instance()
        with torch.no_grad():
            eta = self.module(*instance)
        samples = get_dist(eta, population, self.count_transform).sample((100,))
        output = {}
        period_range = future_data.period_range

        for i, location in enumerate(historic_data.keys()):
            if not self.is_flat:
                s = samples[:, 0, :, i].T
            else:
                s = samples[:, i, :].T
            output[location] = Samples(period_range, s)
        return DataSet(output)

    def _get_prediction_dataset(self, historic_data):
        return self.tensorifier.convert(historic_data)


class ModelConfiguration(RNNConfiguration):
    weight_decay: float = 1e-6
    max_epochs: int | None = None
    context_length: int = 12
    use_population: bool = True


class ProblemConfiguration(BaseModel):
    prediction_length: int = 3
    replace_zeros: bool = False
    debug: bool = False
    validate: bool = False


model_config = ModelConfiguration(weight_decay=1e-6,
                                  n_hidden=4,
                                  max_epochs=200,
                                  context_length=12,
                                  embed_dim=2,
                                  num_rnn_layers=1,
                                  n_layers=0)


class Estimator:
    features: list[str] = ['rainfall', 'mean_temperature']
    count_transform = Log1pTransform()
    is_flat = True
    predictor_cls = Predictor
    def __init__(self,
                 problem_configuration: ProblemConfiguration,
                 model_configuration: ModelConfiguration):
        self.last_val_loss = None
        self.context_length = model_configuration.context_length
        self.prediction_length = problem_configuration.prediction_length
        self.debug = problem_configuration.debug
        self.validate = problem_configuration.validate
        self.max_epochs = model_configuration.max_epochs
        self.problem_configuration = problem_configuration
        self.tensorifier = Tensorifier(
            self.features,
            self.count_transform,
            problem_configuration.replace_zeros,
            use_population=model_configuration.use_population)

        self.model_configuration = model_configuration

        if self.max_epochs is None:
            self.max_epochs = 2500 // self.context_length

    def train(self, data: DataSet):
        assert self.is_flat, "non-Flat model is deprecated"
        DataSet, Module = (TSDataSet, RNNWithLocationEmbedding) if (not self.is_flat) else (FlatTSDataSet, FlatRNN)
        train_dataset, transformer = self._get_transformed_dataset(data)
        if self.validate:
            cutoff = int(len(train_dataset) * 0.8)
            val_dataset = torch.utils.data.Subset(train_dataset,
                                                  range(cutoff, len(train_dataset)))
            train_dataset = torch.utils.data.Subset(train_dataset, range(cutoff))
        assert len(train_dataset.n_categories) == train_dataset[0][1].shape[-1], f"{train_dataset.n_categories} != {train_dataset[0][1].shape[-1]}"
        batch_size = 64 if self.is_flat else 8
        loader = torch.utils.data.DataLoader(train_dataset,
                                             batch_size=batch_size,
                                             shuffle=True,
                                             drop_last=True,
                                             num_workers=3)
        if self.validate:
            val_loader = torch.utils.data.DataLoader(val_dataset,
                                                     batch_size=batch_size,
                                                     shuffle=False,
                                                     drop_last=True,
                                                     num_workers=3)

        module = Module(train_dataset.n_categories,
                        train_dataset.n_features,
                        prediction_length=self.prediction_length,
                        cfg=self.model_configuration)

        lightning_module = DeepARLightningModule(
            module,
            NegativeBinomialLoss(count_transform=self.count_transform))

        trainer = L.Trainer(max_epochs=self.max_epochs if not self.debug else 3,
                            accelerator="cpu")

        trainer.fit(lightning_module, loader, val_loader if self.validate else None)
        self.last_val_loss = lightning_module.last_validation_loss
        return self.predictor_cls(module, self.tensorifier, transformer,
                                  self.context_length, self.prediction_length,
                                  self.count_transform)


    def _get_transformed_dataset(self, data) -> tuple[TSDataSet, StandardScaler]:
        """Convert the data to a format suitable for training."""
        return self._get_single_transformed_dataset(data)

    def _get_single_transformed_dataset(self, data):
        array_dataset, population, parents = self.tensorifier.convert(data)
        transformer = StandardScaler()
        input_features = array_dataset.shape[-1]
        transformed_dataset = transformer.fit_transform(array_dataset.reshape(-1, input_features))
        X = transformed_dataset.reshape(array_dataset.shape).astype(np.float32)
        y = np.array([series.disease_cases for series in data.values()]).T
        if self.problem_configuration.replace_zeros:
            y = np.where(y == 0, np.nan, y)
        assert len(X) == len(y)
        train_dataset = FlatTSDataSet(X, y, population, self.context_length, self.prediction_length, parents)
        return train_dataset, transformer
