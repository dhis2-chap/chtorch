import logging

from chap_core.time_period import Month
# from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.preprocessing import StandardScaler
from joblib import dump, load
from pathlib import Path
import lightning as L
from pydantic import BaseModel
import numpy as np
import torch

from chap_core.data import DataSet
from chap_core.datatypes import Samples

from chtorch.configuration import ModelConfiguration, ProblemConfiguration
from chtorch.data_augmentation import get_augmentation
from chtorch.distribution_loss import NegativeBinomialLoss, NBLossWithNaN
from chtorch.lightning_module import DeepARLightningModule

from chtorch.count_transforms import Log1pTransform
from chtorch.data_loader import TSDataSet, FlatTSDataSet
from chtorch.module import RNNWithLocationEmbedding, FlatRNN
from chtorch.target_scaler import TargetScaler
from chtorch.tensorifier import Tensorifier

logger = logging.getLogger(__name__)


# How should we handle configurations that depend on others
# Hierarchical, default values 0


def print_parameter_counts(model):
    total_params = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            num_params = param.numel()
            print(f"{name:50}: {num_params}")
            total_params += num_params
    print(f"\nTotal trainable parameters: {total_params}")


class ModelBase:
    problem_configuration: ProblemConfiguration
    model_configuration: ModelConfiguration

    def _get_loss_class(self):
        return NegativeBinomialLoss if not self.problem_configuration.predict_nans else NBLossWithNaN

    def _get_tensorifier(self):
        return Tensorifier(
            Log1pTransform(),
            self.model_configuration
            )


class PredictorInfo(BaseModel):
    model_configuration: ModelConfiguration
    problem_configuration: ProblemConfiguration
    n_features: int
    n_categories: list[int]


class Predictor(ModelBase):
    is_flat = True

    def __init__(self, module,
                 predictor_info: PredictorInfo,
                 transformer: StandardScaler,
                 target_scaler=None):
        super().__init__()
        problem_configuration = predictor_info.problem_configuration
        model_configuration = predictor_info.model_configuration
        self._predictor_info = predictor_info
        self.problem_configuration = problem_configuration
        self.model_configuration = model_configuration
        self.module = module
        self.tensorifier = self._get_tensorifier()
        self.transformer = transformer
        self.context_length = model_configuration.context_length
        self.count_transform = Log1pTransform()
        self._loss_class = self._get_loss_class()
        self._target_scaler = target_scaler
        # assert target_scaler is not None, target_scaler

    def save(self, path: Path | str):
        path = Path(path)
        torch.save(self.module.state_dict(), path)
        dump(self.transformer, path.with_suffix('.transformer'))
        with open(path.with_suffix('.predictor_info.json'), 'w') as f:
            f.write(self._predictor_info.model_dump_json())

    @classmethod
    def load(cls, path: str | Path):
        path = Path(path)
        predictor_info = PredictorInfo.parse_file(path.with_suffix('.predictor_info.json'))
        module = FlatRNN(
            num_categories=predictor_info.n_categories,
            input_feature_dim=predictor_info.n_features,
            prediction_length=predictor_info.problem_configuration.prediction_length,
            cfg=predictor_info.model_configuration)
        module.load_state_dict(torch.load(path))
        transformer = load(path.with_suffix('.transformer'))
        return cls(module,
                   predictor_info,
                   transformer)
    
    def serialize(self) -> dict:
        """Serialize the predictor to a dictionary that can be stored in a database."""
        import io
        import pickle
        import base64
        
        # Serialize module state dict
        module_buffer = io.BytesIO()
        torch.save(self.module.state_dict(), module_buffer)
        module_bytes = base64.b64encode(module_buffer.getvalue()).decode('utf-8')
        
        # Serialize transformer
        transformer_bytes = base64.b64encode(pickle.dumps(self.transformer)).decode('utf-8')
        
        # Serialize target scaler if it exists
        target_scaler_bytes = None
        if self._target_scaler is not None:
            target_scaler_bytes = base64.b64encode(pickle.dumps(self._target_scaler)).decode('utf-8')
        
        return {
            'module_state': module_bytes,
            'transformer': transformer_bytes,
            'target_scaler': target_scaler_bytes,
            'predictor_info': self._predictor_info.model_dump()
        }
    
    @classmethod
    def from_serialized(cls, data: dict):
        """Reconstruct a predictor from serialized data."""
        import io
        import pickle
        import base64
        
        # Deserialize predictor info
        predictor_info = PredictorInfo(**data['predictor_info'])
        
        # Create module
        module = FlatRNN(
            num_categories=predictor_info.n_categories,
            input_feature_dim=predictor_info.n_features,
            prediction_length=predictor_info.problem_configuration.prediction_length,
            cfg=predictor_info.model_configuration)
        
        # Load module state
        module_bytes = base64.b64decode(data['module_state'])
        module_buffer = io.BytesIO(module_bytes)
        module.load_state_dict(torch.load(module_buffer))
        
        # Deserialize transformer
        transformer = pickle.loads(base64.b64decode(data['transformer']))
        
        # Deserialize target scaler if it exists
        target_scaler = None
        if data.get('target_scaler') is not None:
            target_scaler = pickle.loads(base64.b64decode(data['target_scaler']))
        
        return cls(module,
                   predictor_info,
                   transformer,
                   target_scaler=target_scaler)

    def predict(self, historic_data: DataSet, future_data: DataSet):
        historic_tensor, population, parents = self._get_prediction_dataset(historic_data)
        historic_tensor = historic_tensor.astype(np.float32)
        _DataSet = TSDataSet if not self.is_flat else FlatTSDataSet
        ts_dataset = _DataSet(historic_tensor, None, population, self.context_length, self.problem_configuration.prediction_length, parents,
                              transformer=self.transformer)
        batch = ts_dataset.last_prediction_instance()
        with torch.no_grad():
            eta, *_ = self.module(batch.X, batch.locations)
            if self._target_scaler is not None:
                locations = batch.locations[:, 0, 0]
                assert len(np.unique(locations)) == len(locations)
                eta = self._target_scaler.scale_by_location(locations, eta)
        samples = self._loss_class.get_dist(eta, population, self.count_transform).sample((1000,))
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


model_config = ModelConfiguration(weight_decay=1e-6,
                                  n_hidden=4,
                                  max_epochs=200,
                                  context_length=12,
                                  embed_dim=2,
                                  num_rnn_layers=1,
                                  n_layers=0)


class Estimator(ModelBase):
    is_flat = True
    predictor_cls = Predictor

    @property
    def covariate_names(self):
        return self.model_configuration.additional_covariates + ['population']

    def __init__(self,
                 problem_configuration: ProblemConfiguration,
                 model_configuration: ModelConfiguration):
        self.count_transform = Log1pTransform()
        self.last_val_loss = None
        self.last_train_loss = None
        self.context_length = model_configuration.context_length
        self.debug = problem_configuration.debug
        self.validate = problem_configuration.validate
        self.max_epochs = model_configuration.max_epochs
        self.problem_configuration = problem_configuration
        self.model_configuration = model_configuration
        self.tensorifier = self._get_tensorifier()
        self.loss = self._get_loss_class()(count_transform=self.count_transform)
        self.n_params = None
        if self.max_epochs is None:
            self.max_epochs = 2500 // self.context_length
        self._validation_dataset = None

    def set_prediction_length_if_needed(self, data: DataSet):
        if self.problem_configuration.prediction_length is not None:
            return
        frequency = get_frequency(data)
        if frequency == 'M':
            self.problem_configuration.prediction_length = 3
        elif frequency == 'W':
            self.problem_configuration.prediction_length = 12

    def train(self, data: DataSet):
        assert self.is_flat, "non-Flat model is deprecated"
        self.set_prediction_length_if_needed(data)
        if self.validate:
            val_periods = {period.id for period in self._validation_dataset.period_range}
            logger.info(f'Validation dataset has period range: {self._validation_dataset.period_range}')
            logger.info(f'Training dataset has period range: {data.period_range}')
            data_periods = {period.id for period in data.period_range}

            assert len(data_periods.intersection(
                val_periods)) == 0, f"Validation periods {val_periods} overlap with training periods {data_periods}"
            assert self._validation_dataset is not None, 'Validation dataset is not set'

        DataSet, Module = (TSDataSet, RNNWithLocationEmbedding) if (not self.is_flat) else (FlatTSDataSet, FlatRNN)
        train_dataset, transformer, target_scaler, val_dataset = self._get_transformed_dataset(
            data,
            self._validation_dataset)
        # train_dataset, val_dataset = self._split_validation(train_dataset)

        for augmentation in self.model_configuration.augmentations:
            train_dataset.add_augmentation(get_augmentation(augmentation))

        assert len(train_dataset.n_categories) == train_dataset[0][1].shape[
            -1], f"{train_dataset.n_categories} != {train_dataset[0][1].shape[-1]}"

        loader = torch.utils.data.DataLoader(train_dataset,
                                             batch_size=self.model_configuration.batch_size,
                                             shuffle=True,
                                             drop_last=True,
                                             num_workers=3)
        if self.validate:
            val_loader = torch.utils.data.DataLoader(val_dataset,
                                                     batch_size=self.model_configuration.batch_size,
                                                     shuffle=False,
                                                     drop_last=False,
                                                     num_workers=3)

        module = Module(train_dataset.n_categories,
                        train_dataset.n_features,
                        prediction_length=self.problem_configuration.prediction_length,
                        output_dim=2 + self.problem_configuration.predict_nans,
                        cfg=self.model_configuration)
        print_parameter_counts(module)
        lightning_module = DeepARLightningModule(
            module,
            self.loss,
            target_scaler=target_scaler,
            cfg=self.model_configuration)

        data_name = data.metadata.name if hasattr(data, 'metadata') else 'default'
        # tb_logger = TensorBoardLogger(save_dir="tb_logs", name=data_name)
        trainer = L.Trainer(max_epochs=self.max_epochs if not self.debug else 3,
                            accelerator="cpu") # , logger=tb_logger)
        # tuner = Tuner(trainer)
        # trainer.tune()
        trainer.fit(lightning_module, loader, val_loader if self.validate else None)
        # self.last_val_loss = lightning_module.last_validation_loss
        # self.last_train_loss = lightning_module.last_train_loss
        print(lightning_module.last_train_losses)
        return self.predictor_cls(module,
                                  PredictorInfo(
                                      problem_configuration=self.problem_configuration,
                                      model_configuration=self.model_configuration,
                                      n_features=train_dataset.n_features,
                                      n_categories=train_dataset.n_categories),
                                  transformer,
                                  target_scaler=target_scaler)

    def add_validation(self, val_dataset: DataSet):
        self._validation_dataset = val_dataset

    def _split_validation(self, train_dataset):
        '''
        This needs to be done somewhere else. Per now there is some overlap in the prediction periods
        of train and validation
        '''
        n_splits = self.problem_configuration.validation_splits
        split = self.problem_configuration.validation_index
        split = split + n_splits
        n_splits = n_splits * 2
        validation_start = int(len(train_dataset) * float(split) / n_splits)
        validation_indices = list(range(validation_start, len(train_dataset)))
        training_indices = list(range(validation_start))
        logger.info(
            f"Splitting dataset into {len(training_indices)} training and {len(validation_indices)} validation samples based on {split + 1}/{n_splits}")
        val_dataset = train_dataset.subset(validation_indices)
        train_dataset = train_dataset.subset(training_indices)
        logger.info(f"Train dataset: {len(train_dataset)} samples, validation dataset: {len(val_dataset)} samples")
        return train_dataset, val_dataset

    @classmethod
    def load_predictor(cls, path: Path | str):
        return Predictor.load(path)

    def _get_transformed_dataset(self, data, validation_dataset=None) -> tuple[TSDataSet, StandardScaler]:
        """Convert the data to a format suitable for training."""
        return self._get_single_transformed_dataset(data, validation_dataset)

    def _get_single_transformed_dataset(self, data: DataSet, validation_dataset: DataSet | None = None) -> tuple[
        TSDataSet, StandardScaler, TargetScaler]:
        array_dataset, population, parents = self.tensorifier.convert(data)
        transformer = StandardScaler()
        input_features = array_dataset.shape[-1]
        transformer.fit_transform(array_dataset.reshape(-1, input_features))
        X = array_dataset.astype(np.float32)
        y = np.array([series.disease_cases for series in data.values()]).T
        target_scaler = TargetScaler(self.count_transform.forward(y, population))

        if self.problem_configuration.replace_zeros:
            y = np.where(y == 0, np.nan, y)

        assert len(X) == len(y)

        train_dataset = FlatTSDataSet(X, y, population, self.context_length, self.problem_configuration.prediction_length, parents,
                                      transformer=transformer)
        if validation_dataset is not None:
            val_array_dataset, val_population, _ = self.tensorifier.convert(validation_dataset)
            val_X = val_array_dataset.astype(np.float32)
            val_y = np.array([series.disease_cases for series in validation_dataset.values()]).T
            full_X = np.concatenate([X[-self.context_length:], val_X], axis=0)
            full_y = np.concatenate([y[-self.context_length:], val_y], axis=0)
            full_population = np.concatenate([population[-self.context_length:], val_population], axis=0)
            val_dataset = FlatTSDataSet(
                full_X, full_y, full_population,
                self.context_length,
                self.problem_configuration.prediction_length, parents,
                transformer=transformer)
            true_length = (len(validation_dataset.period_range) - self.problem_configuration.prediction_length + 1) * len(
                validation_dataset.locations())
            logger.info(
                f'Validation set has {len(val_dataset)} entries. n_periods = {len(validation_dataset.period_range)}-{self.problem_configuration.prediction_length}, n_locations={len(validation_dataset.locations())}')
            assert len(val_dataset) == true_length, (
                len(val_dataset), len(validation_dataset.period_range), self.problem_configuration.prediction_length)
            val_dataset = val_dataset.empty_removed()
        else:
            val_dataset = None

        train_dataset = train_dataset.empty_removed()
        return train_dataset, transformer, target_scaler, val_dataset


def get_frequency(dataset):
    return 'M' if isinstance(dataset.period_range[0], Month) else 'W'
