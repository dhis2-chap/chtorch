import json
from functools import partial
from pathlib import Path

import optuna
from chap_core import ModelTemplateInterface
from chap_core.datatypes import FullData
from chap_core.spatio_temporal_data.temporal_dataclass import DataSet

from chtorch.estimator import ProblemConfiguration, ModelConfiguration, Estimator
import logging

logger = logging.getLogger(__name__)


def suggest_model_configuration(trial):
    n_hidden = trial.suggest_int("n_hidden", 2, 8)
    return ModelConfiguration(
        weight_decay=trial.suggest_loguniform("weight_decay", 1e-8, 1e-3),
        num_rnn_layers=trial.suggest_int("num_rnn_layers", 1, 4),
        n_layers=trial.suggest_int("n_layers", 0, 4),
        n_hidden=n_hidden,
        embed_dim=n_hidden,
        output_embedding_dim=trial.suggest_int("output_embedding_dim", 0, 4),
        dropout=trial.suggest_float("dropout", 0.0, 0.5))


def objective(trial, dataset):
    wd = trial.suggest_loguniform("weight_decay", 1e-8, 1e-3)
    nh = trial.suggest_categorical("n_hidden", [4, 8, 16, 32])
    me = trial.suggest_categorical("max_epochs", [2, 3])
    cl = trial.suggest_categorical("context_length", [7, 10, 12, 15, 20])
    ed = trial.suggest_categorical("embed_dim", [2, 4, 8])
    nrl = trial.suggest_categorical("num_rnn_layers", [4, 8, 16, 32])
    nl = trial.suggest_categorical("n_layers", [4, 8, 16, 32])

    prob_config = ProblemConfiguration(replace_zeros=True)
    model_config = ModelConfiguration(weight_decay=wd,
                                      n_hidden=nh,
                                      max_epochs=me,
                                      context_length=cl,
                                      embed_dim=ed,
                                      num_rnn_layers=nrl,
                                      n_layers=nl)
    estimator = Estimator(prob_config, model_config, validate=True)

    _ = estimator.train(dataset)
    val_loss = estimator.last_val_loss

    return val_loss


def optuna_search(path, n_trials, output_name):
    dataset = DataSet.from_csv(path, FullData)
    tune_hyperparameters(dataset, n_trials, output_name)


def tune_hyperparameters(dataset, n_trials, output_name):
    study = optuna.create_study(direction="minimize")
    study.optimize(partial(objective, dataset=dataset), n_trials=n_trials)
    logger.info(f"Number of finished trials: {len(study.trials)}")
    logger.info("Best trial:")
    best_trial = study.best_trial
    logger.info(f"  Loss value: {best_trial.value}")
    logger.info("  Params:")
    for key, value in best_trial.params.items():
        logger.info(f"  {key}: {value}")
    best_model_config = ModelConfiguration(**best_trial.params)
    with open(output_name, 'w') as f:
        json.dump(best_model_config.model_dump(), f, indent=4)


class HPOConfiguration(ModelConfiguration):
    weight_decay: tuple[float] = (1e-6, 1e-2)
    n_layers: tuple[int] = (0, 3)
    #context_length: tuple[int] = (3, 13)
    n_trials: int = 20


class HPOEstimator:
    estimator_class = Estimator

    def __init__(self, problem_configuration: ProblemConfiguration, model_configuration: HPOConfiguration):
        self._problem_configuration = problem_configuration
        self._model_configuration = model_configuration

    def train(self, data: DataSet):
        model_config = self._find_best_model_config(data)
        return self.estimator_class(self._problem_configuration, model_config).train(data)

    def _objective_func(self, trial, dataset: DataSet):
        hidden_dim = trial.suggest_int('hidden_dim', 2, 10)
        d = {'weight_decay': trial.suggest_float("weight_decay", self._model_configuration.weight_decay[0],
                                                 self._model_configuration.weight_decay[1], log=True),
             'n_layers': trial.suggest_int("n_layers", self._model_configuration.n_layers[0],
                                           self._model_configuration.n_layers[1]),
             'dropout': trial.suggest_float('dropout', 0.0, 0.5),
             'output_embedding': trial.suggest_int('output_embedding', 0, 3),
             'num_rnn_layers': trial.suggest_int('num_rnn_layers', 1, 3),
             'n_hidden': hidden_dim,
             'embed_dim': hidden_dim}

        model_config = ModelConfiguration(**(self._model_configuration.dict() | d))
        problem_configuration = self._problem_configuration.copy()
        problem_configuration.validate = True
        estimator = Estimator(problem_configuration, model_config)
        _ = estimator.train(dataset)
        val_loss = estimator.last_val_loss
        return val_loss

    @classmethod
    def load_predictor(cls, filepath):
        return cls.estimator_class.load_predictor(filepath)

    def _find_best_model_config(self, data: DataSet):
        cur_dir = Path(__file__).parent
        study = optuna.create_study(direction="minimize", storage=f'sqlite:///{cur_dir}/study.db')
        objective = partial(self._objective_func, dataset=data)
        study.optimize(objective, n_trials=self._model_configuration.n_trials)
        logger.info(f"Number of finished trials: {len(study.trials)}")
        logger.info("Best trial:")
        best_trial = study.best_trial
        logger.info(f"  Loss value: {best_trial.value}")
        logger.info("  Params:")
        for key, value in best_trial.params.items():
            logger.info(f"  {key}: {value}")
        best_model_config = ModelConfiguration(**(self._model_configuration.dict() | best_trial.params))
        return best_model_config


class HPOModelTemplate(ModelTemplateInterface):
    def __init__(self, problem_configuration: ProblemConfiguration, auxilliary=False):
        self.problem_configuration = problem_configuration
        self.auxilliary = auxilliary

    def get_config_class(self) -> type[ModelConfiguration]:
        return HPOConfiguration

    def get_model(self, model_configuration: ModelConfiguration = None) -> 'ConfiguredModel':
        return HPOEstimator(self.problem_configuration, model_configuration)
