from typing import Dict, List, Optional, Union
import numpy as np
from chtorch.estimator import Estimator, get_frequency
from chtorch.configuration import ModelConfiguration, ProblemConfiguration

import logging
logger = logging.getLogger(__name__)

from chap_core.datatypes import FullData, SamplesWithTruth 
from chap_core.spatio_temporal_data.temporal_dataclass import DataSet
from chap_core.assessment.dataset_splitting import train_test_generator


import ConfigSpace as CS
from pbmohpo.problems.problem import Problem


class TuneDeepAR(Problem):
    """
    DeepAR Tuning Problem to be used by pbmohpo

    Parameters
    ----------
    data_path: str
        Path to time series data
    """

    def __init__(
        self, data_path: str, seed: Optional[Union[np.random.RandomState, int]] = 42
    ) -> None:
        super().__init__(seed)
        self.data_path = data_path
        self.dataset = DataSet.from_csv(self.data_path, FullData)
        frequency = get_frequency(self.dataset)
        self.full_train, self.test_generator = train_test_generator(self.dataset, prediction_length=12, n_test_sets=1)

        self.train_dataset, val_generator = train_test_generator(self.full_train, prediction_length=12 if frequency == 'M' else 52, n_test_sets=1) # TODO: split off second test set here to use as the outer val set
        self.val_dataset = next(val_generator)[-1]

        self.objectives = ["last_val_loss", "last_train_loss"] # TODO: add meaningful other objectives to Estimator function

    def get_config_space(self) -> CS.ConfigurationSpace:
        """
        Defines the tuning space for DeepAR

        Returns
        -------
        ConfigSpace.ConfigurationSpace
            The configuration space of the problem
        """

        # TODO: make this and the optuna objective consistent
        
        cs = CS.ConfigurationSpace(seed=self.seed)

        cs.add_hyperparameter(
            CS.UniformFloatHyperparameter(
                "weight_decay", lower=1e-8, upper=1e-3, log=True
            )
        )

        cs.add_hyperparameter(
            CS.OrdinalHyperparameter('n_hidden', sequence=[4, 8, 16, 32])
        )

        cs.add_hyperparameter(
            CS.OrdinalHyperparameter('max_epochs', sequence=[2, 3])
        )

        cs.add_hyperparameter(
            CS.OrdinalHyperparameter('context_length', sequence=[7, 10, 12, 15, 20])
        )

        cs.add_hyperparameter(
            CS.OrdinalHyperparameter('embed_dim', sequence=[2, 4, 8])
        )

        cs.add_hyperparameter(
            CS.OrdinalHyperparameter('num_rnn_layers', sequence=[4, 8, 16, 32])
        )

        cs.add_hyperparameter(
            CS.OrdinalHyperparameter('n_layers', sequence=[4, 8, 16, 32])
        )

        return cs

    def get_objective_names(self) -> List:
        """
        Get the names of the objectives

        Returns
        -------
        List
            Names of objectives
        """
        return self.objectives

    def __call__(
        self,
        x: CS.Configuration,
        seed: Optional[Union[np.random.RandomState, int]] = None,
    ) -> Dict:
        """
        Objective function.

        The call method implements the objective function that should be optimized.

        Parameters
        ----------
        seed: int, np.random.RandomState
            Optional seed used to call the objective function.

        Returns
        -------
        Dict
            Dictionary of named objective values
        """

        params = dict(x)
        prob_config = ProblemConfiguration(replace_zeros=True, validate=True)
        model_config = ModelConfiguration(**params)
        estimator = Estimator(prob_config, model_config)
        estimator.add_validation(self.val_dataset)
        predictor = estimator.train(self.train_dataset) # TODO: add cross-validation?
        
        # Create predictions and merge with ground truth, TODO: Fix npdataclass dimensions
        for historic_data, future_data, future_truth in self.test_generator:
            r = predictor.predict(historic_data, future_data)
            samples_with_truth = future_truth.merge(r, result_dataclass=SamplesWithTruth)

        result_dict = {"last_val_loss": estimator.last_val_loss, "last_train_loss": estimator.last_train_loss}
        print(result_dict)
        return result_dict
