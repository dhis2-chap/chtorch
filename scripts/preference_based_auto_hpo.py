from typing import Dict, List, Optional, Union
import numpy as np
from chtorch.estimator import Estimator
from chtorch.configuration import ModelConfiguration, ProblemConfiguration
import logging
logger = logging.getLogger(__name__)

from chap_core.datatypes import FullData
from chap_core.spatio_temporal_data.temporal_dataclass import DataSet

import ConfigSpace as CS
from pbmohpo.problems.problem import Problem
from pbmohpo.decision_makers.decision_maker import DecisionMaker
from pbmohpo.optimizers.eubo import EUBO, qEUBO
from pbmohpo.benchmark import Benchmark


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
        self.dataset = DataSet.from_csv(path, FullData)
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

        _ = estimator.train(self.dataset) # TODO: add cross-validation?

        result_dict = {"last_val_loss": estimator.last_val_loss, "last_train_loss": estimator.last_train_loss}
        print(result_dict)
        return result_dict



if __name__ == "__main__":
    import sys
    path = sys.argv[1]
    prob = TuneDeepAR(data_path=path)

    dm = DecisionMaker(objective_names=prob.get_objective_names(), seed=0)

    print("Decision Maker Preference Scores:")
    print(dm.preferences)

    opt = EUBO(prob.get_config_space())
    bench = Benchmark(
        prob, opt, dm, eval_budget=10, dm_budget=10, eval_batch_size=2, dm_batch_size=1
    )
    print("Running EUBO")
    bench.run()
 
    #optuna_search(path, n_trials, output_name)
