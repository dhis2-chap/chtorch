from chap_core.spatio_temporal_data.temporal_dataclass import DataSet

from chtorch.estimator import Estimator


class AuxilliaryEstimator(Estimator):
    def __init__(self, problem_configuration, model_configuration, auxilliary_datasets: dict[str, DataSet]):
        super().__init__(model_configuration=model_configuration, problem_configuration=problem_configuration)
        self._auxilliary_datasets = auxilliary_datasets


