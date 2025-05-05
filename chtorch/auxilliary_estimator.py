import numpy as np
from chap_core.spatio_temporal_data.temporal_dataclass import DataSet
from sklearn.preprocessing import StandardScaler

from chtorch.data_loader import TSDataSet, MultiDataset
from chtorch.estimator import Estimator, Predictor
from chtorch.target_scaler import MultiTargetScaler


class AuxilliaryPredictor(Predictor):
    def _get_prediction_dataset(self, historic_data: DataSet):
        *r, parents = self.tensorifier.convert(historic_data)
        parents = np.zeros_like(parents)
        return *r, parents


class AuxilliaryEstimator(Estimator):
    predictor_cls = AuxilliaryPredictor

    def __init__(self, problem_configuration, model_configuration, auxilliary_datasets: dict[str, DataSet]):
        super().__init__(model_configuration=model_configuration, problem_configuration=problem_configuration)
        self._auxilliary_datasets = auxilliary_datasets

    def _get_transformed_dataset(self, data) -> tuple[TSDataSet, StandardScaler]:
        """Convert the data to a format suitable for training."""
        tuples = [self._get_single_transformed_dataset(dataset) for dataset in self._auxilliary_datasets.values()]
        datasets = [t[0] for t in tuples]

        main_dataset, transformer, target_scaler = super()._get_transformed_dataset(data)
        target_scaler = MultiTargetScaler([target_scaler] + [t[-1] for t in tuples])
        datasets = [main_dataset] + datasets
        multi_dataset = MultiDataset(datasets, main_dataset_weight=10)
        return multi_dataset, transformer, target_scaler
