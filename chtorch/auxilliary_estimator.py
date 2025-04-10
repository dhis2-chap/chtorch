from chap_core.spatio_temporal_data.temporal_dataclass import DataSet
from sklearn.preprocessing import StandardScaler

from chtorch.data_loader import TSDataSet, MultiDataset
from chtorch.estimator import Estimator


class AuxilliaryEstimator(Estimator):
    def __init__(self, problem_configuration, model_configuration, auxilliary_datasets: dict[str, DataSet]):
        super().__init__(model_configuration=model_configuration, problem_configuration=problem_configuration)
        self._auxilliary_datasets = auxilliary_datasets

    def _get_transformed_dataset(self, data) -> tuple[TSDataSet, StandardScaler]:
        """Convert the data to a format suitable for training."""
        datasets = [self._get_single_transformed_dataset(dataset)[0]
                    for dataset in self._auxilliary_datasets.values()]
        main_dataset, transformer = super()._get_transformed_dataset(data)
        datasets = [main_dataset]*10 + datasets
        multi_dataset = MultiDataset(datasets)
        print(multi_dataset)
        return multi_dataset, transformer
