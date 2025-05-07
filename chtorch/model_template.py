import numpy as np
from chap_core import ModelTemplateInterface
from chap_core.data.datasets import ISIMIP_dengue_harmonized

from chtorch.estimator import Estimator
from chtorch.configuration import ModelConfiguration, ProblemConfiguration
from chtorch.auxilliary_estimator import AuxilliaryEstimator
from chtorch.problem_adaptions import adapt_dataset
from chtorch.validation import filter_dataset


class TorchModelTemplate(ModelTemplateInterface):
    def __init__(self, problem_configuration: ProblemConfiguration, auxilliary=False):
        self.problem_configuration = problem_configuration
        self.auxilliary = auxilliary

    def get_config_class(self):
        return ModelConfiguration

    def get_model(self, config: ModelConfiguration):
        if self.auxilliary:
            datasets = {country_name: filter_dataset(adapt_dataset(data, self.problem_configuration), self.problem_configuration.prediction_length) for country_name, data in ISIMIP_dengue_harmonized.items()}
            for dataset in datasets.values():
                dataset.plot_aggregate()
            return AuxilliaryEstimator(problem_configuration=self.problem_configuration,
                                       model_configuration=config, auxilliary_datasets=datasets)

        return Estimator(problem_configuration=self.problem_configuration,
                         model_configuration=config)


