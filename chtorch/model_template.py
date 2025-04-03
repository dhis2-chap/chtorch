from chap_core import ModelTemplateInterface
from chap_core.data.datasets import ISIMIP_dengue_harmonized

from chtorch.estimator import ModelConfiguration, Estimator, ProblemConfiguration
from chtorch.auxilliary_estimator import AuxilliaryEstimator


class TorchModelTemplate(ModelTemplateInterface):
    def __init__(self, problem_configuration: ProblemConfiguration, auxilliary=False):
        self.problem_configuration = problem_configuration
        self.auxilliary = auxilliary

    def get_config_class(self):
        return ModelConfiguration

    def get_model(self, config: ModelConfiguration):
        if self.auxilliary:
            datasets = {country_name: ISIMIP_dengue_harmonized[country_name] for country_name in ['brazil', 'thailand']}
            return AuxilliaryEstimator(problem_configuration=self.problem_configuration,
                                       model_configuration=config, auxilliary_datasets=datasets)
        return Estimator(problem_configuration=self.problem_configuration,
                         model_configuration=config)
