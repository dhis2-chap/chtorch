from chap_core import ModelTemplateInterface
from chtorch.estimator import ModelConfiguration, Estimator, ProblemConfiguration


class TorchModelTemplate(ModelTemplateInterface):
    def __init__(self, problem_configuration: ProblemConfiguration):
        self.problem_configuration = problem_configuration

    def get_config_class(self):
        return ModelConfiguration

    def get_model(self, config: ModelConfiguration):
        return Estimator(problem_configuration=self.problem_configuration,
                         model_configuration=config)
