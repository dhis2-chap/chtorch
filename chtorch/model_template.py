from chap_core import ModelTemplateInterface
from chtorch.estimator import ModelConfiguration, Estimator
from pydantic import BaseModel


class TorchModelTemplate(ModelTemplateInterface):
    def __init__(self, prediction_length):
        self.prediction_length = prediction_length

    def get_config_class(self):
        return ModelConfiguration

    def get_model(self, config: ModelConfiguration):
        return Estimator(prediction_length=self.prediction_length,
                         model_configuration=config)





