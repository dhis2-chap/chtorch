from chap_core import ModelTemplateInterface
from chtorch.estimator import ModelConfiguration, Estimator
from pydantic import BaseModel


class TorchModelTemplate(ModelTemplateInterface):
    def get_config_class(self):
        return ModelConfiguration

    def get_model(self, config: ModelConfiguration):
        ...




