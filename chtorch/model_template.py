import numpy as np
from chap_core import ModelTemplateInterface
from chap_core.models.model_template_interface import ModelConfiguration as CHAPModelConfiguration
from chap_core.data.datasets import ISIMIP_dengue_harmonized
from chap_core.external.model_configuration import ModelTemplateSchema, ModelTemplateConfigCommon
from chap_core.models.model_template_interface import InternalModelTemplate

from chtorch.estimator import Estimator
from chtorch.configuration import ModelConfiguration, ProblemConfiguration
from chtorch.auxilliary_estimator import AuxilliaryEstimator
from chtorch.problem_adaptions import adapt_dataset
from chtorch.validation import filter_dataset




class TorchModelTemplate:
    '''
    Model template for configuring deep learning models using pytorch
    '''
    model_config_class = ModelConfiguration

    def __init__(self, problem_configuration: ProblemConfiguration = ProblemConfiguration(), auxilliary=False):
        self.problem_configuration = problem_configuration
        self.auxilliary = auxilliary

    def get_model(self, config: ModelConfiguration):
        if self.auxilliary:
            datasets = {country_name: filter_dataset(adapt_dataset(data, self.problem_configuration),
                                                     self.problem_configuration.prediction_length) for
                        country_name, data in ISIMIP_dengue_harmonized.items()}
            for dataset in datasets.values():
                dataset.plot_aggregate()
            return AuxilliaryEstimator(problem_configuration=self.problem_configuration,
                                       model_configuration=config, auxilliary_datasets=datasets)

        return Estimator(problem_configuration=self.problem_configuration,
                         model_configuration=config)


class ExposedModelTemplate(ModelTemplateInterface):
    _model_template = TorchModelTemplate(ProblemConfiguration(debug=True))
    model_config_class = ModelConfiguration
    model_template_info = ModelTemplateConfigCommon(
        supported_period_type='any',
        required_covariates=['population'],
        allow_free_additional_continuous_covariates=True,
        user_options=model_config_class.model_json_schema()['properties']
    )

    def get_schema(self):
        ...

    def get_model(self, model_configuration: CHAPModelConfiguration = CHAPModelConfiguration):
        config = ModelConfiguration(**model_configuration.user_option_values,
                                    additional_covariates=model_configuration.additional_continuous_covariates)
        return self._model_template.get_model(config)

