import pytest
import yaml
from pydantic import BaseModel

from chtorch.configuration import ModelConfiguration
from chtorch.pytorch_forecasting_estimator import MyModel


@pytest.fixture
def model_config():
    return ModelConfiguration(
        n_lag_periods=3,
        additional_continuous_covariates=["elevation", "population_density"]
    )


class PropertySchema(BaseModel):
    ...


def test_spec_roundrip():
    # Get schema
    schema = ModelConfiguration.model_json_schema()
    # Return to model
    print(schema)
    yaml_schema = yaml.dump(schema, sort_keys=False)
    print(yaml_schema)
