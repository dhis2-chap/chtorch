from typing import Any
import chapkit
from geojson_pydantic import FeatureCollection
import pandas as pd
from chtorch.estimator import Predictor
from chtorch.model_template import ExposedModelTemplate
from chtorch.configuration import ModelConfigurationChapKit, ModelConfigurationChapKitV2

from chap_core.spatio_temporal_data.temporal_dataclass import DataSet




from chapkit.api import AssessedStatus, MLServiceBuilder, MLServiceInfo
from chapkit.modules.artifact import ArtifactHierarchy
from chapkit.modules.ml import FunctionalModelRunner


async def on_train(config: ModelConfigurationChapKitV2=None, data: pd.DataFrame=None, geo: FeatureCollection | None = None):
    assert config is not None, "Config must be provided"
    #config = ModelConfigurationChapKit.model_validate(config.model_dump())
    assert isinstance(config, ModelConfigurationChapKitV2), f"Config is {type(config)}"
    assert isinstance(config, chapkit.BaseConfig)
    model_config = config
    model_template = ExposedModelTemplate()
    estimator = model_template.get_model_from_chapkit_config(model_config)
    dataset = DataSet.from_pandas(data)
    model = estimator.train(dataset)
    return model.serialize()


async def on_predict(config: ModelConfigurationChapKitV2, model: Any, historic: pd.DataFrame, future: pd.DataFrame, geo: FeatureCollection | None = None):
    historic_data = DataSet.from_pandas(historic)
    future_data = DataSet.from_pandas(future)
    model = Predictor.from_serialized(model)
    y_pred = model.predict(historic_data, future_data)
    pd = y_pred.to_pandas()
    # convert time_period column to str
    pd["time_period"] = pd["time_period"].astype(str)
    return pd

# Create ML service info with metadata
info = MLServiceInfo(
    display_name="Chapkit Torch Deep Learning Model",
    version="1.0.0",
    summary="ML service for disease prediction using weather data",
    description="This is a deep learning model template for CHAP. It is based on pytorch and can be used to train and predict using deep learning models. This typically need some configuration to fit the specifics of a dataset.",
    author="Knut Rand",
    author_note="This model might need configuration of hyperparameters in order to work properly. When the model shows signs of overfitting, reduce 'state_dim' and/or increase 'dropout' and 'weight_decay'.",
    author_assessed_status=AssessedStatus.red,
    contact_email="knutdrand@gmail.com",
    organization="HISP Centre, University of Oslo",
    organization_logo_url="https://landportal.org/sites/default/files/2024-03/university_of_oslo_logo.png",
    citation_info='Climate Health Analytics Platform. 2025. "Torch Deep Learning Model". HISP Centre, University of Oslo. https://dhis2-chap.github.io/chap-core/external_models/overview_of_supported_models.html',
    required_covariates=["population"],
    allow_free_additional_continuous_covariates=True,
)

# Create artifact hierarchy for ML artifacts
HIERARCHY = ArtifactHierarchy(
    name="ml_pipeline",
    level_labels={0: "trained_model", 1: "predictions"},
)

# Create functional model runner
runner = FunctionalModelRunner(on_train=on_train, on_predict=on_predict)

# to use db instead of in memory, use this and set database=db
#db = (chapkit.SqliteDatabaseBuilder()
    #.from_file('chtorch_artifacts.db')
    #.build()
#)
# Build the FastAPI application
app = (
    MLServiceBuilder(
        info=info,
        config_schema=ModelConfigurationChapKitV2,
        hierarchy=HIERARCHY,
        runner=runner,
    )
    .with_monitoring()
    #.with_database(db)
    .build()
)

#runner = FunctionalChapModelRunner(info, database, config_type=ModelConfigurationChapKit, on_train=on_train, on_predict=on_predict)
#app = ChapModelService(
    #runner=runner,
    #database=database,
#).create_fastapi()

