from typing import Any
import chapkit
from geojson_pydantic import FeatureCollection
import pandas as pd
from chtorch.estimator import Predictor
from chtorch.model_template import ExposedModelTemplate
from chtorch.configuration import ModelConfigurationChapKit

from chap_core.spatio_temporal_data.temporal_dataclass import DataSet




from chapkit.api import AssessedStatus, MLServiceBuilder, MLServiceInfo
from chapkit.modules.artifact import ArtifactHierarchy
from chapkit.modules.ml import FunctionalModelRunner


async def on_train(config: ModelConfigurationChapKit=None, data: pd.DataFrame=None, geo: FeatureCollection | None = None):
    assert config is not None, "Config must be provided"
    #config = ModelConfigurationChapKit.model_validate(config.model_dump())
    assert isinstance(config, ModelConfigurationChapKit), f"Config is {type(config)}"
    assert isinstance(config, chapkit.BaseConfig)
    model_config = config
    model_template = ExposedModelTemplate()
    estimator = model_template.get_model_from_chapkit_config(model_config)
    dataset = DataSet.from_pandas(data)
    model = estimator.train(dataset)
    return model.serialize()


async def on_predict(config: ModelConfigurationChapKit, model: Any, historic: pd.DataFrame, future: pd.DataFrame, geo: FeatureCollection | None = None):
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
    display_name="Disease Prediction ML Service",
    version="1.0.0",
    summary="ML service for disease prediction using weather data",
    description="Train and predict disease cases based on rainfall and temperature data using Linear Regression",
    author="ML Team",
    author_assessed_status=AssessedStatus.yellow,
    contact_email="ml-team@example.com",
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
        config_schema=ModelConfigurationChapKit,
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

