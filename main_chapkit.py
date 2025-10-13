from typing import Any
from geojson_pydantic import FeatureCollection
import pandas as pd
from chtorch.estimator import Estimator, Predictor
from chapkit import (
    DataFrameSplit,
    SqlAlchemyChapDatabase,
)
from chapkit.model import (
    ChapModelService,
)
from sklearn.linear_model import LinearRegression
from chtorch.model_template import ExposedModelTemplate
from chtorch.configuration import ModelConfiguration, ModelConfigurationChapKit
from chtorch.runner import MyRunner
from chtorch.runner import MyConfig
from chtorch.runner import info

from chapkit.model import (
    ChapModelConfig,
    FunctionalChapModelRunner,
    ChapModelService,
)
from chap_core.spatio_temporal_data.temporal_dataclass import DataSet


def on_train(config: ModelConfigurationChapKit=None, data: pd.DataFrame=None, geo: FeatureCollection | None = None):
    model_config = config
    model_template = ExposedModelTemplate()
    estimator = model_template.get_model_from_chapkit_config(model_config)
    dataset = DataSet.from_pandas(data)
    model = estimator.train(dataset)
    return model.serialize()


def on_predict(config: ModelConfigurationChapKit, model: Any, historic: pd.DataFrame, future: pd.DataFrame, geo: FeatureCollection | None = None):
    historic_data = DataSet.from_pandas(historic)
    future_data = DataSet.from_pandas(future)
    model = Predictor.from_serialized(model)
    y_pred = model.predict(historic_data, future_data)
    pd = y_pred.to_pandas()
    # convert time_period column to str
    pd["time_period"] = pd["time_period"].astype(str)
    return pd
    return DataFrameSplit.from_pandas(y_pred.to_pandas())
    #return y_pred.to_pandas()


database = SqlAlchemyChapDatabase("target/chapkit.db", config_type=ModelConfigurationChapKit)
runner = FunctionalChapModelRunner(info, database, config_type=ModelConfigurationChapKit, on_train=on_train, on_predict=on_predict)
app = ChapModelService(
    runner=runner,
    database=database,
).create_fastapi()

