from chtorch.estimator import Predictor
from chapkit import (
    SqlAlchemyChapDatabase,
)
from chapkit.model import (
    ChapModelService,
)
from sklearn.linear_model import LinearRegression
from chtorch.model_template import ExposedModelTemplate
from chtorch.configuration import ModelConfiguration
from chtorch.runner import MyRunner
from chtorch.runner import MyConfig
from chtorch.runner import info

from chapkit.model import (
    ChapModelConfig,
    FunctionalChapModelRunner,
    ChapModelService,
)
from chap_core.spatio_temporal_data.temporal_dataclass import DataSet

def on_train(model_config, training_data, geo):
    df = training_data
    print("Time period type, ", type(df.time_period[0]))
    model_template = ExposedModelTemplate()

    estimator = model_template.get_model_from_chapkit_config(model_config)
    dataset = DataSet.from_pandas(df)
    model = estimator.train(dataset)
    #model = predictor.serialize()  # do we need to serialize here?
    return model


def on_predict(model_config, model, historic_data, future_data, geo):
    df = future_data
    X = df[["rainfall", "mean_temperature"]]
    #model = Predictor.from_serialized(model)
    y_pred = model.predict(X)
    df["sample_0"] = y_pred
    print("Predictions: ", y_pred)
    return df


database = SqlAlchemyChapDatabase("target/chapkit.db", config_type=ModelConfiguration)
runner = FunctionalChapModelRunner(info, database, config_type=ModelConfiguration, on_train=on_train, on_predict=on_predict)
app = ChapModelService(
    runner=runner,
    database=database,
).create_fastapi()

"""
database = SqlAlchemyChapDatabase("target/chapkit.db", config_type=MyConfig)
runner = MyRunner(info, database, config_type=MyConfig)

app = ChapModelService(
    runner=runner,
    database=database,
).create_fastapi()

"""