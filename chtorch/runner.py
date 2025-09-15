from chtorch.configuration import ModelConfiguration
from chtorch.model_template import ExposedModelTemplate


from chap_core.datatypes import create_tsdataclass
from chap_core.spatio_temporal_data.temporal_dataclass import DataSet
from chapkit import ChapServiceInfo, PredictParams, TrainParams
from chapkit.model import ChapModelConfig, ChapModelRunnerBase


from uuid import UUID, uuid4


class MyConfig(ChapModelConfig, ModelConfiguration):
    pass




#runner = get_runner_from_funcs(my_train, my_predict)


class MyRunner(ChapModelRunnerBase[MyConfig]):

    def train(self, params: TrainParams) -> 'SerializedModel':
        ...

    def store_model(model: 'SerializedModel') -> UUID:
        ...

    async def on_train(self, params: TrainParams) -> UUID:
        df = params.body.data
        #model = self._train(...)
        #return self._store_model(model)

        print("Time period type, ", type(df.time_period[0]))
        model_template = ExposedModelTemplate()
        model_config = params.config

        estimator = model_template.get_model_from_chapkit_config(model_config)
        data_fields = estimator.covariate_names + [model_template.model_template_info.target]
        dc = create_tsdataclass(data_fields)
        print(df)
        print(type(df))
        dataset = DataSet.from_pandas(df, dc)
        predictor = estimator.train(dataset)
        model = predictor.serialize()

        artifact_id = uuid4()
        self._database.add_artifact(artifact_id, params.config, model)

        return artifact_id

    

    async def on_predict(self, params: PredictParams) -> UUID:
        df = params.body.future
        X = df[["rainfall", "mean_temperature"]]
        serialized_model = params.artifact
        from chtorch.estimator import Predictor
        model = Predictor.from_serialized(model)

        y_pred = params.artifact.predict(X)
        df["sample_0"] = y_pred

        print("Predictions: ", y_pred)

        artifact_id = uuid4()
        self._database.add_artifact(artifact_id, params.config, df)

        return artifact_id


info = ChapServiceInfo(
    author="Knut Rand",
    author_note="This model might need configuration of hyperparameters in order to work properly. When the model shows signs of overfitting, reduce 'state_dim' and/or increase 'dropout' and 'weight_decay'.",
    author_assessed_status="red",
    contact_email="knutdrand@gmail.com",
    description="This is a deep learning model template for CHAP. It is based on pytorch and can be used to train and predict using deep learning models. This typically need some configuration to fit the specifics of a dataset.",
    display_name="Torch Deep Learning Model",
    organization="HISP Centre, University of Oslo",
    organization_logo_url="https://landportal.org/sites/default/files/2024-03/university_of_oslo_logo.png",
    citation_info='Climate Health Analytics Platform. 2025. "Torch Deep Learning Model". HISP Centre, University of Oslo. https://dhis2-chap.github.io/chap-core/external_models/overview_of_supported_models.html',
    data_configuration = {
        "required_covariates": ["population"],
        "target": "disease_cases",
        "allow_free_additional_continuous_covariates": True,
        "supported_period_type": "any",
    }
)
