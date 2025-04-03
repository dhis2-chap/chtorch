from chap_core.assessment.prediction_evaluator import evaluate_model
from chap_core.climate_predictor import QuickForecastFetcher

from chtorch.estimator import Estimator, ModelConfiguration, ProblemConfiguration

def test_estimator(ch_dataset):
    estimator = Estimator(ProblemConfiguration(prediction_length=3, debug=True),
                          model_configuration=ModelConfiguration(context_length=12))


    evaluate_model(estimator, ch_dataset, prediction_length=3,
                   n_test_sets=3,
                   weather_provider=QuickForecastFetcher)
