import pytest
from chap_core.assessment.prediction_evaluator import evaluate_model
from chap_core.climate_predictor import QuickForecastFetcher

from chtorch.auxilliary_estimator import AuxilliaryEstimator
from chtorch.estimator import Estimator, ModelConfiguration, ProblemConfiguration


@pytest.fixture
def model_configuration():
    return ModelConfiguration(context_length=12)


@pytest.fixture
def problem_configuration():
    return ProblemConfiguration(prediction_length=3, debug=True)


def test_estimator(ch_dataset, model_configuration, problem_configuration):
    estimator = Estimator(problem_configuration,
                          model_configuration=model_configuration)

    evaluate_model(estimator, ch_dataset, prediction_length=3,
                   n_test_sets=3,
                   weather_provider=QuickForecastFetcher)


def test_auxilliary_dataset(ch_dataset, auxilliary_datasets, model_configuration, problem_configuration):
    estimator = AuxilliaryEstimator(model_configuration=model_configuration,
                                    problem_configuration=problem_configuration,
                                    auxilliary_datasets=auxilliary_datasets)
    evaluate_model(estimator, ch_dataset, prediction_length=3,
                   n_test_sets=3,
                   weather_provider=QuickForecastFetcher)

@pytest.mark.skip
def test_save(train_test, tmp_path):
    train, test = train_test
    estimator = Estimator(ProblemConfiguration(prediction_length=3, debug=True),
                          ModelConfiguration(context_length=12))
    predictor = estimator.train(train)
    predictor.save(tmp_path / 'test_model')
    assert Path('test_model').exists()
