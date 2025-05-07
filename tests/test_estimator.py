import pytest
from chap_core.assessment.prediction_evaluator import evaluate_model
from chap_core.climate_predictor import QuickForecastFetcher

from chtorch.auxilliary_estimator import AuxilliaryEstimator
from chtorch.cli import run_validation_training
from chtorch.estimator import Estimator
from chtorch.configuration import ModelConfiguration, ProblemConfiguration
from chtorch.hpo import HPOConfiguration, HPOEstimator


@pytest.fixture
def model_configuration():
    return ModelConfiguration(context_length=12, direct_ar=True)


@pytest.fixture
def hpo_model_configuration():
    return HPOConfiguration(context_length=(3, 12),
                            n_trials=2)


@pytest.fixture
def problem_configuration():
    return ProblemConfiguration(prediction_length=3, debug=True)


@pytest.mark.parametrize('problem_configuration',
                         [ProblemConfiguration(prediction_length=3, debug=True, predict_nans=v) for v in [False]])#, False]])
def test_estimator(ch_dataset, model_configuration, problem_configuration):
    estimator = Estimator(problem_configuration,
                          model_configuration=model_configuration)

    evaluate_model(estimator, ch_dataset, prediction_length=3,
                   n_test_sets=3,
                   weather_provider=QuickForecastFetcher)

@pytest.mark.parametrize('problem_configuration',
                         [ProblemConfiguration(prediction_length=3, debug=True, predict_nans=v) for v in [False]])#, False]])
def test_validation(ch_dataset, model_configuration, problem_configuration):
    run_validation_training(ch_dataset, model_configuration, problem_configuration)

def test_hpo_estimator(ch_dataset, hpo_model_configuration, problem_configuration):
    estimator = HPOEstimator(problem_configuration, hpo_model_configuration)
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


def test_save(train_test, tmp_path):
    train, test = train_test
    estimator = Estimator(ProblemConfiguration(prediction_length=3, debug=True),
                          ModelConfiguration(context_length=12))
    predictor = estimator.train(train)
    out_path = tmp_path / 'test_model'
    predictor.save(out_path)
    predictor.load(out_path)
    # assert out_path.exists()
