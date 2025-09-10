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
    return ModelConfiguration(context_length=12,
                              direct_ar=True,
                              embed_dim=2, n_hidden=2,
                              state_dim=4, max_dim=8)


@pytest.fixture
def hpo_model_configuration():
    return HPOConfiguration(context_length=(3, 12),
                            n_trials=2)


@pytest.fixture
def problem_configuration():
    return ProblemConfiguration(prediction_length=None, debug=True)


@pytest.mark.parametrize('problem_configuration',
                         [ProblemConfiguration(prediction_length=None, debug=True, predict_nans=v) for v in [False]])#, False]])
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


def test_serialize_deserialize(train_test):
    """Test that serialize/from_serialized methods work correctly."""
    train, test_data = train_test
    # test_data is a tuple of (historic_data, future_data, future_truth)
    historic_data, future_data, _ = test_data
    
    # Create and train a predictor
    estimator = Estimator(
        ProblemConfiguration(prediction_length=3, debug=True),
        ModelConfiguration(context_length=12, n_hidden=2, state_dim=4)
    )
    original_predictor = estimator.train(train)
    
    # Serialize the predictor
    serialized_data = original_predictor.serialize()
    
    # Verify serialized data structure
    assert isinstance(serialized_data, dict)
    assert 'module_state' in serialized_data
    assert 'transformer' in serialized_data
    assert 'target_scaler' in serialized_data
    assert 'predictor_info' in serialized_data
    
    # Deserialize to create a new predictor
    from chtorch.estimator import Predictor
    restored_predictor = Predictor.from_serialized(serialized_data)
    
    # Verify the restored predictor has the same configuration
    assert restored_predictor.model_configuration == original_predictor.model_configuration
    assert restored_predictor.problem_configuration == original_predictor.problem_configuration
    assert restored_predictor.context_length == original_predictor.context_length
    
    # Test that the restored predictor can make predictions without errors
    predictions = restored_predictor.predict(historic_data, future_data)
    
    # Basic validation that predictions are valid
    assert predictions is not None
    assert len(predictions.keys()) > 0  # Has predictions for at least one location
