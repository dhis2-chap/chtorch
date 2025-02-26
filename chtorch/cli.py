"""Console script for chtorch."""
from pathlib import Path

# todo


import typer
from chap_core.assessment.prediction_evaluator import evaluate_model
from chap_core.climate_predictor import QuickForecastFetcher
from chap_core.datatypes import FullData
from chap_core.spatio_temporal_data.temporal_dataclass import DataSet

from chtorch.estimator import Estimator


def validation_training(dataset_path: str):
    dataset = DataSet.from_csv(dataset_path, FullData)
    estimator = Estimator(context_length=1, prediction_length=3, validate=True)
    predictor = estimator.train(dataset)

def evaluate(dataset_path: str):
    '''
    This function should just be type hinted with common types,
    and it will run as a command line function
    Simple function

    >>> main_function()
    '''

    dataset = DataSet.from_csv(dataset_path, FullData)
    stem = Path(dataset_path).stem + '.pdf'
    results = evaluate_model(
        Estimator(context_length=52, prediction_length=12), dataset, prediction_length=12,
                                 n_test_sets=20, report_filename=stem,
                                 weather_provider=QuickForecastFetcher)
    print(results)


def main():
    typer.run(validation_training)


if __name__ == "__main__":
    main()
