"""Console script for chtorch."""
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Literal, Optional
from chap_core.assessment.dataset_splitting import train_test_generator
from chap_core.assessment.prediction_evaluator import evaluate_model, backtest
from chap_core.climate_predictor import QuickForecastFetcher
from chap_core.datatypes import FullData
from chap_core.geometry import Polygons
from chap_core.rest_api_src.worker_functions import samples_to_evaluation_response, dataset_to_datalist
from chap_core.spatio_temporal_data.temporal_dataclass import DataSet
from chtorch.validation import validate_dataset, filter_dataset
from cyclopts import App

from chtorch.estimator import Estimator
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
app = App()


@app.command()
def validation_training(dataset_path: str, frequency: Literal['M', 'W'] = 'M', max_epochs: Optional[int] = None):
    dataset = DataSet.from_csv(dataset_path, FullData)
    dataset, _ = train_test_generator(dataset, prediction_length=12 if frequency == 'M' else 52, n_test_sets=1)
    kwargs = get_kwargs(frequency) | dict(max_epochs=max_epochs)
    estimator = Estimator(validate=True, **kwargs)
    predictor = estimator.train(dataset)


def get_kwargs(frequency):
    return dict(context_length=52, prediction_length=12) if frequency == 'W' else dict(context_length=12,
                                                                                       prediction_length=4)


def get_commit_hash(path="."):
    return subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=path).decode().strip()


@app.command()
def evaluate(dataset_path: str, frequency: Literal['M', 'W'] = 'M', max_epochs: Optional[int] = None, remove_last_year: bool = True):
    '''
    This function should just be type hinted with common types,
    and it will run as a command line function
    Simple function

    >>> main_function()
    '''
    dataset = DataSet.from_csv(dataset_path, FullData)
    n_test_sets = 3 if frequency == 'M' else 26
    kwargs = get_kwargs(frequency) | dict(max_epochs=max_epochs)
    dataset = filter_dataset(dataset, n_test_sets+kwargs['prediction_length'])
    stem = Path(dataset_path).stem
    if remove_last_year:
        dataset, _ = train_test_generator(dataset, prediction_length=12 if frequency == 'M' else 52, n_test_sets=1)
    validate_dataset(dataset, lag=12)
    estimator = Estimator(**kwargs)

    predictions_list = backtest(estimator, dataset, prediction_length=kwargs['prediction_length'],
                                n_test_sets=n_test_sets, stride=1,
                                weather_provider=QuickForecastFetcher)
    name_lookup = Polygons(dataset.polygons).id_to_name_tuple_dict()
    name_lookup = {id: f'{t[0]}' for id, t in name_lookup.items()}
    response = samples_to_evaluation_response(
        predictions_list,
        quantiles=[0.05, 0.25, 0.5, 0.75, 0.95],
        real_data=dataset_to_datalist(dataset, 'dengue'))
    for evaluation_entry in response.predictions:
        evaluation_entry.orgUnit = name_lookup[evaluation_entry.orgUnit]
    for real_case in response.actualCases.data:
        real_case.ou = name_lookup[real_case.ou]
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    hash = get_commit_hash()
    with open(f'{stem}_evaluation_{timestamp}_{hash}.json', 'w') as f:
        f.write(response.json())
    logger.info(f'Evaluation results saved to {stem}_evaluation_{timestamp}.json')


def main():
    app()


if __name__ == "__main__":
    main()
