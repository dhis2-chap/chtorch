"""Console script for chtorch."""
from datetime import datetime
from pathlib import Path
from typing import Literal, Optional
import pygit
from chap_core.assessment.dataset_splitting import train_test_generator
from chap_core.assessment.prediction_evaluator import evaluate_model, backtest
from chap_core.climate_predictor import QuickForecastFetcher
from chap_core.datatypes import FullData
from chap_core.rest_api_src.worker_functions import samples_to_evaluation_response, dataset_to_datalist
from chap_core.spatio_temporal_data.temporal_dataclass import DataSet
from cyclopts import App
from chtorch.estimator import Estimator
import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
app = App()


@app.command()
def validation_training(dataset_path: str, frequency: Literal['M', 'W'] = 'M'):
    dataset = DataSet.from_csv(dataset_path, FullData)
    kwargs = get_kwargs(frequency)
    estimator = Estimator(validate=True, **kwargs)
    predictor = estimator.train(dataset)


def get_kwargs(frequency):
    return dict(context_length=52, prediction_length=12) if frequency == 'W' else dict(context_length=12,
                                                                                       prediction_length=4)
def get_commit_hash():
    return pygit.get_commit_hash()

@app.command()
def evaluate(dataset_path: str, frequency: Literal['M', 'W'] = 'M', max_epochs: Optional[int] = None):
    '''
    This function should just be type hinted with common types,
    and it will run as a command line function
    Simple function

    >>> main_function()
    '''

    dataset = DataSet.from_csv(dataset_path, FullData)
    stem = Path(dataset_path).stem
    kwargs = get_kwargs(frequency) | dict(max_epochs=max_epochs)
    dataset, _ = train_test_generator(dataset, prediction_length=12 if frequency == 'M' else 52, n_test_sets=1)
    estimator = Estimator(**kwargs)
    predictions_list = backtest(estimator, dataset, prediction_length=kwargs['prediction_length'],
                                n_test_sets=12 if frequency == 'M' else 26, stride=1,
                                weather_provider=QuickForecastFetcher)
    response = samples_to_evaluation_response(
        predictions_list,
        quantiles=[0.05, 0.25, 0.5, 0.75, 0.95],
        real_data=dataset_to_datalist(dataset, 'dengue'))
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    hash = get_commit_hash()
    with open(f'{stem}_evaluation_{timestamp}_{hash}.json', 'w') as f:
        f.write(response.json())
    logger.info(f'Evaluation results saved to {stem}_evaluation_{timestamp}.json')


def main():
    app()


if __name__ == "__main__":
    main()
