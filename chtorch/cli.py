"""Console script for chtorch."""
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Literal, Optional

import numpy as np
from chap_core.assessment.dataset_splitting import train_test_generator
from chap_core.assessment.prediction_evaluator import backtest
from chap_core.climate_predictor import QuickForecastFetcher
from chap_core.datatypes import FullData
from chap_core.geometry import Polygons
from chap_core.rest_api_src.worker_functions import samples_to_evaluation_response, dataset_to_datalist
from chap_core.spatio_temporal_data.temporal_dataclass import DataSet
from chap_core.time_period import Month

from chtorch.hpo import HPOConfiguration, HPOModelTemplate
from chtorch.model_template import TorchModelTemplate
from chtorch.validation import filter_dataset
from cyclopts import App

from chtorch.estimator import Estimator, ModelConfiguration, ProblemConfiguration
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
app = App()


@app.command()
def validation_training(dataset_path: str, frequency: Literal['M', 'W'] = 'M',
                        cfg: ModelConfiguration = ModelConfiguration(),
                        p_cfg: ProblemConfiguration = ProblemConfiguration()):
    dataset = DataSet.from_csv(dataset_path, FullData)
    dataset, _ = train_test_generator(dataset, prediction_length=12 if frequency == 'M' else 52, n_test_sets=1)
    p_cfg.validate = True
    estimator = Estimator(model_configuration=cfg, problem_configuration=p_cfg)
    estimator.train(dataset)
    logger.info(estimator.last_val_loss)


def get_kwargs(frequency):
    return dict(context_length=52, prediction_length=12) if frequency == 'W' else dict(context_length=12,
                                                                                       prediction_length=4)


def get_commit_hash(path="."):
    return subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=path).decode().strip()


def smape(target, samples):
    na_mask = np.isnan(target)
    target = target[~na_mask]
    if not len(target):
        return 0
    target = target[..., None]
    samples = samples[~na_mask]
    s = np.abs(samples) + np.abs(target)
    return np.mean(np.where(s == 0,
                            0,
                            2 * np.abs(samples - target) / s))


@app.command()
def hpo(dataset_path: str,
        cfg: HPOConfiguration = HPOConfiguration(),
        p_cfg: ProblemConfiguration = ProblemConfiguration(),
        year_fraction: float = 0.5):
    dataset, n_test_sets = _get_dataset(dataset_path, None, False, year_fraction)
    model_configuration = cfg
    cfg.context_length = (3, 13) if isinstance(dataset.period_range[0], Month) else (12, 53)
    model_template = HPOModelTemplate(p_cfg, auxilliary=False)
    estimator = model_template.get_model(model_configuration)
    predictions_list = list(backtest(estimator, dataset, prediction_length=p_cfg.prediction_length,
                                     n_test_sets=n_test_sets, stride=1,
                                     weather_provider=QuickForecastFetcher))
    _write_output(dataset, dataset_path, model_configuration, predictions_list)


@app.command()
def evaluate(dataset_path: str,
             frequency: Literal['M', 'W'] = 'M',
             remove_last_year: bool = True,
             cfg: ModelConfiguration = ModelConfiguration(),
             p_cfg: ProblemConfiguration = ProblemConfiguration(),
             cfg_path: Optional[Path] = None,
             aux: bool = False,
             year_fraction: float = 0.5,
             ):
    '''
    This function should just be type hinted with common types,
    and it will run as a command line function
    Simple function

    >>> main_function()
    '''
    dataset, n_test_sets = _get_dataset(dataset_path, frequency, remove_last_year, year_fraction)
    frequency = 'M' if isinstance(dataset.period_range[0], Month) else 'W'
    if cfg_path:
        model_configuration = ModelConfiguration.parse_file(cfg_path)
    else:
        model_configuration = cfg
    model_template = TorchModelTemplate(p_cfg, auxilliary=aux)
    cfg.context_length = 12 if frequency == 'M' else 38

    estimator = model_template.get_model(model_configuration)
    predictions_list = list(backtest(estimator, dataset, prediction_length=p_cfg.prediction_length,
                                     n_test_sets=n_test_sets, stride=1,
                                     weather_provider=QuickForecastFetcher))
    _write_output(dataset, dataset_path, model_configuration, predictions_list, p_cfg.predict_nans)


def _write_output(dataset, dataset_path, model_configuration, predictions_list, predict_nans=False):
    name_lookup = Polygons(dataset.polygons).id_to_name_tuple_dict()
    stem = Path(dataset_path).stem
    score = np.mean([smape(d.disease_cases, d.samples) for p in predictions_list for d in p.values()])
    logger.info(f'SMAPE: {score}')
    name_lookup = {id: f'{t[0]}' for id, t in name_lookup.items()}
    response = samples_to_evaluation_response(
        predictions_list,
        quantiles=[0.05, 0.25, 0.5, 0.75, 0.95],
        real_data=dataset_to_datalist(dataset, 'dengue'))
    for evaluation_entry in response.predictions:
        evaluation_entry.orgUnit = name_lookup[evaluation_entry.orgUnit]
    for real_case in response.actualCases.data:
        real_case.ou = name_lookup[real_case.ou]
    do_aggregate = True
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    hash = get_commit_hash()
    run_id = f'{timestamp}_{hash}'
    filename = f'{stem}_evaluation_{run_id}.json'
    with open(filename, 'w') as f:
        f.write(response.json())
    with open(f'{filename}.params.json', 'w') as f:
        f.write(model_configuration.json())
    with open(f'{filename}.score.txt', 'w') as f:
        f.write(str(score))
    logger.info(f'Evaluation results saved to {filename}')
    if do_aggregate:
        a_dataset = dataset.aggregate_to_parent()
        new_list = []
        for p in predictions_list:
            p.set_polygons(dataset.polygons)
            new_list.append(p.aggregate_to_parent(field_name='samples', nan_indicator=None if predict_nans else 'disease_cases'))
        a_predictions_list = new_list
        a_response = samples_to_evaluation_response(
            a_predictions_list,
            quantiles=[0.05, 0.25, 0.5, 0.75, 0.95],
            real_data=dataset_to_datalist(a_dataset, 'dengue'))
        with open(f'{stem}_evaluation_aggregated_{run_id}.json', 'w') as f:
            f.write(a_response.json())


def _get_dataset(dataset_path, frequency, remove_last_year, year_fraction):
    dataset = DataSet.from_csv(dataset_path, FullData)
    frequency = 'M' if isinstance(dataset.period_range[0], Month) else 'W'
    n_test_sets = 12 if frequency == 'M' else 52
    n_test_sets = int(n_test_sets * year_fraction)
    kwargs = get_kwargs(frequency)
    unused_periods = n_test_sets + kwargs['prediction_length']
    removed_periods = 12 if frequency == 'M' else 52
    if remove_last_year:
        unused_periods += removed_periods
    dataset = filter_dataset(dataset, unused_periods)
    if remove_last_year:
        dataset, _ = train_test_generator(dataset, prediction_length=removed_periods, n_test_sets=1)
    # validate_dataset(dataset, lag=12)
    return dataset, n_test_sets


def main():
    app()


if __name__ == "__main__":
    main()
