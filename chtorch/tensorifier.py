import pickle
import logging
from dataclasses import dataclass
from datetime import datetime

from numpydantic import NDArray
import numpy as np
from pydantic import BaseModel

from chap_core.datatypes import TimeSeriesData
from chap_core.spatio_temporal_data.temporal_dataclass import DataSet
from chap_core.time_period.date_util_wrapper import TimeStamp

from chtorch.configuration import TensorifierConfig
from chtorch.count_transforms import CountTransform
from chtorch.target_based_transforms import has_previous_cases

logger = logging.getLogger(__name__)

def nan_helper(y):
    return np.isnan(y), lambda z: z.nonzero()[0]


def interpolate_nans(row):
    row = row.copy()
    nans, x = nan_helper(row)
    row[nans] = np.interp(x(nans), x(~nans), row[~nans])
    return row


def ar_transorm(counts):
    counts = interpolate_nans(counts)
    return np.log(counts + 1)


def get_covid_mask(period_range):
    mask = (period_range > TimeStamp.parse('2020-01-01')) & (period_range < TimeStamp.parse('2022-01-01'))
    return mask


class TensorOutput(BaseModel):
    X: NDArray
    y: NDArray
    ar: NDArray
    static_categoricals: NDArray


def concatenate_pydantic(data: list[TensorOutput]) -> TensorOutput:
    return TensorOutput(**{key: np.concatenate([getattr(d, key) for d in data]) for key in data[0].dict().keys()})


def smooth_population(population: np.ndarray) -> np.ndarray:
    return np.full_like(population, np.median(population))


@dataclass
class Tensorifier:
    """
    This class converts a dataset into tensors wich all have time index as first index and location index as second index.
    """
    count_transform: CountTransform
    config: TensorifierConfig

    def save(self, path: str):
        with open(path, 'wb') as f:
            pickle.dump(self, f)
            logger.info(f"Saved tensorifier to {path}")

    def load(self, path: str):
        with open(path, 'rb') as f:
            tensorifier = pickle.load(f)
            logger.info(f"Loaded tensorifier from {path}")
            return tensorifier

    @property
    def features(self):
        return self.config.additional_covariates

    def _debug_plot(self, data: DataSet):
        y = np.concatenate([interpolate_nans(location_data.disease_cases) for location_data in data.values()])
        population = np.concatenate([smooth_population(location_data.population) for location_data in data.values()])
        self.count_transform.__class__(np).plot_correlation(y, population)

    def convert(self, data: DataSet) -> np.ndarray:
        matrices = []
        populations = []
        for name, value in data.items():
            logger.info(f"Converting {name} with features {self.features} and population {value.population.shape}")
            m, pop = self._convert_for_location(value)
            matrices.append(m)
            populations.append(pop)
        parent_dict = data.get_parent_dict()
        parents = set(parent_dict.values())
        lookup_dict = {k: i for i, k in enumerate(parents)}
        parent_codes = np.array([lookup_dict[parent_dict[name]] for name in data.keys()])
        matrices = np.array(matrices)
        populations = np.array(populations)
        return matrices.swapaxes(0, 1), populations.T, parent_codes

    def _convert_for_location(self, location_data: TimeSeriesData):
        feature_columns = [getattr(location_data, feature) for feature in self.features]
        for name, column in zip(self.features, feature_columns):
            if np.isnan(column).any():
                assert False, f"Feature {name} contains NaNs: {column}"
            if np.isinf(column).any():
                assert False, f"Feature {name} contains infs: {column}"
        year_position = np.array(
            [year_position_from_datetime(period.start_timestamp.date) for period in location_data.time_period])

        population = smooth_population(location_data.population)
        population_column = np.log(population)
        cases = location_data.disease_cases

        if np.all(np.isnan(cases)):
            logger.warning(f"All cases are NaN for cases len {len(cases)}")
            target_column = np.zeros_like(cases)
        else:
            target_column = interpolate_nans(cases)
        target_column = self.count_transform.forward(target_column, population)
        na_mask = np.isnan(cases)
        extra_columns = [year_position]
        if self.config.mask_covid:
            extra_columns.append(get_covid_mask(location_data.time_period))
        if self.config.previous_cases:
            extra_columns.append(has_previous_cases(cases))
        extra_columns.extend([na_mask, target_column])
        if self.config.use_population:
            extra_columns.append(population_column)

        return np.array(
            feature_columns + extra_columns).T, population


def year_position_from_datetime(dt: datetime) -> float:
    day = dt.timetuple().tm_yday
    return day / 365
