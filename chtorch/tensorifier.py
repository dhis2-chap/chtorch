from dataclasses import dataclass
from datetime import datetime
from numpydantic import NDArray
import numpy as np
import pandas as pd
from chap_core.datatypes import TimeSeriesData
from chap_core.spatio_temporal_data.temporal_dataclass import DataSet
from chap_core.time_period.date_util_wrapper import TimeStamp
from pydantic import BaseModel
import logging

from chtorch.count_transforms import CountTransform, Log1pTransform

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
    features: list[str]
    count_transform: CountTransform
    replace_zeros: bool = False

    def _debug_plot(self, data: DataSet):
        y = np.concatenate([interpolate_nans(location_data.disease_cases) for location_data in data.values()])
        population = np.concatenate([smooth_population(location_data.population) for location_data in data.values()])
        self.count_transform.__class__(np).plot_correlation(y, population)

    def convert(self, data: DataSet) -> np.ndarray:
        matrices = []
        populations = []
        for name, value in data.items():
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
        if self.replace_zeros:
            cases = np.where(cases == 0, np.nan, cases)
        target_column = interpolate_nans(cases)
        target_column = self.count_transform.forward(target_column, population)
        assert not np.isnan(target_column).any(), f"Target column contains NaNs: {location_data.disease_cases}"
        assert not np.isinf(target_column).any(), f"Target column contains infs: {location_data.disease_cases}"

        na_mask = np.isnan(cases)

        assert not np.isnan(population_column).any(), f"Population column contains NaNs: {location_data.population}"
        assert not np.isinf(population_column).any(), f"Population column contains infs: {location_data.population}"
        return np.array(
            feature_columns + [year_position, get_covid_mask(location_data.time_period), na_mask, target_column,
                               population_column]).T, population


def year_position_from_datetime(dt: datetime) -> float:
    day = dt.timetuple().tm_yday
    return day / 365
