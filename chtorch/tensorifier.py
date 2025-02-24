from datetime import datetime

import numpy as np
from chap_core.datatypes import TimeSeriesData
from chap_core.spatio_temporal_data.temporal_dataclass import DataSet
from chap_core.time_period.date_util_wrapper import TimeStamp


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


class Tensorifier:
    def __init__(self, features):
        self.features = features

    def convert(self, data: DataSet):
        matrices = [self._convert_for_location(value) for value in data.values()]
        return np.array(matrices).swapaxes(0, 1)

    def _convert_for_location(self, location_data: TimeSeriesData):
        feature_columns = [getattr(location_data, feature) for feature in self.features]
        year_position = np.array(
            [year_position_from_datetime(period.start_timestamp.date) for period in location_data.time_period])
        target_column = ar_transorm(location_data.disease_cases)
        na_mask = np.isnan(location_data.disease_cases)
        population_column = np.log(location_data.population)
        return np.array(
            feature_columns + [year_position, get_covid_mask(location_data.time_period), na_mask, target_column, population_column]).T


def year_position_from_datetime(dt: datetime) -> float:
    day = dt.timetuple().tm_yday
    return day / 365
