from typing import Iterable

import numpy as np
import pandas as pd
from chap_core.database.dataset_tables import DataSet
from gluonts.dataset.pandas import PandasDataset


class GluonTsDatset:
    start: pd.Timestamp
    target: np.ndarray
    feat_dynamic_real: np.ndarray | None = None
    feat_static_cat: list[int] | None = None


class GluonTSAdaptor:
    def __init__(self, features=('rainfall', 'mean_temperature')):
        self.features = list(features)


    def to_gluonts(self, dataset: DataSet, start_index=0, static=None, real=None) -> Iterable[dict]:
        df = dataset.to_pandas()
        return [{
            "start": group["time_period"].iloc[0],
            "target": group["disease_cases"].values,
            "feat_dynamic_real": group[self.features].values,
            "feat_static_cat": [location],
            } for location, group in df.groupby("location")]

