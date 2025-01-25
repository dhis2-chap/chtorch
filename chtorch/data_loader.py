#import torch
#from pytorch_forecasting import Baseline, DeepAR, TimeSeriesDataSet
from pytorch_forecasting import TimeSeriesDataSet

class Adaptor:
    def __init__(self, ch_dataset):
        self._ch_dataset = ch_dataset
        df = ch_dataset.to_pandas()
        dataset = TimeSeriesDataSet(
            df,
            group_ids=["location"],
            target="disease_cases",
            time_idx="time_period",
            min_encoder_length=0,
            max_encoder_length=12,
            min_prediction_length=3,
            max_prediction_length=3,
            time_varying_unknown_reals=["value"],
        )



