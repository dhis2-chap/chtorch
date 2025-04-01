import numpy as np
from chap_core.data import DataSet

def validate_dataset(dataset: DataSet, lag=0):

    for location, data in dataset.items():
        n = len(data.disease_cases)
        assert not np.isnan(data.disease_cases[:n-lag]).all(), location
        assert not np.isnan(data.population[:n-lag]).all(), location

def filter_dataset(dataset: DataSet, lag=0):
    new_dict = {}
    for location, data in dataset.items():
        n = len(data.disease_cases)
        if np.isnan(data.disease_cases[:n-lag]).all():
            continue
        new_dict[location] = data
    return DataSet(new_dict, dataset.polygons)

