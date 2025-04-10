# import torch
# from pytorch_forecasting import Baseline, DeepAR, TimeSeriesDataSet

import numpy as np
import torch


class TSDataSet(torch.utils.data.Dataset):
    def __init__(self, X, y, population, context_length, prediction_length, parents=None):
        if y is not None:
            assert y.shape == population.shape, f"y and population should have the same shape, got {y.shape} and {population.shape}"
        self.X = X  # time, location, feature
        self.y = y
        self.population = population
        self.total_length = context_length + prediction_length
        self.context_length = context_length
        self.prediction_length = prediction_length
        n_locations = X.shape[1]
        self.locations = np.array([np.arange(n_locations) for _ in range(context_length)])[..., None]
        self.n_locations = n_locations
        self.parents = parents

    @property
    def n_categories(self):
        return [self.n_locations, self.parents.max() + 1] if self.parents is not None else [self.n_locations, 0]

    @property
    def n_features(self):
        return self.X.shape[-1]

    def __len__(self):
        return len(self.X) - self.total_length + 1

    def __getitem__(self, i) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        x = self.X[i:i + self.context_length]
        y = self.y[i + self.context_length:i + self.total_length]
        population = self.population[i + self.context_length:i + self.total_length]
        assert y.shape == population.shape, f"y and population should have the same shape, got {y.shape} and {population.shape}"
        return x, self.locations, y, population

    def last_prediction_instance(self):
        last_population = self.population[-1]
        repeated_population = np.array([last_population for _ in range(self.prediction_length)])
        return (torch.from_numpy(self.X[None, -self.context_length:, ...]),
                torch.from_numpy(self.locations[None, ...]),
                torch.from_numpy(repeated_population[None, ...]))


class FlatTSDataSet(TSDataSet):
    def __len__(self):
        return (len(self.X) - self.total_length + 1) * self.X.shape[1]

    def __getitem__(self, item):
        i, j = divmod(item, self.X.shape[1])
        x = self.X[i:i + self.context_length, j]
        y = self.y[i + self.context_length:i + self.total_length, j]
        population = self.population[i + self.context_length:i + self.total_length, j]
        assert y.shape == population.shape, f"y and population should have the same shape, got {y.shape} and {population.shape}"
        p = self.parents[j]
        locations = np.array([(j, p) for _ in range(self.context_length)])
        return x, locations, y, population

    def last_prediction_instance(self):
        last_population = self.population[-1:].T
        repeated_population = np.repeat(last_population, self.prediction_length, axis=1)
        location_row = np.array([self.locations[0].ravel(), self.parents]).T
        location = np.array([location_row for _ in range(self.context_length)]).swapaxes(0, 1)
        assert location.shape == (self.n_locations, self.context_length, 2), location.shape
        return (torch.from_numpy(self.X[-self.context_length:, ...].swapaxes(0, 1)),
                torch.from_numpy(location),
                torch.from_numpy(repeated_population))


class MultiDataset(torch.utils.data.Dataset):
    def __init__(self, datasets: list[torch.utils.data.Dataset], main_dataset_weight: int = 1):
        self.datasets = datasets
        self.n_datasets = len(datasets)
        lens = [len(dataset) for dataset in datasets]

        self.cumulative_lens = np.cumsum(lens)
        self._category_offsets = np.cumsum([0] + [dataset.n_categories[0] for dataset in datasets])
        self._len = sum(lens)
        self._extra_len  = (main_dataset_weight-1) * lens[0]


    def __str__(self):
        return f"MultiDataset: {self.n_datasets} datasets, {self._len} samples, {self.n_features} features and {self.n_categories} categories. "

    @property
    def n_categories(self):
        return [self._category_offsets[-1], self.n_datasets]

    @property
    def n_features(self):
        return self.datasets[0].n_features

    def __len__(self):
        return self._len + self._extra_len

    def __getitem__(self, item):
        dataset_idx, new_idx = self._split_index(item)
        x, locations, y, population = self.datasets[dataset_idx][new_idx]
        locations = locations.copy()
        locations[:, 0] += self._category_offsets[dataset_idx]
        locations[:, 1] = dataset_idx
        #assert all(i<n for i, n in zip(np.max(locations, axis=0), self.n_categories)), \
        #    f"Locations {locations} exceed categories {self.n_categories}"
        return x, locations, y, population

    def _split_index(self, item):
        if item >= self._len:
            return 0, (item-self._len) % len(self.datasets[0])
        i = np.searchsorted(self.cumulative_lens, item, side='right')
        j = item-self.cumulative_lens[i-1] if i > 0 else item
        return i, j
