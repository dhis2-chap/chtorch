import numpy as np


def adapt_dataset(dataset, problem_configuration):
    for location, group in dataset.items():
        if problem_configuration.replace_zeros:
            group.disease_cases[group.disease_cases == 0] = np.nan
        elif problem_configuration.replace_nans:
            group.disease_cases[np.isnan(group.disease_cases)] = 0
    return dataset
