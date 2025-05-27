import numpy as np


def adapt_dataset(dataset, problem_configuration):
    name = dataset.metadata.name
    if hasattr(dataset, 'metadata'):
        name = dataset.metadata.name
        if problem_configuration.replace_zeros:
            name += '_replace_zeros'
        if problem_configuration.replace_nans:
            name += '_replace_nans'
        dataset.metadata.name = name

    for location, group in dataset.items():
        if problem_configuration.replace_zeros:
            group.disease_cases[group.disease_cases == 0] = np.nan
        elif problem_configuration.replace_nans:
            group.disease_cases[np.isnan(group.disease_cases)] = 0
    return dataset
