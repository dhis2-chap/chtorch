import numpy as np


def has_previous_cases(disease_cases):
    disease_cases = np.where(np.isnan(disease_cases), 0, disease_cases)
    mask = disease_cases > 0
    previous_cases = np.logical_or.accumulate(mask)
    return previous_cases


def test_has_previous():
    disease_cases = np.array([0, np.nan, 50, 0, 100, 0, 0])
    previous_cases = has_previous_cases(disease_cases)
    np.testing.assert_array_equal(
        previous_cases,
        [False, False, True, True, True, True, True]
    )

