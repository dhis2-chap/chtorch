import numpy as np


def decay_transform(input_series: np.ndarray, decay_period: int) -> np.ndarray:
    occurances = input_series > 0
    cur_value = 0
    decayed = np.zeros_like(input_series, dtype=int)
    for i, occ in enumerate(occurances):
        if occ:
            cur_value = decay_period
        else:
            cur_value = max(0, cur_value - 1)
        decayed[i] = cur_value
    return decayed


def test_decay_transform():
    input_series = np.array([0, 0, 50, 0, 100, 0, 0])
    decayed_series = decay_transform(input_series, 3)
    np.testing.assert_array_equal(
        decayed_series,
        [0, 0, 3, 2, 3, 2, 1]
    )
