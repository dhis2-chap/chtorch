import dataclasses
import numpy as np
import pytest
from math import prod

'''
The main idea of the gridded model is to represeent the geospatial climate data in a grid,
while the health data are represented in in aggregates over both polygons in space and different
time periods (weeks, months). To begin with we eill assume that the time resolution is the same for the aggregated and
gridded data.

The  point is to train a very flexible model to calculate low dimensional feature vectors for each pixel value, and then
to have a symmetric pooling functions that aggregates all the low dimensional feature vectors (for each pixel) into two numbers
representing the mean and precision of the distribution of the aggreagted disease counts.

THis module will try to define the forward function of this mehcansim, while the backwards pass needs to be defined
specially for each nn framework.
'''

FeatureID = str
RegionID = str
FullGriddedArray = np.ndarray  # T M N F
L = 1
T = 2
M = 5
N = 6
PADDING = 1
M_i = M - PADDING*2
N_i = N - PADDING*2

F = 2

GriddedDict = dict[FeatureID, np.ndarray]


# First we define the test data

@pytest.fixture
def gridded_data() -> dict[FeatureID, np.ndarray]:  # T, M, N
    '''
    Return a gridded dataset in simplest possible format
    Dimensions are feature, time, space**2
    '''
    return {
        'rainfall': np.random.random(T * M * N).reshape((T, M, N)),
        'temperature': np.random.random(T * M * N).reshape((T, M, N))
    }


AggregationDict = dict[RegionID, np.ndarray]
AttributionMatrix = np.ndarray
RegionAttribution = dict[RegionID, AttributionMatrix]
GridTensor = np.ndarray  # T, M, N, F


@pytest.fixture
def aggregated_data() -> AggregationDict:  # T
    """Disease case counts for all regions for all timepoints"""
    return {'oslo': np.arange(T)}


@pytest.fixture
def region_attribution() -> RegionAttribution:  # M, N
    """
    Matrices that represent how much of each pixel should be attributed to each region
    """
    return {'Oslo': np.array([[1., 1., 1., 0.],
                              [0.5, 1., 1., 0.5],
                              [0., 0., 1., 0.]])}


# Dicts area human readable, but we need tensore to feed to the network.
def get_grid_tensor(grid_dict: FullGriddedArray) -> GridTensor:
    array = np.array(list(grid_dict.values()))
    print(array.shape)
    return array.transpose((1, 2, 3, 0))


def test_get_grid_tensor(gridded_data):
    tensor = get_grid_tensor(gridded_data)
    assert tensor.shape == (T, M, N, F)


# We combine this into a simple dataclass

@dataclasses.dataclass
class MixedData:
    gridded_data: GriddedDict
    aggregated_data: AggregationDict
    region_attribution: RegionAttribution


@pytest.fixture()
def mixed_data(gridded_data, aggregated_data, region_attribution):
    return MixedData(
        gridded_data,
        aggregated_data, region_attribution
    )


# Now lets define a simple network over the gridded data
def my_network(input_grid: FullGriddedArray) -> FullGriddedArray:
    return (input_grid + 2.0 / 3)[..., PADDING:-PADDING, PADDING:-PADDING, :2]


def test_network_shape(gridded_data):
    tensor = get_grid_tensor(gridded_data)
    output = my_network(tensor)
    assert output.shape == (T, M - PADDING * 2, N - PADDING * 2, 2)


@pytest.fixture()
def output_grid(gridded_data: FullGriddedArray) -> np.ndarray:
    '''
    We make a fixture for the output grid
    '''
    shape = (T, M_i, N_i, 2)
    return np.random.random(prod(shape)).reshape(shape)


def pooling_function(gridded_data: FullGriddedArray,
                     attribution_matrix: AttributionMatrix):  # ((T, N, M, 2), (N, M)) -> (T, N, M, 2)
    '''
    For now we assume that the vector dimension after the network is two, representing ca mean and variance.
    Thus we can just sum these over the network
    '''
    return gridded_data * attribution_matrix[..., np.newaxis,:, :, np.newaxis]


def test_pooling_function(output_grid, region_attribution):
    region_parameters = pooling_function(output_grid, list(region_attribution.values())[0])
    assert region_parameters.shape == (T, M_i, N_i, 2)

