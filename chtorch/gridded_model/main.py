import dataclasses
import numpy as np
import pytest
from math import prod

import scipy

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
M_i = M - PADDING * 2
N_i = N - PADDING * 2

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
    return {'oslo': np.array([[1., 1., 1., 0.],
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
    return (gridded_data * attribution_matrix[..., np.newaxis, :, :, np.newaxis]).sum(axis=(-2, -3))


def test_pooling_function(output_grid, region_attribution):
    region_parameters = pooling_function(output_grid, list(region_attribution.values())[0])
    assert region_parameters.shape == (T, 2)


# Now maybe we have some untransformed variables for the our distribution.
# If the outcome is normally distributed, we dont need any transformations as sums of  normal varibles are normal with
# sum of means and variances
# Lognormal would also be ok, but count data might be different. We want the aggregated mean be the sum of the means, but
# it is less simple how the variances are related. Thus what is remaining is both the transformation and the distribution.
# Taking
# Note that the means should probably be exped or softmaxed before summing

def get_parameters(raw_params: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return np.exp(raw_params[..., 0]), np.sqrt(np.exp(raw_params[..., 1]))


def log_pdf(parameters, observed):
    return scipy.stats.norm.logpdf(observed - parameters[0] / parameters[1])


def test_log_pdf(mixed_data):
    """
    This is what the whole pipeline would look like from the initial
    mixed data.
    """
    tensor = get_grid_tensor(mixed_data.gridded_data)
    output_grid = my_network(tensor)
    pooled = {region_name:
                  pooling_function(output_grid, attribution_matrix=attribution_matrix)
              for region_name, attribution_matrix in mixed_data.region_attribution.items()}
    log_pdfs = {region_name:
                    log_pdf(get_parameters(pooled_params), mixed_data.aggregated_data[region_name])
                for region_name, pooled_params in pooled.items()}
    assert len(log_pdfs) == 1
    assert next(iter(log_pdfs.values())).shape == (T,)


'''
What's missing now is an additional temporal aggregation step, which would need some temporal attribution
akin to the spatioal attribution.

We also have to figure out how the eta's for each pixel should be pooled in order to
as simply as possible represent the sum of different negative binomial variables.

As for networks, the feature dimension should be denesly connected while the
gemotric relation ship are more natural to get in terms convolutional connections.
A simple RNN should suffice for the temporal dimension.
'''
