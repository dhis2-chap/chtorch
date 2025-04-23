"""
The outputs from the model (eta) will be pretty standard normal on an untrained network.
We want the transformations from eta to distribution parameters to be pretty good for such a network.
There might be troubles around 0.

A goal should be that the network converges quickly to the theoretical minimum on a single batch
We also want the network to be able to learn the distribution parameters for a wide range of distributions.
It might be good that the over-dispersion are 0 to begin with.

"""
