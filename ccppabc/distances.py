import numpy as np


def test_dist(data, model):
    """
    The test case expects a set of samples from a gaussian.
    The distance is determined by the magnitude of the distance
    between the measured means of the data samples and model samples.
    """

    return np.abs(np.mean(data) - np.mean(model))

