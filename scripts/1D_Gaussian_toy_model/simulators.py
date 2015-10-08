import numpy as np
from scipy.stats import multivariate_normal


def test_simulator(params):

    means = params

    return multivariate_normal(mean=means).rvs(1000)
