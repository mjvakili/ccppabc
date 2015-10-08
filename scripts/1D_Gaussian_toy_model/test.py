import numpy as np
from scipy.stats import multivariate_normal
import sys
sys.path.insert(0, '../../ccppabc/')

from pmc_abc import ABC
from prior import Prior
from simulators import test_simulator
from distances import test_dist


thetadat = 0.0
testdat = multivariate_normal(mean=thetadat).rvs(8)

prior_dict = {
              "mean":
               {
                "shape": 'uniform',
                'min'  : -1.,
                'max'  :  1.,
               }
             }

test_prior = Prior(prior_dict)

eps = 0.2

test_abc = ABC(testdat, test_simulator, test_dist, test_prior, eps)


def run_serial():

    test_abc.basename = "test_abc_serial"
    test_abc.N_particles = 50
    print "testing ABC implementation in serial..."

    test_abc.run_abc()

    inferred_theta = np.loadtxt("{0}_19_thetas.dat".format(test_abc.basename))
    print "data val", "inferred val"
    print thetadat, inferred_theta

def run_parallel(N=4):

    test_abc.N_threads = N
    test_abc.basename = "test_abc_parallel"
    test_abc.N_particles = 50
    print "testing ABC implementation in parallel..."

    test_abc.run_abc()

    inferred_theta = np.loadtxt("{0}_19_thetas.dat".format(test_abc.basename))
    print "data val", "inferred val"
    print thetadat, inferred_theta


