'''


Module for standard MCMC inference

Author(s): Chang, MJ


'''
import sys
import numpy as np
import emcee

from numpy.linalg import solve
from emcee.utils import MPIPool

# --- Local ---
import data as Data
from hod_sim import HODsim
from group_richness import richness

"""data, covariance matrix, and the inverse covariance """

xir_data = np.loadtxt("xir_Mr20.dat")
covariance = np.loadtxt("clustering_covariance_Mr20.dat")
N_mocks = 500
N_bins  = len(xir_data)
inv_c =  solve(np.eye(len(xir_data)) , covariance)*(N_mocks - 2 - N_bins)/(N_mocks - 1)



"""likelihood"""

def lnlike(theta):
    
    model = Zheng07(threshold = -20.)
    model.param_dict['logM0'] = theta[0]
    model.param_dict['sigma_logM'] = np.exp(theta[1])
    model.param_dict['logMmin'] = theta[2]     
    model.param_dict['alpha'] = theta[3]
    model.param_dict['logM1'] = theta[4]
    model.populate_mock()
    r , xir_model = model.mock.compute_galaxy_clustering()
    res = xir_data - xir_model
    return -0.5*np.sum(np.dot(np.dot(res , inv_c) , res))

"""prior"""

def lnprior(theta):
    a , b , c , d , e = theta
    if 10. < a < 13. and np.log(.1) < b < np.log(.7) and  11.02< c < 13.02 and .8< d < 1.3 and 13.< e < 14.:
        return 0.0
    return -np.inf

"""posterior"""

def lnprob(theta):
    lp = lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(theta)

"""Initializing Walkers"""

ndim, nwalkers = 5, 100
pos = [np.array([11.5 , np.log(.4) , 12.02 , 1.03 , 13.5]) + 1e-4*np.random.randn(ndim) for i in range(nwalkers)]

"""Initializing MPIPool"""

pool = MPIPool()
if not pool.is_master():
    pool.wait()
    sys.exit(0)

sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob , pool = pool)

sampler.run_mcmc(pos, 500)

pool.close()

"""
samples = sampler.chain[:, 50:, :].reshape((-1, ndim))

# Burn in.
pos, _, _ = sampler.run_mcmc(pos, 1000)
sampler.reset()

# Production.
pos, _, _ = sampler.run_mcmc(pos, 4000)
"""
