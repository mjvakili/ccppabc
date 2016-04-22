'''
    
Standard MCMC implementaion

Author(s): Chang, MJ


- Nwalker : Number of walkers
- Nchains_burn : Number of burn-in chains
- Nchains_pro : Number of production chains   
- ndim    : Dimensionality of the parameter space
- observables : list of observables. Options are 'nbar', 'gmf', 'xi'
-data_dict : dictionary that specifies the observation keywords

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


Nwalkers = 10
Nchains_burn = 1
Nchains_pro = 1
    
# data observables
fake_obs = []       # list of observables 
for obv in observables: 
    if obv == 'nbar': 
        data_nbar, data_nbar_var = Data.data_nbar(**data_dict)
        fake_obs.append(data_nbar)
    if obv == 'gmf': 
        data_gmf, data_gmf_sigma = Data.data_gmf(**data_dict)
        fake_obs.append(data_gmf)
    if obv == 'xi': 
        # import xir and full covariance matrix of xir 
        data_xi, data_xi_cov = Data.data_xi_full_cov(**data_dict)           
        # take the inverse of the covariance matrix
        data_xi_invcov = solve(np.eye(len(data_xi)) , data_xi_cov)          
        fake_obs.append(data_xi)

# 'True' HOD parameters
data_hod_dict = Data.data_hod_param(Mr=data_dict['Mr'])
data_hod = np.array([
    data_hod_dict['logM0'],                 # log M0 
    np.log(data_hod_dict['sigma_logM']),    # log(sigma)
    data_hod_dict['logMmin'],               # log Mmin
    data_hod_dict['alpha'],                 # alpha
    data_hod_dict['logM1']                  # log M1
    ])
Ndim = len(data_hod)

# Priors
prior_min, prior_max = PriorRange(prior_name)
prior_range = np.zeros((len(prior_min),2))
prior_range[:,0] = prior_min
prior_range[:,1] = prior_max

def lnPost(theta):
    '''log-posterior
    '''
    # prior calculations 
    if prior_min[0] < theta[0] < prior_max[0] and \
       prior_min[1] < theta[1] < prior_max[1] and \
       prior_min[2] < theta[2] < prior_max[2] and \
       prior_min[3] < theta[3] < prior_max[3] and \
       prior_min[4] < theta[4] < prior_max[4]:
           lnPrior = 0.0
    else:
        lnPrior = -np.inf

    if not np.isfinite(lnPrior):
        return -np.inf
    return lnPrior + lnLike(theta)



    """Initializing Walkers"""

    pos = [np.array([11. , np.log(.4) , 11.5 , 1.0 , 13.5]) + 1e-3*np.random.randn(Ndim) for i in range(Nwalkers)]

    """Initializing MPIPool"""

    pool = MPIPool(loadbalance=True)
    if not pool.is_master():
       pool.wait()
       sys.exit(0)

    """Initializing the emcee sampler"""
    sampler = emcee.EnsembleSampler(Nwalkers, Ndim, lnprob, pool=pool)
    
    # Burn in + Production
    sampler.run_mcmc(pos, Nchains_burn + Nchains_pro)

    # Production.
    samples = sampler.chain[:, Nchains_burn:, :].reshape((-1, Ndim))
    #closing the pool 
    pool.close()
    
    np.savetxt("mcmc_sample.dat" , samples)
    
if __name__=="__main__": 
    
    mcmc(20,10, 10, 5, observables=['nbar', 'xi'], data_dict={'Mr':20, 'Nmock':500})

