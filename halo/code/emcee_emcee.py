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


def mcmc(Nwalkers, Nchains_burn, Nchains_pro , Ndim, observables=['nbar', 'xi'], data_dict={'Mr':20, 'Nmock':500}): 
    '''
    Standard MCMC implementaion
    

    Parameters
    -----------
    - Nwalker : Number of walkers
    - Nchains_burn : Number of burn-in chains
    - Nchains_pro : Number of production chains   
    - ndim    : Dimensionality of the parameter space
    - observables : list of observables. Options are 'nbar', 'gmf', 'xi'
    -data_dict : dictionary that specifies the observation keywords
    '''
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
            data_xi, data_xi_cov = Data.data_xi_full_cov(**data_dict)   # import xir and full covariance matrix of xir
            data_xi_invcov = solve(np.eye(len(data_xi)) , data_xi_cov)  # take the inverse of the covariance matrix
            fake_obs.append(data_xi)

    # True HOD parameters
    if data_dict["Mr"] == 20: 
        data_hod = np.array([11.38 , np.log(0.26) , 12.02 , 1.06 , 13.31])
    else: 
        raise NotImplementedError
    
    # Priors
    prior_min = [10., np.log(0.1), 11.02, 0.8, 13.]
    prior_max = [13., np.log(0.7), 13.02, 1.3, 14.]
    prior_range = np.zeros((len(prior_min),2))
    prior_range[:,0] = prior_min
    prior_range[:,1] = prior_max


    # simulator
    our_model = HODsim()    # initialize model
    kwargs = {'prior_range': prior_range, 'observables': observables}
    def simz(tt): 
        sim = our_model.sum_stat(tt, **kwargs)
        if sim is None: 
            print 'Simulator is giving NoneType.'
            pickle.dump(tt, open("simz_crash_theta.p", 'wb'))
            print 'The input parameters are', tt
            pickle.dump(kwargs, open('simz_crash_kwargs.p', 'wb'))
            print 'The kwargs are', kwargs
            raise ValueError
        return sim
    
    def lnlike(theta):

        """log-likelihood without the term -.5*log(det(cov))"""

        nbar_model , xi_model = simz(theta)
        res_nbar = fake_obs[0] - nbar_model
        res_xi   = fake_obs[1] - xi_model
	chi_nbar = -0.5*(res_nbar)**2. / data_nbar_var 
        chi_xi   = -0.5*np.sum(np.dot(np.dot(res_xi , data_xi_invcov) , res_xi))

        return chi_nbar + chi_xi


    def lnprior(theta):

        """log-prior"""

        a , b , c , d , e = theta
        if prior_min[0] < a < prior_max[0] and \
           prior_min[1] < b < prior_max[1] and \
           prior_min[2] < c < prior_max[2] and \
           prior_min[3] < d < prior_max[3] and \
           prior_min[4] < e < prior_max[4]:
            return 0.0
        else:
            return -np.inf
    
    def lnprob(theta):
 
        """log-posterior"""
        
        lp = lnprior(theta)
        if not np.isfinite(lp):
           return -np.inf
        return lp + lnlike(theta)



    """Initializing Walkers"""

    ndim, nwalkers = 5, 100
    pos = [np.array([11.5 , np.log(.4) , 12.02 , 1.03 , 13.5]) + 1e-6*np.random.randn(Ndim) for i in range(Nwalkers)]

    """Initializing MPIPool"""

    #pool = MPIPool()
    #if not pool.is_master():
    #   pool.wait()
    #   sys.exit(0)

    """Initializing the emcee sampler"""

    sampler = emcee.EnsembleSampler(Nwalkers, Ndim, lnprob)# , pool = pool)

    # Burn in.
    pos, _, _ = sampler.run_mcmc(pos, Nchains_burn)
    sampler.reset()

    # Production.
    pos, _, _ = sampler.run_mcmc(pos, Nchains_pro)
    
    #closing the pool 
    #pool.close()

    #saving the mcmc samples
    np.savetxt("mcmc_sample.dat" , sampler.flat_chain)

if __name__=="__main__": 
    
    mcmc(10, 10, 10, 5, observables=['nbar', 'xi'], data_dict={'Mr':20, 'Nmock':500})

