'''


Module for standard MCMC inference

Author(s): Chang, MJ

Usage: 

   1- If you don't have a local installation of ipython parallel:
    
      ***pip install --user "ipython[parallel]"***
   
   2- Start up an ipcluster in another terminal:
 
      ***ipcluster start -n 4(number of cores)***

   3- Execute this file.
      

'''

import sys
import numpy as np
import emcee
from numpy.linalg import solve
from IPython.parallel import Client

# --- Local ---
import util
import data as Data
from hod_sim import HODsim
from hod_sim import HODsimulator 
from group_richness import richness
from prior import PriorRange
import corner

  
def lnPost(theta, **kwargs):
    '''log Posterior 
    '''
    fake_obs = kwargs['data']
    fake_obs_cov = kwargs['data_cov']
    kwargs.pop('data', None)
    kwargs.pop('data_cov', None)
    observables = kwargs['observables']
    prior_range = kwargs['prior_range']
    prior_min = prior_range[:,0]
    prior_max = prior_range[:,1]

    # Prior 
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

    # Likelihood
    model_obvs = HODsimulator(theta, **kwargs)
    ind = 0 
    if 'nbar' in observables: 
        res_nbar = fake_obs[ind] - model_obvs[ind] 
        ind += 1 
    if 'gmf' in observables: 
        res_gmf = fake_obs[ind] - model_obvs[ind]
        ind += 1
    if 'xi' in observables: 
        res_xi = fake_obs[ind] - model_obvs[ind]
    
    chi_tot = 0.
    ind = 0 
    if 'nbar' in observables: 
        chi_tot += -0.5*(res_nbar)**2. / fake_obs_cov[ind] 
        ind += 1
    if 'gmf' in observables: 
        raise NotImplementedError('GMF Likelihood has not yet been implemented')
    if 'xi' in observables: 
        chi_tot += -0.5*np.sum(np.dot(np.dot(res_xi , fake_obs_cov[ind]) , res_xi))
    lnLike = chi_tot

    return lnPrior + lnLike


def mcmc_ipython_par(Nwalkers, Nchains_burn, Nchains_pro, observables=['nbar', 'xi'], 
        data_dict={'Mr':20, 'Nmock':500}, prior_name = 'first_try', threads=1): 
    '''
    Standard MCMC implementaion
    

    Parameters
    -----------
    - Nwalker : 
        Number of walkers
    - Nchains_burn : 
        Number of burn-in chains
    - Nchains_pro : 
        Number of production chains   
    - observables : 
        list of observables. Options are 'nbar', 'gmf', 'xi'
    - data_dict : dictionary that specifies the observation keywords
    '''
    # data observables
    fake_obs = []       # list of observables 
    fake_obs_cov = [] 
    for obv in observables: 
        if obv == 'nbar': 
            data_nbar, data_nbar_var = Data.data_nbar(**data_dict)
            fake_obs.append(data_nbar)
            fake_obs_cov.append(data_nbar_var)
        if obv == 'gmf': 
            data_gmf, data_gmf_sigma = Data.data_gmf(**data_dict)
            fake_obs.append(data_gmf)
            fake_obs_cov.append(data_gmf)
        if obv == 'xi': 
            # import xir and full covariance matrix of xir
            data_xi, data_xi_cov = Data.data_xi_full_cov(**data_dict)   
            data_xi_invcov = Data.data_xi_inv_cov(**data_dict)
            fake_obs.append(data_xi)
            fake_obs_cov.append(data_xi_invcov)

    # True HOD parameters
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
    
    # Initializing Walkers 
    random_guess = np.array([11. , np.log(.4) , 11.5 , 1.0 , 13.5])
    pos0 = np.repeat(random_guess, Nwalkers).reshape(Ndim, Nwalkers).T + \
            1e-1 * np.random.randn(Ndim * Nwalkers).reshape(Nwalkers, Ndim)

    # Initializing the emcee sampler
    hod_kwargs = {
            'prior_range': prior_range, 
            'data': fake_obs, 
            'data_cov': fake_obs_cov, 
            'observables': observables, 
            'Mr': data_dict['Mr']
            }
    # Set up the interface to the ipcluster.
    c = Client()
    view = c[:]
    view.push({"lnPost": lnPost})
     
    # Modules necessary in posterior calculation should be called here
    view.execute("import numpy as np")
    view.execute("from hod_sim import HODsimulator")

    # Setting up the Sampler
    sampler = emcee.EnsembleSampler(Nwalkers, Ndim, lnPost, kwargs=hod_kwargs, pool = view)

    # Setting up a file for saving the chains
    chain_file = ''.join([util.dat_dir(), 
        util.observable_id_flag(observables), 
        '_Mr', str(data_dict["Mr"]), '_theta.mcmc_chain.dat'])
    f = open(chain_file, "w")
    f.close()

    # Running the Sampler and writing out the chains
    for result in sampler.sample(pos0, iterations=Nchains_burn + Nchains_pro, storechain=False):
        position = result[0]
        f = open(chain_file, "a")
        for k in range(position.shape[0]):
	    output_str = '\t'.join(position[k].astype('str')) + '\n'
            f.write(output_str)
        f.close()

if __name__=="__main__": 
    mcmc_ipython_par(20, 1, 1, observables=['nbar', 'xi'])
