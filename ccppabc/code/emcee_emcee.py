import os 
import sys
import numpy as np
import emcee
from numpy.linalg import solve
from emcee.utils import MPIPool
import scipy.optimize as op

# --- Local ---
import util
import data as Data
from hod_sim import MCMC_HODsim 
from group_richness import richness
from prior import PriorRange
import corner
from numpy.linalg import solve


def lnPost(theta, **kwargs):
    def lnprior(theta, **kwargs):
        '''log prior 
        '''
        fake_obs = kwargs['data']
    	fake_obs_icov = kwargs['data_icov']
    	kwargs.pop('data', None)
    	kwargs.pop('data_icov', None)
    	observables = kwargs['observables']
    	prior_range = kwargs['prior_range']
    	prior_min = prior_range[:,0]
    	prior_max = prior_range[:,1]
        #print prior_range
    	# Prior 
        if prior_min[0] < theta[0] < prior_max[0] and \
       	    prior_min[1] < theta[1] < prior_max[1] and \
            prior_min[2] < theta[2] < prior_max[2] and \
            prior_min[3] < theta[3] < prior_max[3] and \
            prior_min[4] < theta[4] < prior_max[4]:
                return 0
    
        else:
		return -np.inf

    def lnlike(theta, **kwargs):

    	fake_obs = kwargs['data']
    	fake_obs_icov = kwargs['data_icov']
    	kwargs.pop('data', None)
    	kwargs.pop('data_icov', None)
    	observables = kwargs['observables']
    	prior_range = kwargs['prior_range']
    	# Likelihood
    	model_obvs = generator(theta, prior_range , observables)
        #print "model=" , model_obvs
    	if observables == ['xi']:
            res = fake_obs - model_obvs[0]
 	    nbin = len(res)
            f = (124. - 2. - nbin)/(124. - 1.)
    	if observables == ['nbar','xi']:
            #print model_obvs[1] , fake_obs[1:]
            res = fake_obs - np.hstack([model_obvs[0], model_obvs[1]])
            nbin = len(res)
            f = (124. - 2. - nbin)/(124. - 1.)
    	if observables == ['nbar', 'gmf']:
            # omitting the first gmf bin!!!!
            res = fake_obs - np.hstack([model_obvs[0], model_obvs[1][1:]])
	    nbin = len(res)
            f = (124. - 2. - nbin)/(124. - 1.)
        #neg_chi_tot = - 0.5 * np.sum(np.dot(res , np.dot(fake_obs_icov , res)))
        neg_chi_tot = - 0.5 * f * np.sum(np.dot(res , solve(fake_obs_icov , res)))
        #print neg_chi_tot
    	return neg_chi_tot

    lp = lnprior(theta , **kwargs)
    if not np.isfinite(lp):
        return -np.inf
    #print lp + lnlike(theta , **kwargs)
    return lp + lnlike(theta, **kwargs)


def mcmc_mpi(Nwalkers, Nchains, observables=['nbar', 'xi'], 
        data_dict={'Mr':21, 'b_normal': 0.25}, prior_name = 'first_try', mcmcrun=None): 
    '''
    Standard MCMC implementaion
    
    Parameters
    -----------
    - Nwalker : 
        Number of walkers
    - Nchains : 
        Number of MCMC chains   
    - observables : 
        list of observables. Options are: ['nbar','xi'],['nbar','gmf'],['xi']
    - data_dict : dictionary that specifies the observation keywords
    '''
    #Initializing the vector of observables and inverse covariance matrix
    if observables == ['xi']:
        fake_obs = Data.data_xi(**data_dict)
        #fake_obs_icov = Data.data_inv_cov('xi', **data_dict)
        fake_obs_icov = Data.data_cov(inference='mcmc', **data_dict)[1:16 , 1:16]
    if observables == ['nbar','xi']:
        fake_obs = np.hstack([Data.data_nbar(**data_dict), Data.data_xi(**data_dict)])
        fake_obs_icov = Data.data_cov(inference='mcmc', **data_dict)[:16 , :16]
    if observables == ['nbar','gmf']:
        ##### FIRST BIN OF GMF DROPPED ###############
        # CAUTION: hardcoded 
        fake_obs = np.hstack([Data.data_nbar(**data_dict), Data.data_gmf(**data_dict)[1:]])
        fake_obs_icov = np.zeros((10,10))
        #print Data.data_cov(**data_dict)[17: , 17:].shape
        
        # Covariance matrix being adjusted accordingly 
        fake_obs_icov[1:,1:] = Data.data_cov(inference='mcmc', **data_dict)[17: , 17:]
        fake_obs_icov[0,1:] = Data.data_cov(inference='mcmc', **data_dict)[0 , 17:]
        fake_obs_icov[1:,0] = Data.data_cov(inference='mcmc', **data_dict)[17: , 0]
        fake_obs_icov[0,0] = Data.data_cov(inference='mcmc', **data_dict)[0 , 0]

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
    
    # mcmc chain output file 
    chain_file = ''.join([util.mcmc_dir(),
        util.observable_id_flag(observables), 
        '.', mcmcrun, 
        '.mcmc_chain.dat'
        ])
    #print chain_file

    if os.path.isfile(chain_file) and continue_chain:   
        print 'Continuing previous MCMC chain!'
        sample = np.loadtxt(chain_file) 
        Nchain = Niter - (len(sample) / Nwalkers) # Number of chains left to finish 
        if Nchain > 0: 
            pass
        else: 
            raise ValueError
        print Nchain, ' iterations left to finish'

        # Initializing Walkers from the end of the chain 
        pos0 = sample[-Nwalkers:]
    else:
        # new chain 
        f = open(chain_file, 'w')
        f.close()
        Nchain = Niter
         
        # Initializing Walkers
        random_guess = data_hod
        pos0 = np.repeat(random_guess, Nwalkers).reshape(Ndim, Nwalkers).T + \
                         5.e-2 * np.random.randn(Ndim * Nwalkers).reshape(Nwalkers, Ndim)
        print pos0.shape
    # Initializing MPIPool
    pool = MPIPool()
    if not pool.is_master():
        pool.wait()
        sys.exit(0)

    # Initializing the emcee sampler
    hod_kwargs = {
            'prior_range': prior_range, 
            'data': fake_obs, 
            'data_icov': fake_obs_icov, 
            'observables': observables, 
            'Mr': data_dict['Mr']
            }
    sampler = emcee.EnsembleSampler(Nwalkers, Ndim, lnPost, pool=pool, kwargs=hod_kwargs)

    # Initializing Walkers 
    for result in sampler.sample(pos0, iterations=Nchain, storechain=False):
        position = result[0]
        #print position
        f = open(chain_file, 'a')
        for k in range(position.shape[0]): 
            output_str = '\t'.join(position[k].astype('str')) + '\n'
            f.write(output_str)
        f.close()

    pool.close()


if __name__=="__main__": 
    generator = MCMC_HODsim(Mr = 21, b_normal=0.25)
    continue_chain = False
    Nwalkers = int(sys.argv[1])
    print 'N walkers = ', Nwalkers
    Niter = int(sys.argv[2])
    print 'N iterations = ', Niter
    obv_flag = sys.argv[3]
    if obv_flag == 'nbarxi':
        obv_list = ['nbar', 'xi']
    elif obv_flag == 'nbargmf':
        obv_list = ['nbar', 'gmf']
    elif obv_flag == 'xi':
        obv_list = ['xi']
    else:
        raise ValueError
    print 'Observables: ', ', '.join(obv_list)
    mcmc_name = sys.argv[4]
    print 'MCMC name ', mcmc_name 
    mcmc_mpi(Nwalkers, Niter, observables=obv_list, mcmcrun=mcmc_name)# , continue_chain = False)



"""
DEFUNCT MCMC 
    def mcmc_multi(Nwalkers, Niter, observables=['nbar', 'xi'], 
            data_dict={'Mr':20, 'Nmock':500}, prior_name = 'first_try', 
            threads=1, continue_chain=False): 
        '''
        Standard MCMC implementaion
        

        Parameters
        -----------
        - Nwalker : 
            Number of walkers
        - Niter : 
            Number of chain iterations 
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
                fake_obs_cov.append(data_gmf_sigma**2.)
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
        
        # mcmc chain output file 
        chain_file = ''.join([
            util.dat_dir(), 
            util.observable_id_flag(observables), 
            '_Mr', str(data_dict["Mr"]), 
            '_theta.Niter', str(Niter), 
            '.mcmc_chain.dat'
            ])

        if os.path.isfile(chain_file) and continue_chain:   
            print 'Continuing previous MCMC chain!'
            sample = np.loadtxt(chain_file) 
            Nchain = Niter - (len(sample) / Nwalkers) # Number of chains left to finish 
            if Nchain > 0: 
                pass
            else: 
                raise ValueError
            print Nchain, ' iterations left to finish'

            # Initializing Walkers from the end of the chain 
            pos0 = sample[-Nwalkers:]
        else: 
            # new chain 
            f = open(chain_file, 'w')
            f.close()
            Nchain = Niter
             
            # Initializing Walkers 
            random_guess = np.array([11. , np.log(.4) , 11.5 , 1.0 , 13.5])
            pos0 = np.repeat(random_guess, Nwalkers).reshape(Ndim, Nwalkers).T + \
                    1e-1 * np.random.randn(Ndim * Nwalkers).reshape(Nwalkers, Ndim)
            print pos0.shape
        # Initializing the emcee sampler
        hod_kwargs = {
                'prior_range': prior_range, 
                'data': fake_obs, 
                'data_cov': fake_obs_cov, 
                'observables': observables, 
                'Mr': data_dict['Mr']
                }
        sampler = emcee.EnsembleSampler(Nwalkers, Ndim, lnPost, kwargs=hod_kwargs, threads=threads)
        
        for result in sampler.sample(pos0, iterations=Nchain, storechain=False):
            position = result[0]
            print "position=" , position
            f = open(chain_file, 'a')
            for k in range(position.shape[0]): 
                output_str = '\t'.join(position[k].astype('str')) + '\n'
                f.write(output_str)
            f.close()
"""
