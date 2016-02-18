'''
temporary test with one parameter. Just to make sure
we can get mcmc to work.

'''
continue_chain = True
import os
import sys
import numpy as np
import emcee
from numpy.linalg import solve
from emcee.utils import MPIPool
import util
import data as Data
from hod_sim import HODsimulator
from group_richness import richness
from prior import PriorRange
import corner


def lnPost(theta, **kwargs):
    '''
    log Posterior
    '''
    fake_obs = kwargs['data']
    inv_cov = kwargs['inv_cov']
    observables = kwargs['observables']
    prior_range = kwargs['prior_range']
    prior_min = np.array(prior_range[:,0])
    prior_max = np.array(prior_range[:,1])

    # Prior
    if prior_min[0] < theta[0] < prior_max[0]:# and \
       #prior_min[1] < theta[1] < prior_max[1] and \
       #prior_min[2] < theta[2] < prior_max[2] and \
       #prior_min[3] < theta[3] < prior_max[3] and \
      # prior_min[4] < theta[4] < prior_max[4]:
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
        res = res_nbar
        ind += 1
    if 'gmf' in observables:
        res_gmf = fake_obs[ind] - model_obvs[ind]
        res = np.hstack((res, res_gmf))
        ind += 1
    if 'xi' in observables:
        res_xi = fake_obs[ind] - model_obvs[ind]
        res = np.hstack((res, res_xi))

    lnLike = - 0.5 * np.sum(np.dot(np.dot(res, inv_cov), res))

    return lnPrior + lnLike


def mcmc_mpi(Nwalkers, Nchains_burn,
             observables=['nbar', 'gmf'],
             data_dict={'Mr':21, 'Nmock':50},
             prior_name='first_try',
             output_dir=None):

    fake_obs = []
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
    - data_dict :
        dictionary that specifies the observation keywords
    '''
    '''
    # data observables
    fake_obs = []       # list of observables
    nbt = gmt = xit = None
    for obv in observables:
        if obv == 'nbar':
            data_nbar, data_nbar_var = Data.data_nbar(**data_dict)
            fake_obs.append(data_nbar)
            nbt = True
        if obv == 'gmf':
            data_gmf, data_gmf_sigma = Data.data_gmf(**data_dict)
            fake_obs.append(data_gmf)
            gmt = True
        if obv == 'xi':
            # import xir and full covariance matrix of xir
            data_xi, data_xi_cov = Data.data_xi_full_cov(**data_dict)
            fake_obs.append(data_xi)
            xit = True

    if nbt:
        if gmt:
            if xit:
                inv_cov = Data.data_nbar_gmf_xi_fullicov(**data_dict)
            else:
                inv_cov = Data.data_nbar_gmf_inv_cov(**data_dict)
        elif xit:
            inv_cov = Data.data_nbar_xi_inv_cov(**data_dict)
        else:
            inv_cov = 1. / data_nbar_var
    elif gmt:
        if xit:
            inv_cov = Data.data_gmf_xi_inv_cov(**data_dict)
        else:
            inv_cov = 1. / data_gmf_sigma ** 2
    elif xit:
        inv_cov = Data.data_xi_inv_cov(**data_dict)
    else:
        print "you must supply observables fool"
        assert(0)
    '''

    for obv in observables:
        if obv == 'xi':
            
            data_xi, data_xi_cov = Data.data_xi_full_cov(**data_dict)
            fake_obs.append(data_xi)
            inv_cov = Data.data_xi_inv_cov(**data_dict)
    print data_xi
    print inv_cov
     
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
    ##prior_min, prior_max = PriorRange(prior_name)
    prior_min, prior_max = np.array([0.9]) , np.array([1.1])

    prior_range = np.zeros((len(prior_min),2))
    prior_range[:,0] = prior_min
    prior_range[:,1] = prior_max

    # mcmc chain output file
    chain_file = ''.join([
        output_dir,
        util.observable_id_flag(observables),
        '_Mr', str(data_dict["Mr"]),
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
        random_guess = data_hod + 1e-4
        pos0 = np.repeat(random_guess, Nwalkers).reshape(Ndim, Nwalkers).T + \
                         1e-4 * np.random.randn(Ndim * Nwalkers).reshape(Nwalkers, Ndim)

    # Initializing MPIPool
    pool = MPIPool()
    if not pool.is_master():
        pool.wait()
        sys.exit(0)

    # Initializing the emcee sampler
    hod_kwargs = {
            'prior_range': prior_range,
            'data': fake_obs,
            'inv_cov': inv_cov,
            'observables': observables,
            'Mr': data_dict['Mr']
            }
    sampler = emcee.EnsembleSampler(Nwalkers, Ndim, lnPost,
                                    pool=pool, kwargs=hod_kwargs)

    for result in sampler.sample(pos0, iterations=Nchain,
                                 storechain=False):
        position = result[0]
        f = open(chain_file, 'a')
        for k in range(position.shape[0]):
            output_str = '\t'.join(position[k].astype('str')) + '\n'
            f.write(output_str)
        f.close()

    pool.close()


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
        f = open(chain_file, 'a')
        for k in range(position.shape[0]):
            output_str = '\t'.join(position[k].astype('str')) + '\n'
            f.write(output_str)
        f.close()

if __name__=="__main__":

    Niter = int(sys.argv[1])
    print 'N iterations = ', Niter
    Nwalkers = int(sys.argv[2])
    print 'N walkers = ', Nwalkers
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

    if len(sys.argv) > 4:
        out_dir = sys.argv[4]
        if out_dir[-1] != '/':
            out_dir += '/'
        print 'Output to ', out_dir
        mcmc_mpi(Nwalkers, Niter, observables=obv_list, output_dir=out_dir)
    else:
        mcmc_mpi(Nwalkers, Niter, observables=obv_list, output_dir=None)
