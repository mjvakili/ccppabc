'''

Module for ABC-PMC inference

Author(s): Chang, MJ


'''
import time
import pickle
import numpy as np

import abcpmc
from abcpmc import mpi_util

# --- Local --- 
import util
import data as Data
from hod_sim import HODsim
from prior import PriorRange
from group_richness import richness

# --- Plotting ---
from plotting import plot_thetas

def ABCpmc_HOD(T, eps_val, N_part=1000, prior_name='first_try', observables=['nbar', 'gmf'], data_dict={'Mr':20, 'Nmock': 500}):
    '''
    ABC-PMC implementation. 

    Parameters
    ----------
    - T : Number of iterations 
    - eps_val : 
    - N_part : Number of particles
    - observables : list of observables. Options are 'nbar', 'gmf', 'xi'
    - data_dict : dictionary that specifies the observation keywords 
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
            data_xi, data_cov_ii = Data.data_xi(**data_dict)
            fake_obs.append(data_xi)

    # True HOD parameters
    data_hod_dict = Data.data_hod_param(Mr=data_dict['Mr'])
    data_hod = np.array([
        data_hod_dict['logM0'],                 # log M0 
        np.log(data_hod_dict['sigma_logM']),    # log(sigma)
        data_hod_dict['logMmin'],               # log Mmin
        data_hod_dict['alpha'],                 # alpha
        data_hod_dict['logM1']                  # log M1
        ])
    
    # Priors
    prior_min, prior_max = PriorRange(prior_name)
    prior = abcpmc.TophatPrior(prior_min, prior_max)
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

    def multivariate_rho(datum, model): 
        dists = [] 
        for i_obv, obv in enumerate(observables): 
            if obv == 'nbar': 
                dist_nz = (datum[i_obv] - model[i_obv])**2. / data_nbar_var 
                dists.append(dist_nz)
            if obv == 'gmf': 
                dist_gr = np.sum((datum[i_obv] - model[i_obv])**2. / data_gmf_sigma**2.)
                dists.append(dist_gr)
            if obv == 'xi': 
                dist_xi = np.sum((datum[i_obv] - model[i_obv])**2. / data_cov_ii)
                dists.append(dist_xi)
        return np.array(dists)

    mpi_pool = mpi_util.MpiPool()
    abcpmc_sampler = abcpmc.Sampler(
            N=N_part,       # N_particles
            Y=fake_obs,         # data
            postfn=simz,    # simulator 
            dist=multivariate_rho,       # distance function  
            pool=mpi_pool)  
    abcpmc_sampler.particle_proposal_cls = abcpmc.ParticleProposal

    #eps = abcpmc.ConstEps(T, [1.e13,1.e13])
    eps = abcpmc.MultiConstEps(T, eps_val)
    pools = []
    for pool in abcpmc_sampler.sample(prior, eps):
        print("T:{0},ratio: {1:>.4f}".format(pool.t, pool.ratio))
        print eps(pool.t)
        # plot theta
        plot_thetas(pool.thetas, pool.ws , pool.t, 
                Mr=data_dict["Mr"], truths=data_hod, plot_range=prior_range, observables=observables)
        # write theta and w to file 
        theta_file = ''.join([util.dat_dir(), util.observable_id_flag(observables), 
            '_Mr', str(data_dict["Mr"]), '_theta_t', str(pool.t), '.dat'])
        w_file = ''.join([util.dat_dir(), util.observable_id_flag(observables), 
            '_Mr', str(data_dict["Mr"]), '_w_t', str(pool.t), '.dat'])
        np.savetxt(theta_file, pool.thetas)
        np.savetxt(w_file, pool.ws)

        if pool.t < 3: 
            eps.eps = np.percentile(np.atleast_2d(pool.dists), 50 , axis = 0)
        elif (pool.t > 2) and (pool.t < 20):
            eps.eps = np.percentile(np.atleast_2d(pool.dists), 75 , axis = 0)
            abcpmc_sampler.particle_proposal_cls = abcpmc.ParticleProposal
        else:
            eps.eps = np.percentile(np.atleast_2d(pool.dists), 90 , axis = 0)
            abcpmc_sampler.particle_proposal_cls = abcpmc.ParticleProposal
        #if eps.eps < eps_min:
        #    eps.eps = eps_min
            
        pools.append(pool)
    mpi_pool.close()
    #abcpmc_sampler.close()
    
    return pools

if __name__=="__main__": 
    ABCpmc_HOD(20, [1.e10,1.e10], N_part=100, observables=['nbar', 'xi'])
    ABCpmc_HOD(20, [1.e10,1.e10], N_part=100, observables=['nbar', 'xi', 'gmf'])
