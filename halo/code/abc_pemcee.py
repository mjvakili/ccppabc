import time
import numpy as np

import abcpmc
from abcpmc import mpi_util

# --- Local --- 
import data as Data
#from distance import rho 
from hod_sim import HODsim
from group_richness import richness

# --- Plotting ---
from plotting import plot_thetas

def ABCpmc(T, eps_val, N_part=1000, threads=1, observables=['nbar', 'gmf'], data_dict={'Mr':20, 'Nmock': 500}):
    '''
    ABC-PMC implementation. 

    Parameters
    ----------
    - T : Number of iterations 
    - eps_val : ???? (@mjv: what does this do?)
    - N_part : Number of particles
    - Threads : Number of MPI threads (not sure if this keyword actually works or not 
    - observables : list of observables. Options are 'nbar', 'gmf', 'xi'
    - data_dict : dictionary that specifies the observation keywords 
    '''
    # data observables
    data = []       # list of observables 
    for obv in observables: 
        if obv == 'nbar': 
            data_nbar, data_nbar_var = Data.data_nbar(**data_dict)
            data.append(data_nbar)
        if obv == 'gmf': 
            data_gmf, data_gmf_sigma = Data.data_gmf(**data_dict)
            data.append(data_gmf)
        if obv == 'xi': 
            data_xi, data_cov_ii = Data.data_xi(**data_dict)
            data.append(data_xi)
    
    # True HOD parameters
    if Mr == 20: 
        data_hod = np.array([11.38 , np.log(0.26) , 12.02 , 1.06 , 13.31])
    else: 
        raise NotImplementedError
    
    # Priors
    prior_min = [10., np.log(0.1), 11.02, 0.8, 13.]
    prior_max = [13., np.log(0.7), 13.02, 1.3, 14.]
    prior = abcpmc.TophatPrior(prior_min, prior_max)
    prior_range = np.zeros((len(prior_min),2))
    prior_range[:,0] = prior_min
    prior_range[:,1] = prior_max

    # simulator
    our_model = HODsim()    # initialize model
    kwargs = {'prior_range': prior_range, 'observables': observables}
    def simz(tt): 
        return our_model.sum_stat(tt, **kwargs)

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
            Y=data,         # data
            postfn=simz,    # simulator 
            dist=multivariate_rho,       # distance function  
            threads=threads,
            pool=mpi_pool)  
    abcpmc_sampler.particle_proposal_cls = abcpmc.ParticleProposal

    #eps = abcpmc.ConstEps(T, [1.e13,1.e13])
    eps = abcpmc.ConstEps(T, eps_val)       # (@mjv: what does this do?)
    pools = []
    for pool in abcpmc_sampler.sample(prior, eps):
        print("T:{0},ratio: {1:>.4f}".format(pool.t, pool.ratio))
        print eps(pool.t)
        # plot theta
        plot_thetas(pool.thetas, pool.ws , pool.t, 
                Mr=Mr, truths=data_hod, plot_range=prior_range, observables=observables)
        # write theta and w to file 
        theta_file = ''.join([util.dat_dir(), util.observable_id_flag(observables), 
            '_Mr', str(Mr), '_theta_t', str(pool.t), '.dat'])
        w_file = ''.join([util.dat_dir(), util.observable_id_flag(observables), 
            '_Mr', str(Mr), '_w_t', str(pool.t), '.dat'])
        np.savetxt(theta_file, pooltheta)
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
    #abcpmc_sampler.close()
    
    return pools

if __name__=="__main__": 
    ABCpmc(10, 60, N_part=100)
