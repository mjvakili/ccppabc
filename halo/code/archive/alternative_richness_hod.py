'''

Documentation here

Authors : MJ, Chang

'''
import time
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend("Agg")
from halotools.empirical_models import Zheng07

import corner

from scipy.stats import norm , gamma 
from scipy.stats import multivariate_normal

import abcpmc
from abcpmc import mpi_util

# --- Local --- 
from group_richness import richness
from distance import rho 

"""Load data and variance"""
data_gmf = np.loadtxt("gmf_Mr20.dat")
sigma = np.loadtxt("gmf_noise_Mr20_2.dat")
bins = np.loadtxt("gmf_bins_Mr20_2.dat")

mock_nbar = np.loadtxt("nbar_Mr20.dat")
data_nbar = np.mean(mock_nbar)
covar_nz = np.var(mock_nbar)

data = [data_nbar , data_gmf]

"""True HOD parameters"""

data_hod = np.array([11.38 , np.log(0.26) , 12.02 , 1.06 , 13.31])

"""Prior"""

#prior = abcpmc.TophatPrior([11.91,.38,12.78,1.1,13.9],[11.92,.4,12.8,1.2,14.])

prior_dict = {
    'logM0'  : {'shape': 'uniform', 'min': 10.  ,  'max': 13.},
    'sigma_logM': {'shape': 'uniform', 'min': np.log(.1) ,  'max': np.log(.7)},
    'logMmin': {'shape': 'uniform', 'min': 11.02,  'max': 13.02},   
    'alpha': {'shape': 'uniform', 'min': .8 ,  'max': 1.3},
    'logM1'  : {'shape': 'uniform', 'min': 13.  ,  'max': 14.},
}

"""Plot range"""

plot_range = []
for key in ['logM0', 'sigma_logM', 'logMmin','alpha','logM1']: 
	plot_range.append([prior_dict[key]['min'], prior_dict[key]['max']])
prior_range = np.array(plot_range)
print prior_range[:,0] , prior_range[:,1]

ourmodel = HODsim()
simz = ourmodel.sum_stat



mpi_pool = mpi_util.MpiPool()
def sample(T, eps_val, eps_min):
    prior = abcpmc.TophatPrior([10.,np.log(.1),11.02,.8,13.],[13.,np.log(.7),13.02,1.3,14.])

    abcpmc_sampler = abcpmc.Sampler(N = 1000, Y=data, postfn=simz, dist=distance, pool=mpi_pool)
    abcpmc_sampler.particle_proposal_cls = abcpmc.ParticleProposal
    #abcpmc.Sampler.particle_proposal_kwargs = {'k': 50}
    #abcpmc_sampler.particle_proposal_cls = abcpmc.KNNParticleProposal

    eps = abcpmc.ConstEps(T, [1.e13,1.e13])
    pools = []
    for pool in abcpmc_sampler.sample(prior, eps):
        print("T:{0},ratio: {1:>.4f}".format(pool.t, pool.ratio))
        print eps(pool.t)

        plot_thetas(pool.thetas , pool.ws , pool.t)
        np.savetxt("/home/mj/public_html/nbar_gmf5_Mr20_theta_t"+str(t)+".dat" , theta)
        np.savetxt("/home/mj/public_html/nbar_gmf5_Mr20_w_t"+str(t)+".dat" , w)
        if pool.t<3: 
            eps.eps = np.percentile(np.atleast_2d(pool.dists), 50 , axis = 0)
        elif (pool.t>2)and(pool.t<20):
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

T = 40
eps = 60
pools = sample(T, eps, 5.)
