import numpy as np
import matplotlib.pyplot as plt
import time
plt.switch_backend("Agg")
from halotools.empirical_models import Zheng07 , model_defaults
from astropy.table import Table
import corner
from scipy.stats import norm , gamma 
from scipy.stats import multivariate_normal
from scipy.spatial import cKDTree
import abcpmc
import seaborn as sns
sns.set_style("white")
np.random.seed()
from abcpmc import mpi_util
from scipy.stats import ks_2samp
from halotools.sim_manager import supported_sims
from halotools.empirical_models.mock_helpers import (three_dim_pos_bundle,
                                                     infer_mask_from_kwargs)
from halotools.mock_observables.clustering import wp


cat = supported_sims.HaloCatalog()
L = cat.Lbox
rbins = model_defaults.default_rbins
pi_bins = np.linspace(0,125,200)
model = Zheng07(threshold = -21.)
print 'Data HOD Parameters ', model.param_dict

"""data and covariance"""
mock_nbar = np.loadtxt("mock_nbar.dat")
data_nbar = np.mean(mock_nbar)
mocks_wp = np.loadtxt("wps.dat")
data_wp = np.mean(mocks_wp , axis = 0)

data = [data_nbar , data_wp]

covariance = np.loadtxt("wp_covariance.dat")
cii = np.diag(covariance)
covar_nz = np.var(mock_nbar)

"""list of true parameters"""

data_hod = np.array([11.92 , 0.39 , 12.79 , 1.15 , 13.94])


"""Prior"""

prior = abcpmc.TophatPrior([9.,.1,12.5,.9,13.6],[15.,1.,13.09,1.45,14.25])
prior_dict = {
 
    'logM0'  : {'shape': 'uniform', 'min': 9.  ,  'max': 15.},
    'sigma_logM': {'shape': 'uniform', 'min': 0. ,  'max': 1.},
    'logMmin': {'shape': 'uniform', 'min': 12.5,  'max': 13.09},   
    'alpha': {'shape': 'uniform', 'min': .9 ,  'max': 1.45},
    'logM1'  : {'shape': 'uniform', 'min': 13.6  ,  'max': 14.25},
}

"""Plot range"""

plot_range = []
for key in ['logM0', 'sigma_logM', 'logMmin','alpha','logM1']: 
	plot_range.append([prior_dict[key]['min'], prior_dict[key]['max']])
prior_range = np.array(plot_range)

"""simulator"""

class HODsim(object): 
    
    def __init__(self): 
        self.model = Zheng07(threshold = -21.)
    
    def sum_stat(self, theta):
        
        self.model.param_dict['alpha'] = theta[3]
        self.model.param_dict['logMmin'] = theta[2]     
        self.model.param_dict['sigma_logM'] = theta[1]
        self.model.param_dict['logM0'] = theta[0]
        self.model.param_dict['logM1'] = theta[4]
        if np.all((prior_range[:,0] < theta)&(theta < prior_range[:,1])):
          try:
            self.model.populate_mock()
	    nbar = self.model.mock.number_density
            pos = three_dim_pos_bundle(table=self.model.mock.galaxy_table,
                               key1='x', key2='y', key3='z')
            w_p = wp(pos, rbins, pi_bins , period = np.array([L,L,L]))
            return [nbar , w_p]
          except ValueError:
            return [10. , np.zeros(14)]
        else:
            return [10. , np.zeros(14)]

ourmodel = HODsim()
simz = ourmodel.sum_stat

"""distance"""

def distance(data, model, type = 'multiple distance'): 
    
    if type == 'added distance': 
        dist_nz = np.abs(d_data[0] - d_model[0])/d_data[0]
        dist_xi = np.sum(np.abs(d_data[1] - d_model[1])/d_data[1])
        dist = dist_nz + dist_xi 

    elif type == 'multiple distance':
        
        dist_nbar = (data[0] - model[0])**2. / covar_nz
        dist_xi = np.sum((data[1] - model[1])**2. / cii)
        dist = [dist_nbar , dist_xi]
        #print dist
    elif type == 'group distance':

        #dist_nz = (d_data[0] - d_model[0])**2. / covar_nz
        dist = ks_2samp(d_data , d_model)[0]**2. 
        #dist = np.array([dist_nz , dist_ri])
        
    return dist





def plot_thetas(theta , w , t): 
    fig = corner.corner(
        theta, weights = w.flatten() , truths= data_hod,
        truth_color="red", plot_datapoints=True, fill_contours=False, levels=[0.68], 
                color='k', bins=25, smooth= True, 
        range=plot_range, 
        labels=[r"$\log M_{0}$", r"$\sigma_{log M}$", r"$\log M_{min}$" , r"$\alpha$" , r"$\log M_{1}$" ]
        )
    
    plt.savefig("/home/mj/public_html/nbar_wp_v1_t"+str(t)+".png")
    plt.close()
    fig = corner.corner(
        theta , truths= data_hod,
        truth_color="red", plot_datapoints=True, fill_contours=False, levels=[0.68],
                color='k', bins=25, smooth= True,
        range=plot_range,
        labels=[r"$\log M_{0}$", r"$\sigma_{log M}$", r"$\log M_{min}$" , r"$\alpha$" , r"$\log M_{1}$" ]
        )

    plt.savefig("/home/mj/public_html/nbar_wp_v1_now_t"+str(t)+".png")
    plt.close()
    np.savetxt("/home/mj/public_html/nbar_wp_v1_theta_t"+str(t)+".dat" , theta)
    np.savetxt("/home/mj/public_html/nbar_wp_v1_w_t"+str(t)+".dat" , w)


mpi_pool = mpi_util.MpiPool()
def sample(T, eps_val, eps_min):

    abcpmc_sampler = abcpmc.Sampler(N=100, Y=data, postfn=simz, dist=distance, pool=mpi_pool)
    abcpmc_sampler.particle_proposal_cls = abcpmc.ParticleProposal
    eps = abcpmc.MultiConstEps(T , [1.e6 , 1.e6])
    #eps = abcpmc.MultiExponentialEps(T,[1.e41 , 1.e12] , [eps_min , eps_min])
    pools = []
    for pool in abcpmc_sampler.sample(prior, eps):
        print("T: {0}, ratio: {1:>.4f}".format(pool.t, pool.ratio))
        print eps(pool.t)
        
        plot_thetas(pool.thetas , pool.ws, pool.t)
        
        if (pool.t < 7):
            eps.eps = np.median(np.atleast_2d(pool.dists), axis = 0)
        #elif (pool.t < 3):
        #    eps.eps = np.percentile(np.atleast_2d(pool.dists), 60 , axis = 0)
        else:
            #abcpmc_sampler.particle_proposal_cls = abcpmc.ParticleProposal
            eps.eps = np.median(np.atleast_2d(pool.dists), axis = 0)
        #for i in xrange(len(eps.eps)):
        #    if eps.eps[i] < eps_min[i]:
        #        eps.eps[i] = eps_min[i]
            
        pools.append(pool)
        
    #abcpmc_sampler.close()
    
    return pools

T=40
eps=1.e9
pools = sample(T, eps, [.9, 13.])

