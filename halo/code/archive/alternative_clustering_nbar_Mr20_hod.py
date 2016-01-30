import numpy as np
import matplotlib.pyplot as plt
import time
plt.switch_backend("Agg")
from halotools.empirical_models import Zheng07
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


model = Zheng07(threshold = -20.)
print 'Data HOD Parameters ', model.param_dict

"""data and covariance"""
#We shouldn't treat a mean of mocks as a data!

mock_nbar = np.loadtxt("nbar_Mr20.dat")
#data_nbar = np.mean(mock_nbar)
#mocks_xir = np.loadtxt("xir.dat")
#data_xir = np.mean(mocks_xir , axis = 0)

model.populate_mock()
data_nbar = model.mock.number_density
data_xir  = model.mock.compute_galaxy_clustering()[1]
np.savetxt("xir_Mr20.dat" , data_xir)
data = [data_nbar , data_xir]

covariance = np.loadtxt("clustering_covariance_Mr20.dat")
cii = np.diag(covariance)
covar_nbar = np.var(mock_nbar)

"""list of true parameters"""

data_hod = np.array([11.38 , 0.26 , 12.02 , 1.06 , 13.31])


"""Prior"""

prior = abcpmc.TophatPrior([10.,.1,11.02,.8,13.],[13.,.5,13.02,1.3,14.])
prior_dict = {
 
    'logM0'  : {'shape': 'uniform', 'min': 10.  ,  'max': 13.},
    'sigma_logM': {'shape': 'uniform', 'min': .1 ,  'max': .5},
    'logMmin': {'shape': 'uniform', 'min': 11.02,  'max': 13.02},   
    'alpha': {'shape': 'uniform', 'min': .8 ,  'max': 1.3},
    'logM1'  : {'shape': 'uniform', 'min': 13.  ,  'max': 14.},
}

"""Plot range"""

plot_range = []
for key in ['logM0', 'sigma_logM', 'logMmin','alpha','logM1']: 
	plot_range.append([prior_dict[key]['min'], prior_dict[key]['max']])
prior_range = np.array(plot_range)

"""simulator"""

class HODsim(object): 
    
    def __init__(self): 
        self.model = Zheng07()
    
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
            r , xi_r = self.model.mock.compute_galaxy_clustering()
            return [nbar , xi_r]
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
        
        dist_nbar = (data[0] - model[0])**2. / covar_nbar
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
        labels=[r'$\logM_{0}$',r'$\sigma_{\logM}$',r'$\logM_{min}$',r'$\alpha$',r'$\logM_{1}$'], range=plot_range , quantiles=[0.16,0.5,0.84], show_titles=True, title_args={"fontsize": 12},plot_datapoints=True, fill_contours=True, levels=[0.68, 0.95], color = 'b' , bins =20 , smooth = 1.)

    plt.savefig("/home/mj/public_html/nbar_clustering_Mr20_t"+str(t)+".png")
    plt.close()

    fig = corner.corner(
        theta, truths= data_hod,
        labels=[r'$\logM_{0}$',r'$\sigma_{\logM}$',r'$\logM_{min}$',r'$\alpha$',r'$\logM_{1}$'], range=plot_range , quantiles=[0.16,0.5,0.84], show_titles=True, title_args={"fontsize": 12},plot_datapoints=True, fill_contours=True, levels=[0.68, 0.95], color = 'b' , bins =20 , smooth = 1.)

    plt.savefig("/home/mj/public_html/nbar_clustering_Mr20_now_t"+str(t)+".png")
    plt.close()
    np.savetxt("/home/mj/public_html/nbar_clustering_Mr20_theta_t"+str(t)+".dat" , theta)
    np.savetxt("/home/mj/public_html/nbar_clustering_Mr20_w_t"+str(t)+".dat" , w)


mpi_pool = mpi_util.MpiPool()
def sample(T, eps_val, eps_min):

    abcpmc_sampler = abcpmc.Sampler(N=1000, Y=data, postfn=simz, dist=distance, pool=mpi_pool)
    abcpmc_sampler.particle_proposal_cls = abcpmc.OLCMParticleProposal
    eps = abcpmc.MultiConstEps(T , [1.e13 , 1.e13])
    #eps = abcpmc.MultiExponentialEps(T,[1.e41 , 1.e12] , [eps_min , eps_min])
    pools = []
    for pool in abcpmc_sampler.sample(prior, eps):
        print("T: {0}, ratio: {1:>.4f}".format(pool.t, pool.ratio))
        print eps(pool.t)
        
        plot_thetas(pool.thetas , pool.ws, pool.t)
        
        if (pool.t < 3):
            eps.eps = np.percentile(np.atleast_2d(pool.dists), 50 , axis = 0)
	    #eps.eps = np.mean(np.atleast_2d(pool.dists), axis = 0)
        elif (pool.t > 2)and(pool.t<20):
            eps.eps = np.percentile(np.atleast_2d(pool.dists), 75 , axis = 0)
            abcpmc_sampler.particle_proposal_cls = abcpmc.ParticleProposal
        else:
            eps.eps = np.percentile(np.atleast_2d(pool.dists), 90 , axis = 0)
            abcpmc_sampler.particle_proposal_cls = abcpmc.ParticleProposal
        #for i in xrange(len(eps.eps)):
        #    if eps.eps[i] < eps_min[i]:
        #        eps.eps[i] = eps_min[i]
            
        pools.append(pool)
        
    #abcpmc_sampler.close()
    
    return pools

T=40
eps=1.e9
pools = sample(T, eps, [1., 14.])

