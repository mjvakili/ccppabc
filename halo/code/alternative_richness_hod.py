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


def richness(group_id): 
    gals = Table() 
    gals['groupid'] = group_id
    gals['dummy'] = 1
    grouped_table = gals.group_by('groupid')
    grp_richness = grouped_table['dummy'].groups.aggregate(np.sum)
    return grp_richness



"""Load data and variance"""
data_gmf = np.loadtxt("gmf_Mr20.dat")
sigma = np.loadtxt("gmf_noise_Mr20.dat")
bins = np.loadtxt("gmf_bins_Mr20.dat")

mock_nbar = np.loadtxt("nbar_Mr20.dat")
data_nbar = np.mean(mock_nbar)
covar_nz = np.var(mock_nbar)

data = [data_nbar , data_gmf]

"""True HOD parameters"""

data_hod = np.array([11.38 , 0.26 , 12.02 , 1.06 , 13.31])

"""Prior"""

prior = abcpmc.TophatPrior([10.,.1,11.02,.8,13.],[13.,.5,13.02,1.3,14.])

#prior = abcpmc.TophatPrior([11.91,.38,12.78,1.1,13.9],[11.92,.4,12.8,1.2,14.])

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
print prior_range[:,0] , prior_range[:,1]

"""simulator"""

class HODsim(object): 
    
    def __init__(self): 
        self.model = Zheng07()
    
    def sum_stat(self, theta):
        
        self.model.param_dict['logM0'] = theta[0]
        self.model.param_dict['sigma_logM'] = theta[1]
        self.model.param_dict['logMmin'] = theta[2]     
        self.model.param_dict['alpha'] = theta[3]
        self.model.param_dict['logM1'] = theta[4]
        if np.all((prior_range[:,0] < theta)&(theta < prior_range[:,1])):
          try:
            self.model.populate_mock()
	    nbar = self.model.mock.number_density
            group_id =self. model.mock.compute_fof_group_ids()
            group_richness = richness(group_id)
            y = plt.hist(group_richness , bins)[0] / 250.**3.
            plt.close()
            return [nbar , y]       
          except ValueError:
            return [10. , np.ones_like(bins)[:-1]*1000.]
        else:
            return [10. , np.ones_like(bins)[:-1]*1000.]

ourmodel = HODsim()
simz = ourmodel.sum_stat



def distance(data, model, type = 'multiple distance'): 
    
    if type == 'added distance': 
        dist_nz = np.abs(d_data[0] - d_model[0])/d_data[0]
        dist_xi = np.sum(np.abs(d_data[1] - d_model[1])/d_data[1])
        
        dist = dist_nz + dist_xi 

    elif type == 'multiple distance':
        
        dist_nz = (data[0] - model[0])**2. / covar_nz
        dist_gr = np.sum((data[1] - model[1])**2. / sigma**2.)
        
        dist = np.array([dist_nz , dist_gr])
    elif type == 'chisq distance':

        dist = np.sum((data - model)**2. / sigma **2.)
        
    return dist


def plot_thetas(theta , w , t): 
    fig = corner.corner(
        theta, weights = w.flatten() , truths= data_hod,
        labels=[r'$\logM_{0}$',r'$\sigma_{\logM}$',r'$\logM_{min}$',r'$\alpha$',r'$\logM_{1}$'],
                range=plot_range , quantiles=[0.16,0.5,0.84],
                show_titles=True, title_args={"fontsize": 12},
                plot_datapoints=True, fill_contours=True, levels=[0.68, 0.95], 
                color='b', bins=20, smooth=1.0)
    
    plt.savefig("/home/mj/public_html/nbar_gmf_Mr20_t"+str(t)+".png")
    plt.close()
    fig = corner.corner(
        theta , truths= data_hod,
        labels=[r'$\logM_{0}$',r'$\sigma_{\logM}$',r'$\logM_{min}$',r'$\alpha$',r'$\logM_{1}$'],
                range=plot_range , quantiles=[0.16,0.5,0.84],
                show_titles=True, title_args={"fontsize": 12},
                plot_datapoints=True, fill_contours=True, levels=[0.68, 0.95], 
                color='b', bins=20, smooth=1.0)

    plt.savefig("/home/mj/public_html/nbar_gmf_Mr20_now_t"+str(t)+".png")
    plt.close()
    np.savetxt("/home/mj/public_html/nbar_gmf_Mr20_theta_t"+str(t)+".dat" , theta)
    np.savetxt("/home/mj/public_html/nbar_gmf_Mr20_w_t"+str(t)+".dat" , w)


#alpha = 75
#T = 1
#eps_start = .9
#eps_min = 10.**-6.
mpi_pool = mpi_util.MpiPool()
def sample(T, eps_val, eps_min):
    abcpmc_sampler = abcpmc.Sampler(N = 1200, Y=data, postfn=simz, dist=distance, pool=mpi_pool)
    abcpmc_sampler.particle_proposal_cls = abcpmc.OLCMParticleProposal
    #abcpmc.Sampler.particle_proposal_kwargs = {'k': 50}
    #abcpmc_sampler.particle_proposal_cls = abcpmc.KNNParticleProposal
    eps = abcpmc.ConstEps(T, [1.e13,1.e13])
    pools = []
    for pool in abcpmc_sampler.sample(prior, eps):
        print("T:{0},ratio: {1:>.4f}".format(pool.t, pool.ratio))
        print eps(pool.t)

        plot_thetas(pool.thetas , pool.ws , pool.t)
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
