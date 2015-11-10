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


model = Zheng07(threshold = -21.)
print 'Data HOD Parameters ', model.param_dict

def richness(group_id): 
    gals = Table() 
    gals['groupid'] = group_id
    gals['dummy'] = 1
    grouped_table = gals.group_by('groupid')
    grp_richness = grouped_table['dummy'].groups.aggregate(np.sum)
    return grp_richness

#generate data

model.populate_mock()
group_id = model.mock.compute_fof_group_ids()
data_richness = richness(group_id)
np.savetxt("richness.dat" , data_richness)
nz = np.loadtxt("nz.dat")
covar_nz = np.cov(nz)
avg_nz = np.mean(nz)

data = data_richness
print data.shape
#n_mocks = 1000
#n_bins = 12

#print hist_bins

#histograms = np.zeros((n_mocks , n_bins))

#avg_nz = []

#for i in xrange(n_mocks): 
    
#    model.populate_mock()
    
    # number density
#    avg_nz.append(model.mock.number_density)
    
#     # richness histogram
#     group_id = model.mock.compute_fof_group_ids()
#     group_richness = richness(group_id)
# 
#     #print 'Group Richness computation takes ', time.time() - hod_time, ' seconds'
#     hist, bin_edge = np.histogram(group_richness, bins=hist_bins)
# 
#     #bin_mid = 0.5 * (bin_edge[1:] + bin_edge[:-1])
#    
#     histograms[i,:] = hist

# np.savetxt("group_rich.dat", histograms)

#np.savetxt("nz.dat", avg_nz)

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
print prior_range[:,0] , prior_range[:,1]

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
          #print "lame"
          try:
            self.model.populate_mock()
            group_id =self. model.mock.compute_fof_group_ids()
            group_richness = richness(group_id)
            return group_richness
          except ValueError:
            return np.ones(1000)*1000.
        else:
            return np.ones(1000)*1000.

ourmodel = HODsim()
simz = ourmodel.sum_stat

"""distance"""

def distance(d_data, d_model, type = 'group distance'): 
    
    if type == 'added distance': 
        dist_nz = np.abs(d_data[0] - d_model[0])/d_data[0]
        dist_xi = np.sum(np.abs(d_data[1] - d_model[1])/d_data[1])
        
        dist = dist_nz + dist_xi 

    elif type == 'separate distance':
        
        dist_nz = (d_data[0] - d_model[0])**2. / covar_nz
        dist_xi = np.sum((d_data[1] - d_model[1])**2. * snr_gr)
        
        dist = np.array([dist_nz , dist_xi])
    elif type == 'group distance':

        #dist_nz = (d_data[0] - d_model[0])**2. / covar_nz
        dist_ri = ks_2samp(d_data , d_model)[0]**2. 
        #dist = np.array([dist_nz , dist_ri])
        
    return dist_ri





def plot_thetas(theta , w , t): 
    fig = corner.corner(
        theta, weights = w.flatten() , truths= data_hod,
        truth_color="red", plot_datapoints=True, fill_contours=False, levels=[0.68], 
                color='k', bins=25, smooth= True, 
        range=plot_range, 
        labels=[r"$\log M_{0}$", r"$\sigma_{log M}$", r"$\log M_{min}$" , r"$\alpha$" , r"$\log M_{1}$" ]
        )
    
    plt.savefig("/home/mj/public_html/mpi_t"+str(t)+".png")
    plt.close()
    fig = corner.corner(
        theta , truths= data_hod,
        truth_color="red", plot_datapoints=True, fill_contours=False, levels=[0.68],
                color='k', bins=25, smooth= True,
        range=plot_range,
        labels=[r"$\log M_{0}$", r"$\sigma_{log M}$", r"$\log M_{min}$" , r"$\alpha$" , r"$\log M_{1}$" ]
        )

    plt.savefig("/home/mj/public_html/mpi_now_t"+str(t)+".png")

    np.savetxt("/home/mj/public_html/richness_theta_t"+str(t)+".dat" , theta)
    np.savetxt("/home/mj/public_html/richness_w_t"+str(t)+".dat" , w)


#alpha = 75
#T = 1
#eps_start = .9
#eps_min = 10.**-6.
mpi_pool = mpi_util.MpiPool()
def sample(T, eps_val, eps_min):
    abcpmc_sampler = abcpmc.Sampler(N=500, Y=data, postfn=simz, dist=distance, pool=mpi_pool)
    abcpmc_sampler.particle_proposal_cls = abcpmc.OLCMParticleProposal
    eps = abcpmc.ConstEps(T, eps_val)
    pools = []
    for pool in abcpmc_sampler.sample(prior, eps):
        print("T: {0}, eps: {1:>.4f}, ratio: {2:>.4f}".format(pool.t, eps(pool.t), pool.ratio))
        plot_thetas(pool.thetas , pool.ws,pool.t)
        
        eps.eps = np.percentile(pool.dists, 75)
        if eps.eps < eps_min:
            eps.eps = eps_min
            
        pools.append(pool)
        
    #abcpmc_sampler.close()
    
    return pools

T=40
eps=3.
pools = sample(T, eps, 1.e-16)

#print pools[-1]

#mpi_pool = mpi_util.MpiPool()
#sampler = abcpmc.Sampler(N=10, Y=data, postfn=simz, dist=distance, thre)
#sampler.particle_proposal_cls = abcpmc.ParticleProposal




"""
if mpi_pool.isMaster():
       print("start sampling")    
       pools = []
for pool in sampler.sample(prior, eps):
       ##print pool
       #eps.eps = mpi_util.mpiBCast(pool.eps)
       ##print("T: {0}, eps: {1:>.4f}, ratio: {2:>.4f}".format(pool.t, pool.eps, pool.ratio))

          #for i, (mean, std) in enumerate(zip(np.mean(pool.thetas, axis=0), np.std(pool.thetas, axis=0))):
          #    print(u"    theta[{0}]: {1:>.4f} \u00B1 {2:>.4f}".format(i, mean,std))

       eps.eps = np.percentile(pool.dists, alpha) # reduce eps value
       if mpi_pool.isMaster():
         pools.append(pool)
         print "apped" 

"""
"""   
if mpi_pool.isMaster():
      print pools
      thetas = np.vstack([pool.thetas for pool in pools])
      ws = np.vstack([pool.ws for pool in pools])
      dists = np.vstack([pool.dists for pool in pools])
      time = pools.t
      plot_thetas(thetas , ws , time)
"""     



 #print("T: {0}, eps: {1:>.4f}, ratio: {2:>.4f}".format(pool.t, pool.eps, pool.ratio))

