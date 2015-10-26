import numpy as np
import matplotlib.pyplot as plt
from interruptible_pool import InterruptiblePool
import time
plt.switch_backend("Agg")
from halotools.empirical_models import Zheng07
from astropy.table import Table
import corner 
#import seaborn as seabreeze
from scipy.stats import multivariate_normal

model = Zheng07(threshold = -21.)
print 'Data HOD Parameters ', model.param_dict


N_threads = 10 
N_particles = 500 
N_iter = 20
eps0 = np.array([1.e2])# , 1.e34, 1.e34, 1.e34, 1.e34, 1.e34])

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
data = richness(group_id)

n_mocks = 1000
n_bins = 12
hist_bins = np.rint(3 * np.logspace(0., 1.2, n_bins+1 ))
hist_bins[-1] = 10000
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

"""data and covariance """
"""
nz = np.loadtxt("nz.dat")
histograms = np.loadtxt("group_rich.dat")

covar_nz = np.cov(nz)
covar_gr = np.cov(histograms)

#print covar_nz
#print np.diag(covar_gr)

snr_gr   = 1./np.diag(covar_gr)

avg_nz = np.mean(nz)
avg_gr  = np.mean(histograms.T , axis = 0)

data = [avg_nz, avg_gr]

#alpha , logMmin , sigma_logM , logM0 , logM1
"""
data_hod = np.array([11.92 , 0.39 , 12.79 , 1.15 , 13.94])
"""simulator"""

class HODsim(object): 
    
    def __init__(self): 
        self.model = Zheng07(threshold = -21.)
    
    def sum_stat(self, theta_star):

	#print theta_star
        
        self.model.param_dict['alpha'] = theta_star[3]
        self.model.param_dict['logMmin'] = theta_star[2]     
        self.model.param_dict['sigma_logM'] = theta_star[1]
        self.model.param_dict['logM0'] = theta_star[0]
        self.model.param_dict['logM1'] = theta_star[4]
        #print self.model.param_dict
        #a = time.time()
        try:
            self.model.populate_mock()
            #print "pop time", time.time() - a
            #a = time.time()
            #nz = self.model.mock.number_density
            #print "nz time" , time.time() - a
            #hist = np.zeros((12))
            
            #a = time.time()
            group_id =self. model.mock.compute_fof_group_ids()
            #print "fof time" , time.time() - a
            #a = time.time()
            group_richness = richness(group_id)
            #print "rich time" , time.time() - a
            #a = time.time()
            #hist_temp, bin_edge = np.histogram(group_richness, bins=hist_bins)
    	#print hist , hist_temp
            #hist += hist_temp
            #self.model.populate_mock()          
            return group_richness
        except ValueError:
            return np.zeros(1000)

ourmodel = HODsim()
simz = ourmodel.sum_stat
from scipy.stats import ks_2samp
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

        dist = 1. - ks_2samp(d_data , d_model)[1] 
        
        
    return np.atleast_1d(dist)

"""covariance matrix in abc sampler"""

def covariance(theta , w , type = 'weighted'):

    if type == 'neutral':

      return np.cov(theta)

    if type == 'weighted':
      ww = w.sum() / (w.sum()**2 - (w**2).sum()) 
      mean = np.sum(theta*w[None,:] , axis = 1)/ np.sum(w)
      tmm  = theta - mean.reshape(theta.shape[0] , 1)
      sigma2 = ww * (tmm*w[None,:]).dot(tmm.T)
      
      return np.diag(np.diag(sigma2))  

"""Prior"""

from scipy.stats import uniform
from scipy.stats import norm 
class Prior(object): 
    def __init__(self, prior_dict): 
        self.prior_dict = prior_dict.copy()
    
    def prior(self): 
        priorz = [] 
        for key in self.prior_dict.keys(): 
            
            prior_key = self.prior_dict[key]
            
            if prior_key['shape'] == 'uniform': 
                
                loc = prior_key['min']
                scale = prior_key['max'] - prior_key['min']
                
                priorz.append( uniform(loc, scale))
            
            elif prior_key['shape'] == 'gauss':
                
                loc = prior_key['mean']
                scale = prior_key['stddev']
                
                priorz.append( norm(loc, scale) )
                
        return priorz


prior_dict = {
 
    'logM0'  : {'shape': 'uniform', 'min': 9.  ,  'max': 15.},
    'sigma_logM': {'shape': 'uniform', 'min': 0. ,  'max': 1.},
    'logMmin': {'shape': 'uniform', 'min': 12.5,  'max': 13.09},   
    'alpha': {'shape': 'uniform', 'min': .9 ,  'max': 1.45},
    'logM1'  : {'shape': 'uniform', 'min': 13.6  ,  'max': 14.25},
}
n_params = len(prior_dict.keys())
prior_obj = Prior(prior_dict) 

def prior_sampler(): 
    """ Sample prior distribution and return theta_star 
    """
    theta_star = np.zeros(n_params)
    
    for i in xrange(n_params): 
        np.random.seed()
        theta_star[i] = prior_obj.prior()[i].rvs(size=1)[0]
        
    return theta_star

def pi_priors(tmp_theta): 
    for i in xrange(n_params): 
        try:
            p_theta *= prior_obj.prior()[i].pdf(tmp_theta[i])
        except UnboundLocalError: 
            p_theta = prior_obj.prior()[i].pdf(tmp_theta[i])
            
    return p_theta 


def weighted_sampling(theta, w): 

    #w_cdf = w.cumsum()/w.sum() # normalized CDF
    #np.random.seed()
    #rand1 = np.random.random(1)
    #cdf_closest_index = np.argmin( np.abs(w_cdf - rand1) )
    #closest_theta = theta[:, cdf_closest_index]
    
    index = np.random.choice(range(N_particles), 1, p = w/np.sum(w))[0]
    closest_theta = theta[:,index] 
    
    return closest_theta

from scipy import spatial

def knn_sigma(theta, k = 3):
        tree = spatial.cKDTree(theta.T)
        _, idxs = tree.query(theta.T , k , p=2)
        sigma = np.cov(theta[: , idxs])
        return sigma

def better_multinorm(theta_stst, theta_before, cov): 
    n_par, n_part = theta_before.shape
    
    sig_inv = np.linalg.inv(cov)
    x_mu = theta_before.T - theta_stst

    nrmliz = 1.0 / np.sqrt( (2.0*np.pi)**n_par * np.linalg.det(cov))

    multinorm = nrmliz * np.exp(-0.5 * np.sum( (x_mu.dot(sig_inv[None,:])[:,0,:]) * x_mu, axis=1 ) )

    return multinorm

prior_range = []
for key in ['logM0', 'sigma_logM', 'logMmin','alpha','logM1']: 
	prior_range.append([prior_dict[key]['min'], prior_dict[key]['max']])

plot_range = prior_range
prior_range = np.array(prior_range)
print "prior range is = "  , prior_range

def plot_thetas(theta , w , t): 
    fig = corner.corner(
        theta.T, weights = w.flatten() , truths= data_hod,
        truth_color="red", plot_datapoints=True, fill_contours=False, levels=[0.68], 
                color='k', bins=25, smooth= True, 
        range=plot_range, 
        labels=[r"$\log M_{0}$", r"$\sigma_{log M}$", r"$\log M_{min}$" , r"$\alpha$" , r"$\log M_{1}$" ]
        )
    
    plt.savefig("/home/mj/public_html/knn5_hod5_flat_t"+str(t)+".png")
    plt.close()
    np.savetxt("/home/mj/public_html/knn5_hod5_flat_t"+str(t)+".dat" , theta.T)
    
    np.savetxt("/home/mj/public_html/knn5_hod5_flat_t"+str(t)+".dat" , w.T)




def initial_pool_sampling(i_particle): 
    """ Sample theta_star from prior distribution for the initial pool
    """
    rho = eps0 + 1.
    while np.all(rho < eps0)==False:
        
        theta_star = prior_sampler()
        model_theta = simz(theta_star)
        rho = distance(data, model_theta)
        
    pool_list = [np.int(i_particle)]
    for i_param in xrange(n_params): 
        pool_list.append(theta_star[i_param])
        
    pool_list.append(1./np.float(N_particles))
    for r in rho:
        pool_list.append(r)
    
    return np.array(pool_list)


def initial_pool():

    args_list = np.arange(N_particles)
    """serial"""
    #results = [] 
    #for arg in args_list:  	
    #    results.append(initial_pool_sampling(arg))
    """parallel"""
    pool = InterruptiblePool(processes = N_threads)
    mapfn = pool.map
    results = mapfn(initial_pool_sampling, args_list)
    pool.close()
    pool.terminate()
    pool.join()
    
    results = np.array(results).T
    theta_t = results[1:n_params+1,:]
    w_t = results[n_params+1,:]
    rhos = results[n_params+2:,:]
    #sig_t = knn_sigma(theta_t , k = 10)
    sig_t = covariance(theta_t , w_t)  
    return theta_t, w_t, rhos, sig_t


def importance_pool_sampling(args): 
    
    i_particle = args[0]
    theta_t_1 = args[1]
    w_t_1 = args[2]
    sig_t_1 = args[3]
    eps_t = args[4]
    
    rho = 1.e100    
    while np.all(rho < eps_t)==False:
        
        theta_star = weighted_sampling(theta_t_1, w_t_1)
        theta_starstar = multivariate_normal( theta_star, sig_t_1 ).rvs(size=1)
        #print theta_starstar
        #print prior_range
        #print np.all((prior_range[:,0] < theta_starstar)&(theta_starstar < prior_range[:,1]))        
        while np.all((prior_range[:,0] < theta_starstar)&(theta_starstar < prior_range[:,1]))==False:
          
              theta_star = weighted_sampling(theta_t_1, w_t_1)
              np.random.seed()
              theta_starstar = multivariate_normal( theta_star, sig_t_1 ).rvs(size=1)
        
        model_starstar = simz(theta_starstar)
        rho = distance(data, model_starstar)
    
    p_theta = pi_priors(theta_starstar)
    w_starstar = p_theta/np.sum( w_t_1 * better_multinorm(theta_starstar, theta_t_1, sig_t_1) )    
    
    pool_list = [np.int(i_particle)]
    for i_p in xrange(n_params): 
        pool_list.append(theta_starstar[i_p])
    pool_list.append(w_starstar)
    for r in rho:
        pool_list.append(r)
    
    return pool_list 
    
def pmc_abc(N_threads = N_threads): 
    
    # initial pool
    theta_t, w_t, rhos, sig_t = initial_pool()
    t = 0 # iternation number
    
    plot_thetas(theta_t , w_t, t)
    
    
    while t < N_iter: 
        if t < 4 :
           eps_t = np.percentile(np.atleast_2d(rhos), 50, axis=1)
        else:
           eps_t = np.percentile(np.atleast_2d(rhos), 75, axis=1)
        print 'New Distance Threshold Eps_t = ', eps_t , "t=" , t
        
        theta_t_1 = theta_t.copy()
        w_t_1 = w_t.copy()
        sig_t_1 = sig_t.copy()
    

        args_list = [[i, theta_t_1, w_t_1, sig_t_1, eps_t] for i in xrange(N_particles)]
        """serial"""
        #results = [] 
        #for args in args_list: 
        #    pool_sample = importance_pool_sampling(args)
        #    results.append( pool_sample )
        """parallel"""
        pool = InterruptiblePool(processes = N_threads)
        mapfn = pool.map
        results = mapfn(importance_pool_sampling, args_list)
        pool.close()
        pool.terminate()
        pool.join()
        
        results = np.array(results).T
        theta_t = results[1:n_params+1,:]
        w_t = results[n_params+1,:]
        rhos = results[n_params+2:,:]
        #sig_t = knn_sigma(theta_t , k = 10)
        sig_t = covariance(theta_t , w_t) 
        t += 1
        
        plot_thetas(theta_t, w_t , t)
pmc_abc()
