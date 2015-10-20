import numpy as np
import abcpmc
import matplotlib.pyplot as plt
from interruptible_pool import InterruptiblePool
import time
plt.switch_backend("Agg")
from halotools.empirical_models import Zheng07

from astropy.table import Table
model = Zheng07(threshold = -21.)
print 'Data HOD Parameters ', model.param_dict



def richness(group_id): 
    gals = Table() 
    gals['groupid'] = group_id
    gals['dummy'] = 1
    grouped_table = gals.group_by('groupid')
    grp_richness = grouped_table['dummy'].groups.aggregate(np.sum)
    return grp_richness




n_mocks = 1000
n_bins = 12
hist_bins = np.rint(3 * np.logspace(0., 1.2, n_bins+1 ))
hist_bins[-1] = 10000
print hist_bins

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

#print "data=" , data

data_hod = np.array([1.15 , 0.39, 12.79])

"""simulator"""

class HODsim(object): 
    
    def __init__(self): 
        self.model = Zheng07(threshold = -21.)
    
    def sum_stat(self, theta_star):

	#print theta_star
        
        self.model.param_dict['alpha'] = theta_star[0]
        self.model.param_dict['logMmin'] = theta_star[2]     
        self.model.param_dict['sigma_logM'] = theta_star[1]     

	#print self.model.param_dict
        self.model.populate_mock()

        nz = self.model.mock.number_density
        group_id =self. model.mock.compute_fof_group_ids()
        group_richness = richness(group_id)
        
        
        hist, bin_edge = np.histogram(group_richness, bins=hist_bins)
        
        return [nz, hist]

ourmodel = HODsim()
simz = ourmodel.sum_stat

"""distance"""

def distance(d_data, d_model, type = 'separate distance'): 
    
    if type == 'added distance': 
        dist_nz = np.abs(d_data[0] - d_model[0])/d_data[0]
        dist_xi = np.sum(np.abs(d_data[1] - d_model[1])/d_data[1])
        
        dist = dist_nz + dist_xi 

    elif type == 'separate distance':
        
        dist_nz = np.sqrt((d_data[0] - d_model[0])**2. / covar_nz )
        dist_xi = np.sqrt(np.sum((d_data[1] - d_model[1])**2. * snr_gr))
        
        dist = np.array([dist_nz , dist_xi]) 
        
        
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
      
      return sigma2  


def KNN_covariance(theta , w):

    return np.cov(theta)

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
    'alpha': {'shape': 'uniform', 'min': 1.05,  'max': 1.25},
    'm_min'  : {'shape': 'uniform', 'min': 12.5,  'max': 13.},   
    'sigma': {'shape': 'uniform', 'min': 0.3,  'max': .5},
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



import corner 
import seaborn as seabreeze
plot_range = []
for key in ['alpha', 'sigma', 'm_min']: 
	plot_range.append([prior_dict[key]['min'], prior_dict[key]['max']])

print "prior range is = " plot_range

def plot_thetas(theta , w , t): 
    fig = corner.corner(
        theta.T, weights = w.flatten() , truths= data_hod,
        truth_color="red", plot_datapoints=True, fill_contours=False, levels=[0.68, 0.95], 
                color='b', bins=40, smooth=1.0, 
        range=plot_range, 
        labels=[r"$\alpha$", r"$\sigma$", r"$\log M_{min}$" ]
        )
    
    plt.savefig("/home/mj/public_html/weighted_l2_scatter_hod3_flat_t"+str(t)+".png")
    plt.close()
    np.savetxt("/home/mj/public_html/weighted_l2_theta_hod3_flat_t"+str(t)+".dat" , theta.T)
    
    np.savetxt("/home/mj/public_html/weighted_l2_w_hod3_flat_t"+str(t)+".dat" , w.T)


N_threads = 16 
N_particles = 200 
N_iter = 20
eps0 = np.array([1.e34 , 1.e34])


def initial_pool_sampling(i_particle): 
    """ Sample theta_star from prior distribution for the initial pool
    """
    theta_star = prior_sampler()
    model_theta = simz(theta_star)
    
    rho = distance(data, model_theta)
    
    while np.all(rho > eps0):
        
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

    pool = InterruptiblePool(processes = N_threads)
    mapfn = pool.map
    args_list = [i for i in xrange(N_particles)]
    #results = [] 
    #for arg in args_list:  	results.append(initial_pool_sampling(arg))
    results = mapfn(initial_pool_sampling, args_list)
    
    pool.close()
    pool.terminate()
    pool.join()
    
    results = np.array(results).T
    theta_t = results[1:n_params+1,:]
    w_t = results[n_params+1,:]
    rhos = results[n_params+2:,:]
    sig_t = covariance(theta_t , w_t)
    
    return theta_t, w_t, rhos, sig_t


def weighted_sampling(theta, w): 
    """ Given array of thetas and their corresponding weights, sample
    """
    w_cdf = w.cumsum()/w.sum() # normalized CDF
    
    np.random.seed()
    rand1 = np.random.random(1)
    cdf_closest_index = np.argmin( np.abs(w_cdf - rand1) )
    closest_theta = theta[:, cdf_closest_index]
    
    return closest_theta

def better_multinorm(theta_stst, theta_before, cov): 
    n_par, n_part = theta_before.shape
    
    sig_inv = np.linalg.inv(cov)
    x_mu = theta_before.T - theta_stst

    nrmliz = 1.0 / np.sqrt( (2.0*np.pi)**n_par * np.linalg.det(cov))

    multinorm = nrmliz * np.exp(-0.5 * np.sum( (x_mu.dot(sig_inv[None,:])[:,0,:]) * x_mu, axis=1 ) )

    return multinorm


from scipy.stats import multivariate_normal 

def importance_pool_sampling(args): 
    # args = [i_particle, theta_t_1, w_t_1, sig_t_1, eps_t]
    i_particle = args[0]
    theta_t_1 = args[1]
    w_t_1 = args[2]
    sig_t_1 = args[3]
    eps_t = args[4]
    
    theta_star = weighted_sampling(theta_t_1, w_t_1)
    
    np.random.seed()
    # perturbed theta (Double check)    
    theta_starstar = multivariate_normal( theta_star, sig_t_1 ).rvs(size=1)
    model_starstar = simz(theta_starstar)
    
    rho = distance(data, model_starstar)
    
    while np.all(rho > eps_t):
        theta_star = weighted_sampling(theta_t_1, w_t_1)
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
        
        eps_t = np.percentile(np.atleast_2d(rhos), 75, axis=1)
        print 'New Distance Threshold Eps_t = ', eps_t
        
        theta_t_1 = theta_t.copy()
        w_t_1 = w_t.copy()
        sig_t_1 = sig_t.copy()
    
        """these lines are borrowed from initial sampling to double-check multiprocessing"""
        #pool = InterruptiblePool(processes = N_threads)
    	#mapfn = pool.map
    	#args_list = [i for i in xrange(N_particles)]
    	#results = mapfn(initial_pool_sampling, args_list)
    
    	#pool.close()
    	#pool.terminate()
    	#pool.join()

        pool = InterruptiblePool(processes = N_threads)
        mapfn = pool.map
        args_list = [[i, theta_t_1, w_t_1, sig_t_1, eps_t] for i in xrange(N_particles)]
        #results = [] 
        #for args in args_list: 
        #    pool_sample = importance_pool_sampling(args)
        #    results.append( pool_sample )
        results = mapfn(importance_pool_sampling, args_list)
        pool.close()
        pool.terminate()
        pool.join()
        
                 
        results = np.array(results).T
        theta_t = results[1:n_params+1,:]
        w_t = results[n_params+1,:]
        rhos = results[n_params+2:,:]
        sig_t = covariance(theta_t , w_t)
        
        t += 1
        
        plot_thetas(theta_t, w_t , t)
pmc_abc()

