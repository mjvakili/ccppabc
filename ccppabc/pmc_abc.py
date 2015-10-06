"""

PMC-ABC code 

"""
from prior import Prior
from my_simulator import model


"""covariance matrix in abc sampler"""

def covariance(theta , w , type = 'weighted'):

    if type == 'neutral':

      return np.cov(theta)

    if type == 'normalized neutral':

      return np.corrcoef(theta)

    if type == 'weighted':
      mean = np.sum(theta*w[None,:] , axis = 1)/ np.sum(w)
      tmm  = theta - mean.reshape(theta.shape[0] , 1)
      sigma2 = 1./(w.sum()) * (tmm*w[None,:]).dot(tmm.T)
      return sigma2  



def pmc_abc(data, model, distance):
    """
    """

    # initial pool 
    pass


class ABC(object):

   def __init__(self, data, model, distance, prior_obj, eps0, N_threads = 1):
       """

       eps0       : initial threshold
       data       : observation or the summary statistcs of the observations
       model      : forward model (simulator function)  of the observation. It takes one particle "theta"
                 as an input and returns the value of the forward model for that particle.
       distance   : distance function between the data and the model evaluated at a given theta
 
       prior_obj  : Prior class object
       """
       self.data = data
       self.model = model
       self.distance = distance
       self.prior_obj = prior_obj
       self.eps0 = eps0
       self.N_threads = N_threads	


    def initial_pool_sampling(self , i_particle):
        """ Sample theta_star from prior distribution for the initial pool 
        returns [i_particle, theta_star, weights, rho]

        Parameters
    	----------
    	i_particle : index of particle

    	returns    : particle , weight, and its corresponding distance 

    	"""

    	theta_star = self.prior_obj.sampler()
    	n_param = len(theta_star)

    	model_theta = self.model(theta_star)

    	rho = self.distance(self.data, model_theta)

        while rho > self.eps0:
    	    theta_star = self.prior_obj.sampler()
       	    model_theta = self.model(theta_star)

            rho = distance(self.data, model_theta)

    	pool_list = [np.int(i_particle)]
    	for i_param in xrange(n_params):
            pool_list.append(theta_star[i_param])
    	pool_list.append(1./np.float(N_particles))
    	pool_list.append(rho)

    	return pool_list

    def initial_pool(N_particles , N_threads = 1):

    
    	n_params = len(theta_star)
    	if N_threads == 1 :
       
            args_list = [i for i in xrange(N_particles)]
	    results   = []
            for arg in args_list:
                results.append(initial_pool_sampling(arg)
        else:
            pool = InterruptiblePool(processes = N_threads)
            mapfn = pool.map
            args_list = [i for i in xrange(N_particles)]
            results = mapfn(initial_pool_sampling, args_list)
            pool.close()
            pool.terminate()
            pool.join()
    
        results = np.array(results).T
        theta_t = results[1:n_params+1,:]
        w_t = results[n_params+1,:]
    rhos = results[n_params+2,:]
    sig_t = covariance(theta_t , w_t)
    
    return theta_t, w_t, rhos, sig_t

