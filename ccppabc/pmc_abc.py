"""
PMC-ABC code

Important comment : we have written the sampling in a way that you can turn off
multiprocessing by setting the parameter N_threads to 1 and run the code
in serial. This allows you to troubleshoot the fuck out of your simulator,
or distance metric if there is anything wrong with them in which case you
should be ashamed of yourself.
Rule of thumb: First run the code in serial to make sure everything is fine,
and then take advantage of multiprocessing.
"""

from scipy.stats import multivariate_normal
from prior import Prior
from my_simulator import model
from distances import my_dist
from interruptible_pool import InterruptiblePool
import utils
from plot import plot_thetas

"""Wrapper functions for multiprocessing"""

def unwrap_self_initial_pool_sampling(arg, **kwarg):
    return ABC.initial_pool_sampling(*arg, **kwarg)

def unwrap_self_importance_pool_sampling(arg, **kwarg):
    return ABC.importance_pool_sampling(*arg, **kwarg)



class ABC(object):

   def __init__(self, data, model, distance, prior_obj, eps0, N_threads = 1 , N_particles, T=20):
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
       self.N_particles = N_particles
       self.T = T

    def initial_pool_sampling(self, i_particle):
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
    	pool_list.append(1./np.float(self.N_particles))
    	pool_list.append(rho)

    	return pool_list

    def initial_pool(self):

    	if self.N_threads == 1 :

            args_list = [i for i in xrange(self.N_particles)]
	    results   = []
            for arg in args_list:
                results.append(initial_pool_sampling(arg)
        else:
            pool = InterruptiblePool(processes = self.N_threads)
            mapfn = pool.map
            args_list = [i for i in xrange(self.N_particles)]
            results = mapfn(unwrap_self_initial_pool_sampling,
                            zip([self]*len(args_list), args_list))
	    pool.close()
            pool.terminate()
            pool.join()

        n_params = len(self.prior_obj.prior_dict)

        results = np.array(results).T
        self.theta_t = results[1:n_params+1,:]
        self.w_t = results[n_params+1,:]
        self.rhos = results[n_params+2,:]
        self.sig_t = utils.covariance(theta_t , w_t)

        plot_thetas(self.theta_t, self.w_t, 0, self.prior_obj.prior_dict)


    def importance_pool_sampling(self, arg):
        # args = [i_particle, theta_t_1, w_t_1, sig_t_1, eps_t]
        i_particle = arg

        np.random.seed()
        rho = 1e37

        # perturbed theta (Double check)
        while rho > self.eps_t:

            theta_star = utils.weighted_sampling(self.theta_t, self.w_t)
            theta_starstar = multivariate_normal(theta_star,
                                                 self.sig_t).rvs(size=1)
            model_starstar = self.model(theta_starstar)

            rho = self.distance(self.data, model_starstar)

        p_theta = self.prior_obj.pi_priors(theta_starstar)

        w_starstar = p_theta / np.sum(self.w_t * \
                                      utils.better_multinorm(theta_starstar,
                                                             self.theta_t,
                                                             self.sig_t) )

        pool_list = [np.int(i_particle)]
        for i_p in xrange(n_params):
            pool_list.append(theta_starstar[i_p])
        pool_list.append(w_starstar)
        pool_list.append(rho)

        return pool_list


    def run_abc():

        self.initial_pool()

        while t < self.T:

            self.eps_t = np.percentile(self.rhos, 75)

            if self.N_threads == 1 :

                args_list = [i for i in xrange(self.N_particles)]
                results   = []
                for arg in args_list:
                    results.append(self.importance_pool_sampling(arg)
            else:
                pool = InterruptiblePool(processes = self.N_threads)
                mapfn = pool.map
                args_list = [i for i in xrange(self.N_particles)]
                results = mapfn(unwrap_self_importance_pool_sampling,
                                zip([self]*len(args_list), args_list))
                pool.close()
                pool.terminate()
                pool.join()

            results = np.array(results).T

        self.theta_t = results[1:n_params+1,:]
        self.w_t = results[n_params+1,:]
        self.rhos = results[n_params+2,:]
        self.sig_t = utils.covariance(theta_t , w_t)

        plot_thetas(self.theta_t, self.w_t, t, self.prior_obj.prior_dict)









