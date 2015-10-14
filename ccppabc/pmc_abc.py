"""
PMC-ABC code

you can turn off
multiprocessing by setting the parameter N_threads to 1 and run the code
in serial. This allows you to troubleshoot your simulator,
or distance metric if there is anything wrong with them.
Rule of thumb: First run the code in serial to make sure everything is fine,
and then take advantage of multiprocessing.
"""

import numpy as np
import utils
from plot import plot_thetas
from multiprocessing.pool import Pool
from prior import Prior

class _Initialpoolsampling(object):

    def __init__(self, eps_0, data, model, distance, prior_dict):

        
        self.data = data
	self.model = model
	self.distance = distance
	self.prior_dict = prior_dict
	self.eps_0 = eps_0
        self.n_params = len(self.prior_dict)
 
    def __call__(self, i_particle):
        
        rho = 1.e37
        prior_obj = Prior(self.prior_dict)
        while rho > self.eps_0:
            theta_star = prior_obj.sampler()
            model_theta = self.model(theta_star)
            rho = self.distance(self.data, model_theta)

        pool_list = [np.int(i_particle)]
        for i_param in xrange(self.n_params):
             pool_list.append(theta_star[i_param])
        pool_list.append(1.)
        pool_list.append(rho)
        
        return pool_list



class _Importancepoolsampling(object):

    def __init__(self, eps_t, data, model, distance, prior_dict, theta_t, w_t, sig_t):

        self.data = data
	self.model = model
	self.distance = distance
	self.prior_dict = prior_dict
	self.eps_t = eps_t
	self.theta_t = theta_t
	self.w_t = w_t
	self.sig_t = sig_t
        self.n_params = len(self.prior_dict)

    def __call__(self , i_particle):

        rho = 1.e37
        prior_obj = Prior(self.prior_dict)
        while rho > self.eps_t:
           theta_star = utils.weighted_sampling(self.theta_t, self.w_t)
           np.random.seed()
           theta_starstar = np.random.multivariate_normal(theta_star, self.sig_t, 1)[0]
           #theta_starstar = multivariate_normal(theta_star, self.sig_t).rvs(size=1)
           model_starstar = self.model(theta_starstar)
           rho = self.distance(self.data, model_starstar)

        p_theta = prior_obj.pi_priors(theta_starstar)
        w_starstar = p_theta / np.sum(self.w_t * \
                                      utils.better_multinorm(theta_starstar,
                                                             self.theta_t,
                                                             self.sig_t) )

        pool_list = [np.int(i_particle)]
        theta_starstar = np.atleast_1d(theta_starstar)
        for i_param in xrange(self.n_params):
           pool_list.append(theta_starstar[i_param])
           pool_list.append(w_starstar)
           pool_list.append(rho)

        return pool_list

class ABC(object):

    def __init__(self, data, model, distance, prior_dict, eps0 , N_threads=1,
                 N_particles=100, T=20, basename="abc_run"):
        """
        eps0       : initial threshold
        data       : observation or the summary statistcs of the observations
        model      : forward model (simulator function)  of the observation. It takes one particle "theta"
                     as an input and returns the value of the forward model for that particle.
        distance   : distance function between the data and the model evaluated at a given theta
        prior_dict : dictionary of model parameters and their priors
        """

        self.data = data
        self.model = model
        self.distance = distance
        self.prior_dict = prior_dict
        self.eps0 = eps0
        self.N_threads = N_threads
        self.N_particles = N_particles
        self.T = T
        self.basename = basename
        self.n_params = len(self.prior_dict)

        #if self.N_threads != 1 :


    def initial_pool(self):

        
        initial_wrapper = _Initialpoolsampling(self.eps0 , self.data, self.model, \
                                               self.distance , self.prior_dict)
                                         
        #print wrapper
        #print wrapper(1)

        if self.N_threads == 1:
            results   = []
            for i in range(self.N_particles):

                results.append(initial_wrapper(i))
        else:

            self.pool = Pool(self.N_threads)
            self.mapfn  = self.pool.map
            results = self.mapfn(initial_wrapper , range(self.N_particles))
	    self.pool.close()
            #self.pool.terminate()
            #self.pool.join()

        results = np.array(results).T
        self.theta_t = results[1:self.n_params+1,:]
        self.w_t = results[self.n_params+1,:]
        self.rhos = results[self.n_params+2,:]
        self.sig_t = 2. * utils.covariance(self.theta_t , self.w_t)

        plot_thetas(self.theta_t, self.w_t, self.prior_dict, 0,
                    basename=self.basename)

    def run_abc(self):
        """
        Run PMC_ABC
        """

        self.initial_pool()
        t = 1

        
        while t < self.T:
            
            self.eps_t = np.percentile(self.rhos, 75)
            print "iteration= " , t , "threshold= " , self.eps_t
           
            importance_wrapper = _Importancepoolsampling(self.eps_t, self.data, self.model, \
                                                         self.distance, self.prior_dict, \
                                                         self.theta_t, self.w_t, self.sig_t) 

            if self.N_threads == 1 :
                results   = []
                for i in range(self.N_particles):

                    results.append(importance_wrapper(i))
            else:

                self.pool = Pool(self.N_threads)
                self.mapfn  = self.pool.map
                results = self.mapfn(importance_wrapper, range(self.N_particles))
                self.pool.close()
                #self.pool.terminate()
                #self.pool.join()

            results = np.array(results).T

            self.theta_t = results[1:self.n_params+1,:]
            self.w_t = results[self.n_params+1,:]
            self.rhos = results[self.n_params+2,:]
            self.sig_t = 2. * utils.covariance(self.theta_t, self.w_t)

            plot_thetas(self.theta_t, self.w_t, self.prior_dict, t,
                        basename=self.basename,
                        fig_name="{0}_{1}.png".format(self.basename, str(t)))

            t += 1
