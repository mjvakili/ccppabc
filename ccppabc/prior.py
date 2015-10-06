from scipy.stats import uniform
from scipy.stats import norm

class Prior(object): 

    def __init__(self, prior_dict): 
        """ Class that describes the Prior object

        Parameters
        ----------
        prior_dict : dictionary that specifies the shape and prior distributions  
            (e.g. {'parameter1': {'shape': 'uniform', 'min': 1.0, 'max': 1.2},  ... })

        """

        self.prior_dict = prior_dict.copy()
        
        self.n_params = len(self.prior_dict.keys())   # number of parameters

        self.ordered_keys = prior_dict.keys()
        self.ordered_keys.sort()

        self.priorz = self.prior()

    def prior(self): 
        """ Loops through the parameters
        """

        priorz = [] 

        for key in self.ordered_keys:

            prior_key = self.prior_dict[key]

            if prior_key['shape'] == 'uniform': 

                loc = prior_key['min']
                scale = prior_key['max'] - prior_key['min']

                priorz.append( uniform(loc, scale))

            elif prior_key['shape'] == 'gauss': 
                loc = prior_key['mean']
                scale = prior_key['stddev'] 

                priorz.append( norm(loc, scale) )

            else: 
                raise ValueError("Not specified") 

        return priorz

    def set_prior(self, func, arg_list): 
        """ If you want to set your own priors, snobby motherfucker
        """

        raise NotImplementedError("go fuck yourself")

    def sampler(self): 
        """ Samples the prior distribution to return a vector theta of parameters
        """
    
        theta = np.zeros(self.n_params)

        for i in xrange(self.n_params): 

            np.random.seed()

            theta[i] = self.priorz[i].rvs(size=1)[0]
                                  
        return theta

    def pi_priors(self, theta_in): 
        """ Get the product of the prior probability distributions for each particle 
        represented by theta_in

        $\prod p_i(theta_i)$

        """

        for i in xrange(self.n_params): 
            try:
                p_theta *= self.priorz[i].pdf(theta_in[i])
            except UnboundLocalError: 
                p_theta = self.priorz[i].pdf(theta_in[i])
        
        return p_theta 
