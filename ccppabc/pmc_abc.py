"""

PMC-ABC code 

"""
from prior import Prior

def pmc_abc(data, model, distance):
    """
    """

    # initial pool 
    pass


def initial_pool_sampling(i_particle, eps0, data, model, distance, prior_obj, cpu_count=1): 
    """ Sample theta_star from prior distribution for the initial pool 
    returns [i_particle, theta_star, weights, rho]

    Parameters
    ----------
    i_particle : index of particle
    prior_obj : Prior class object

    """

    theta_star = prior_obj.sampler()
    n_param = len(theta_star)

    model_theta = model(theta_star)

    rho = distance(data, model_theta)

    while rho > eps0: 
        theta_star = prior_obj.sampler()
        model_theta = model(theta_star)

        rho = distance(data, model_theta) 

    pool_list = [np.int(i_particle)] 
    for i_param in xrange(n_params): 
        pool_list.append(theta_star[i_param])
        pool_list.append(1./np.float(N_particles))
        pool_list.append(rho)
                                                                                                                    
    return np.array(pool_list)
